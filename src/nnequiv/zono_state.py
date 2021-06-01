import copy

import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.network import NeuralNetwork, ReluLayer
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from nnequiv.global_state import GLOBAL_STATE


class BranchDecision:
	BOTH = 0
	POS = 1
	NEG = 2

	def __init__(self, cur_network, cur_layer, index, decision):
		self.cur_network = cur_network
		self.cur_layer = cur_layer
		self.index = index
		self.decision = decision

	def __repr__(self):
		return str((self.cur_network, self.cur_layer, self.index, self.decision))

	def __str__(self):
		return self.__repr__()

	def __lt__(self, other):
		return self.cur_network < other.cur_network \
		       or (self.cur_network == other.cur_network and self.cur_layer < other.cur_layer) \
		       or (
				       self.cur_network == other.cur_network and self.cur_layer == other.cur_layer and self.index < other.index)

	def __gt__(self, other):
		return not self.__lt__(other) and not self.__eq__(other)

	def __eq__(self, other):
		return self.cur_network == other.cur_network and self.cur_layer == other.cur_layer and self.index == other.index


class LayerBounds:
	def __init__(self):
		self.output_bounds = None
		self.branching_neurons = None

	def clear_output_bounds(self):
		self.output_bounds = None
		self.branching_neurons = None

	def remaining_splits(self):
		if self.branching_neurons is not None:
			return len(self.branching_neurons)
		else:
			return 0

	def pop_branch(self):
		if self.branching_neurons is None or len(self.branching_neurons) <= 0:
			return None
		rv = self.branching_neurons[0]
		self.branching_neurons = self.branching_neurons[1:]
		return rv

	def process_layer(self, zono, start_with=0):
		if self.output_bounds is None or start_with != 0:
			Timers.tic('layer_bounds_process_layer')
			self.output_bounds = zono.box_bounds()
			new_zeros = None
			if self.branching_neurons is not None:
				new_zeros = self.branching_neurons[
					self.output_bounds[self.branching_neurons, 1] < -Settings.SPLIT_TOLERANCE]

			self.branching_neurons = \
				np.nonzero(np.logical_and(self.output_bounds[start_with:, 0] < -Settings.SPLIT_TOLERANCE,
				                          self.output_bounds[start_with:, 1] > Settings.SPLIT_TOLERANCE))[
					0] + start_with

			if new_zeros is None:
				new_zeros = np.nonzero(self.output_bounds[start_with:, 1] < -Settings.SPLIT_TOLERANCE)[0] + start_with
			Timers.toc('layer_bounds_process_layer')
			return new_zeros

	def copy(self):
		rv = LayerBounds()
		rv.output_bounds = copy.copy(self.output_bounds)
		rv.branching_neurons = copy.copy(self.branching_neurons)


class ZonoState:
	def __init__(self, network_count, state=None):
		self.network_count = network_count
		self.output_zonos = []
		self.active = True
		for x in range(0, self.network_count):
			self.output_zonos.append(None)

		if state is not None:
			self.from_state(state)
			return
		self.branching = []
		self.zono = None
		self.cur_network = 0
		self.cur_layer = 0
		self.split_heuristic = None
		self.lpi = None
		# self.branching = []
		self.initial_zono = None
		self.layer_bounds = LayerBounds()
		self.workload = 1.0
		self.depth = 0

	# self.branching_precise=[]

	def allows_refinement(self):
		return False

	def refine(self, refine_index):
		assert False

	def from_init_zono(self, init: Zonotope, set_initial=True):
		self.zono = init.deep_copy()
		if set_initial:
			self.initial_zono = init.deep_copy()
		if self.lpi is None:
			self.lpi = LpInstance()
			for i, (lb, ub) in enumerate(self.zono.init_bounds):
				self.lpi.add_double_bounded_cols([f"i{i}"], lb, ub)
		self.split_heuristic = TrivialHeuristic(len(init.init_bounds))

	def from_state(self, state):
		Timers.tic('zono_state_from_state')
		self.zono = state.zono.deep_copy()
		self.branching = copy.deepcopy(state.branching)
		self.initial_zono = state.initial_zono.deep_copy()
		self.cur_network = state.cur_network
		self.cur_layer = state.cur_layer
		self.layer_bounds = LayerBounds()
		self.split_heuristic = state.split_heuristic.copy()
		self.lpi = LpInstance(other_lpi=state.lpi)
		# self.branching = copy.copy(state.branching)
		# self.branching_precise = copy.deepcopy(state.branching_precise)
		state.workload /= 2
		self.workload = state.workload
		self.depth = state.depth
		for x in range(0, self.cur_network):
			self.output_zonos[x] = state.output_zonos[x].deep_copy()
		Timers.toc('zono_state_from_state')

	def split(self):
		split_dim = self.split_heuristic.get_split()
		copy_zono = ZonoState(self.network_count, state=self)
		up = self.zono.init_bounds[split_dim][1]
		low = self.zono.init_bounds[split_dim][0]
		mid = low + (up - low) / 2
		self.zono.update_init_bounds(split_dim, (low, mid))
		copy_zono.zono.update_init_bounds(split_dim, (mid, up))
		return copy_zono, self

	def contract_domain(self, row, bias, index, networks, overflow):
		Timers.tic('zono_state_contract_domain')
		tuple_list = self.zono.contract_domain(row, bias)
		self.update_lp(row, bias, tuple_list)
		# TODO(steuber): How often should we really be doing this feasibilitiy this?
		self.check_feasible(overflow, networks)
		zeros = self.layer_bounds.process_layer(self.zono, start_with=index + 1)
		self.set_to_zero(zeros)
		Timers.toc('zono_state_contract_domain')

	def update_lp(self, row, bias, tuple_list):
		Timers.tic('zono_state_update_lp')
		self.lpi.add_dense_row(row, bias)
		for i, l, u in tuple_list:
			if i < self.zono.mat_t.shape[1]:
				self.lpi.set_col_bounds(i, float(l), float(u))
		Timers.toc('zono_state_update_lp')

	def do_first_relu_split(self, networks: [NeuralNetwork], branch_decision=None, index=None):
		# TODO(steuber): Maybe reorder: Only create child zono if feasible?
		Timers.tic('do_first_relu_split')
		network = networks[self.cur_network]
		assert isinstance(network.layers[self.cur_layer], ReluLayer)
		self.depth += 1
		if index is None:
			index = self.layer_bounds.pop_branch()
		if index is None:
			Timers.toc('do_first_relu_split')
			return None
		row = self.zono.mat_t[index]
		bias = self.zono.center[index]
		if branch_decision is None or branch_decision.decision == BranchDecision.BOTH:
			child = self.get_child()
			pos, neg = self, child

			pos.contract_domain(-row, bias, index, networks, self.layer_bounds.output_bounds[index, 1])
			neg.contract_domain(row, -bias, index, networks, -self.layer_bounds.output_bounds[index, 0])
			pos.branching.append(BranchDecision(self.cur_network, self.cur_layer, index, BranchDecision.POS))
			neg.branching.append(BranchDecision(self.cur_network, self.cur_layer, index, BranchDecision.NEG))
			if neg.active:
				neg.zono.mat_t[index] = 0.0
				neg.zono.center[index] = 0.0
				neg.propagate_up_to_split(networks)
				rv = neg
			else:
				rv = None
		elif branch_decision.decision == BranchDecision.POS:
			pos, rv = self, None
			pos.contract_domain(-row, bias, index, networks, self.layer_bounds.output_bounds[index, 1])
			pos.branching.append(BranchDecision(self.cur_network, self.cur_layer, index, BranchDecision.POS))
		elif branch_decision.decision == BranchDecision.NEG:
			neg, rv = self, None
			neg.contract_domain(row, -bias, index, networks, -self.layer_bounds.output_bounds[index, 0])
			neg.branching.append(BranchDecision(self.cur_network, self.cur_layer, index, BranchDecision.NEG))
			neg.zono.mat_t[index] = 0.0
			neg.zono.center[index] = 0.0

		Timers.toc('do_first_relu_split')
		return rv

	def get_child(self):
		return ZonoState(self.network_count, state=self)

	def propagate_layer(self, networks: [NeuralNetwork]):
		Timers.tic('propagate_layer')
		network = networks[self.cur_network]
		layer = network.layers[self.cur_layer]
		assert not isinstance(layer, ReluLayer)
		Timers.tic('propagate_layer_transform')
		layer.transform_zono(self.zono)
		Timers.toc('propagate_layer_transform')
		# self.branching_precise.append(np.array([]))
		# self.branching_precise.append(np.array([True] * self.zono.mat_t.shape[0]))
		Timers.toc('propagate_layer')

	def next_layer(self):
		if self.layer_bounds.remaining_splits() <= 0:
			self.cur_layer += 1
			self.layer_bounds.clear_output_bounds()

	def propagate_up_to_split(self, networks: [NeuralNetwork]):
		Timers.tic('propagate_up_to_split')
		while not self.is_finished(networks):
			network = networks[self.cur_network]
			layer = network.layers[self.cur_layer]
			if isinstance(layer, ReluLayer):
				zeros = self.layer_bounds.process_layer(self.zono)
				self.set_to_zero(zeros)

				if self.layer_bounds.remaining_splits() > 0:
					break

				self.next_layer()
			else:
				self.propagate_layer(networks)
				self.next_layer()
		Timers.toc('propagate_up_to_split')

	def set_to_zero(self, zeros):
		Timers.tic('set_to_zero')
		if zeros is not None:
			# self.branching_precise[-1][zeros]=False
			self.zono.mat_t[zeros] = 0.0
			self.zono.center[zeros] = 0.0
		Timers.toc('set_to_zero')

	def is_finished(self, networks: [NeuralNetwork]):
		Timers.tic('is_finished')
		if not self.active:
			Timers.toc('is_finished')
			return True
		if self.cur_network < self.network_count and self.cur_layer >= len(networks[self.cur_network].layers):
			self.cur_layer = 0
			self.output_zonos[self.cur_network] = self.zono
			new_zono = Zonotope(
				self.initial_zono.center,
				self.initial_zono.mat_t[:, :len(self.initial_zono.init_bounds)],
				init_bounds=self.zono.init_bounds[:len(self.initial_zono.init_bounds)]
			)
			# new_in_star = self.initial_star.copy()
			# new_in_star.lpi = LpInstance(self.star.lpi)
			self.cur_network += 1
			self.from_init_zono(new_zono)

		if self.network_count <= self.cur_network:
			Timers.toc('is_finished')
			return True
		else:
			Timers.toc('is_finished')
			return False

	def get_output_zonos(self):
		return self.output_zonos


	def check_feasible(self, overflow, networks):
		assert self.active
		if self.lpi.get_num_rows()==0:
			# No need to check if there are no constraints
			return True
		Timers.tic('is_feasible')
		feasible = self.lpi.minimize(None, fail_on_unsat=False)
		if feasible is None:
			self.active = False
			GLOBAL_STATE.WRONG += 1
			GLOBAL_STATE.INVALID_DEPTH.append((overflow, self.depth))
			GLOBAL_STATE.FINISHED_FRAC += self.workload
			Timers.toc('is_feasible')
			return False
		else:
			GLOBAL_STATE.VALID_DEPTH_DECISION.append((overflow, self.depth))
			Timers.toc('is_feasible')
			return True

	def wrap_up(self):
		return []


def status_update(in_pipeline):
	Timers.tic('status_update')
	if GLOBAL_STATE.FINISHED_FRAC > 0:
		total = GLOBAL_STATE.WRONG + GLOBAL_STATE.RIGHT
		expected = int(total / GLOBAL_STATE.FINISHED_FRAC)
		percentage = float(total) / float(expected) * 100
		exact = GLOBAL_STATE.RIGHT-GLOBAL_STATE.OVERAPPOXED_RIGHT
		print(
			f"\rPassed First: {GLOBAL_STATE.PASSED_FIRST} | In Pipeline: {in_pipeline} | Refinement Avg: {GLOBAL_STATE.REFINEMENT_AVG} | Wrong: {GLOBAL_STATE.WRONG} | Refined: {GLOBAL_STATE.NEED_REFINEMENT} | Right: {GLOBAL_STATE.RIGHT}  (E: {exact}/O: {GLOBAL_STATE.OVERAPPOXED_RIGHT}) | Total Done: {total} | Expected {expected} ({percentage}%) (Depth {GLOBAL_STATE.MAX_DEPTH})",
			end="")
	Timers.toc('status_update')


class TrivialHeuristic:
	def __init__(self, dimension):
		self.next = 0
		self.max = dimension

	def get_split(self):
		result = self.next
		self.next = (self.next + 1) % self.max
		return result

	def copy(self):
		rv = TrivialHeuristic(self.max)
		rv.next = self.next
		return rv
