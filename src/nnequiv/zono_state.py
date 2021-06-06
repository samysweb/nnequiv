import copy
from collections import namedtuple
from enum import Enum, auto

import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.network import NeuralNetwork, ReluLayer
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from nnequiv.global_state import GLOBAL_STATE

class SplitDecision(Enum):
	DNC = auto() #  Don't care
	BOTH = auto()
	POS = auto()
	NEG = auto()
	def __eq__(self, other):
		if self.value == SplitDecision.DNC.value:
			return True
		else:
			return self.value==other.value

SplitPoint = namedtuple('SplitPoint', ['network', 'layer', 'index', 'decision'])
OverapproxPoint = namedtuple('OverapproxPoint', ['network','layer','index','coefficient'])

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
		if self.output_bounds is None or start_with!=0:
			Timers.tic('layer_bounds_process_layer')
			self.output_bounds = zono.box_bounds()
			new_zeros = None
			if self.branching_neurons is not None:
				new_zeros = self.branching_neurons[self.output_bounds[self.branching_neurons,1]<-Settings.SPLIT_TOLERANCE]

			self.branching_neurons = np.nonzero(np.logical_and(self.output_bounds[start_with:, 0] < -Settings.SPLIT_TOLERANCE,
		                                                   self.output_bounds[start_with:, 1] > Settings.SPLIT_TOLERANCE))[0]+start_with

			if new_zeros is None:
				new_zeros = np.nonzero(self.output_bounds[start_with:,1] < -Settings.SPLIT_TOLERANCE)[0]+start_with
			Timers.toc('layer_bounds_process_layer')
			return new_zeros

	def copy(self):
		rv = LayerBounds()
		rv.output_bounds = copy.copy(self.output_bounds)
		rv.branching_neurons = copy.copy(self.branching_neurons)


class ZonoState:
	def __init__(self, network_count, state=None, do_branching=None):
		self.network_count = network_count
		self.output_zonos = []
		self.active = True
		for x in range(0, self.network_count):
			self.output_zonos.append(None)

		if state is not None:
			self.from_state(state)
			return
		self.zono = None
		self.cur_network = 0
		self.cur_layer = 0
		self.split_heuristic = None
		self.lpi = None
		self.branching = []
		self.initial_zono = None
		self.layer_bounds = LayerBounds()
		self.workload = 1.0
		self.depth=0
		if do_branching is None:
			do_branching = []
		self.do_branching = do_branching
		self.overapprox_nodes = []
		self.do_exact = False
		self.exactCounter=None

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
		self.initial_zono = state.initial_zono.deep_copy()
		self.cur_network = state.cur_network
		self.cur_layer = state.cur_layer
		self.layer_bounds = LayerBounds()
		self.split_heuristic = state.split_heuristic.copy()
		self.lpi = LpInstance(other_lpi=state.lpi)
		self.branching = copy.copy(state.branching)
		self.do_branching = copy.copy(state.do_branching)
		self.overapprox_nodes = state.overapprox_nodes.copy()
		state.workload/=2
		self.workload=state.workload
		self.depth=state.depth
		self.do_exact = state.do_exact
		self.exactCounter = state.exactCounter
		for x in range(0, self.cur_network):
			self.output_zonos[x] = state.output_zonos[x].deep_copy()
		Timers.toc('zono_state_from_state')

	def split(self):
		split_dim = self.split_heuristic.get_split()
		copy_zono = ZonoState(self.network_count,state=self)
		up = self.zono.init_bounds[split_dim][1]
		low = self.zono.init_bounds[split_dim][0]
		mid = low + (up-low)/2
		self.zono.update_init_bounds(split_dim,(low,mid))
		copy_zono.zono.update_init_bounds(split_dim, (mid,up))
		return copy_zono, self

	def contract_domain(self, row, bias, index, networks, overflow):
		Timers.tic('zono_state_contract_domain')
		tuple_list = self.zono.contract_domain(row,bias)
		self.update_lp(row, bias, tuple_list)
		# TODO(steuber): How often should we really be doing this feasibilitiy this?
		self.check_feasible(overflow,networks)
		zeros = self.layer_bounds.process_layer(self.zono,start_with=index+1)
		self.set_to_zero(zeros)
		Timers.toc('zono_state_contract_domain')

	def update_lp(self, row, bias, tuple_list):
		Timers.tic('zono_state_update_lp')
		lp_col_num = self.lpi.get_num_cols()
		lp_row = row[:lp_col_num]
		Timers.tic('zono_state_update_lp_alpha_min')
		alpha_row = row[lp_col_num:]
		ib = np.array(self.zono.init_bounds, dtype=self.zono.dtype)
		alpha_min = self.lpi.compute_residual(alpha_row, ib[lp_col_num:])
		Timers.toc('zono_state_update_lp_alpha_min')
		self.lpi.add_dense_row(lp_row, bias - alpha_min)
		for i, l, u in tuple_list:
			self.lpi.set_col_bounds(i, l, u)
		Timers.toc('zono_state_update_lp')

	def split_decision(self, networks):
		Timers.tic("zono_state_split_decision")
		index = self.layer_bounds.pop_branch()
		if index is None:
			Timers.toc("zono_state_split_decision")
			return None
		if Settings.EQUIV_OVERAPPROX_STRAT == 'DONT':
			Timers.toc("zono_state_split_decision")
			return SplitPoint(self.cur_network, self.cur_layer, index, SplitDecision.BOTH)
		elif Settings.EQUIV_OVERAPPROX_STRAT == 'CEGAR' or Settings.EQUIV_OVERAPPROX_STRAT == 'SECOND_NET' or Settings.EQUIV_OVERAPPROX_STRAT == "REFINE_UNTIL_MAX":
			cur_split_point = SplitPoint(self.cur_network, self.cur_layer, index, SplitDecision.DNC)
			while len(self.do_branching)>0 and cur_split_point>self.do_branching[-1]:
				popped_el = self.do_branching.pop()
				# print(f"Skipping {popped_el}")
			if len(self.do_branching)>0 and cur_split_point==self.do_branching[-1]:
				Timers.toc("zono_state_split_decision")
				return self.do_branching.pop()
			else:
				if Settings.EQUIV_OVERAPPROX_STRAT == 'SECOND_NET' and self.cur_network==0:
					Timers.toc("zono_state_split_decision")
					return SplitPoint(self.cur_network, self.cur_layer, index, SplitDecision.BOTH)
				elif Settings.EQUIV_OVERAPPROX_STRAT == "REFINE_UNTIL_MAX" and len(self.branching)<GLOBAL_STATE.MAX_REFINE_COUNT:
					Timers.toc('zono_state_split_decision')
					return SplitPoint(self.cur_network, self.cur_layer, index, SplitDecision.BOTH)
				else:
					self.overapproximate(index, networks)
					Timers.toc("zono_state_split_decision")
					return None



	def overapproximate(self, index, networks: [NeuralNetwork]):
		Timers.tic('overapprox')
		row = self.zono.mat_t[index]
		bias = self.zono.center[index]
		l, u = self.layer_bounds.output_bounds[index]
		factor = u / (u - l)
		new_dim_u = max(u * (-l) / (u - l), u * u / (u - l))
		assert new_dim_u > 0.0
		self.zono.mat_t[index] = factor * row
		self.zono.center[index] = factor * bias
		dim = self.zono.add_dimension(0.0, new_dim_u)
		self.zono.mat_t[index, dim] = 1.0
		self.overapprox_nodes.append(OverapproxPoint(self.cur_network, self.cur_layer, index, dim))
		Timers.toc('overapprox')


	def do_first_relu_split(self, networks: [NeuralNetwork]):
		#TODO(steuber): Maybe reorder: Only create child zono if feasible?
		Timers.tic('do_first_relu_split')
		network = networks[self.cur_network]
		assert isinstance(network.layers[self.cur_layer], ReluLayer)

		self.depth+=1
		cur_split_decision = self.split_decision(networks)
		if cur_split_decision is None:
			Timers.toc('do_first_relu_split')
			return None
		split_net, split_layer, index, decision = cur_split_decision
		row = self.zono.mat_t[index]
		bias = self.zono.center[index]
		pos, neg = None, None
		rv = None
		if decision == SplitDecision.BOTH:
			child = ZonoState(self.network_count, state=self)
			pos, neg = self, child
			rv = neg
		elif decision == SplitDecision.NEG:
			pos, neg = None, self
		elif decision == SplitDecision.POS:
			pos,neg = self, None

		if pos is not None:
			pos.contract_domain(-row, bias, index, networks,self.layer_bounds.output_bounds[index,1])
			pos.branching.append(SplitPoint(split_net, split_layer, index, SplitDecision.POS))
		if neg is not None:
			neg.contract_domain(row, -bias, index, networks,-self.layer_bounds.output_bounds[index,0])
			if neg.active:
				neg.zono.mat_t[index] = 0.0
				neg.zono.center[index] = 0.0

				neg.propagate_up_to_split(networks)
				neg.branching.append(SplitPoint(split_net, split_layer, index, SplitDecision.NEG))
			else:
				rv = None

		Timers.toc('do_first_relu_split')
		return rv

	def propagate_layer(self, networks: [NeuralNetwork]):
		Timers.tic('propagate_layer')
		network = networks[self.cur_network]
		layer = network.layers[self.cur_layer]
		assert not isinstance(layer, ReluLayer)
		Timers.tic('propagate_layer_transform')
		layer.transform_zono(self.zono)
		Timers.toc('propagate_layer_transform')
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
			new_dims = self.zono.mat_t.shape[1] - self.initial_zono.mat_t.shape[1]
			init_center = self.initial_zono.center.copy()
			init_mat = self.initial_zono.mat_t.copy()
			if new_dims > 0:
				init_mat = np.pad(init_mat, ((0, 0), (0, new_dims)))
			new_zono = Zonotope(
				init_center,
				init_mat,
				init_bounds=self.zono.init_bounds
			)
			new_zono.pos1_gens = None
			new_zono.neg1_gens = None
			new_zono.init_bounds_nparray = None
			# new_in_star = self.initial_star.copy()
			# new_in_star.lpi = LpInstance(self.star.lpi)
			self.cur_network += 1
			self.from_init_zono(new_zono, set_initial=False)

		if self.network_count <= self.cur_network:
			Timers.toc('is_finished')
			return True
		else:
			Timers.toc('is_finished')
			return False

	def get_output_zonos(self):
		rv = []
		dim_count = self.output_zonos[-1].mat_t.shape[1]
		for i in range(len(self.output_zonos) - 1):
			missing_dims = dim_count - self.output_zonos[i].mat_t.shape[1]
			if missing_dims > 0:
				mat_t = np.pad(
					self.output_zonos[i].mat_t,
					((0, 0), (0, missing_dims))
				)
				rv.append(Zonotope(self.output_zonos[i].center, mat_t, self.output_zonos[i].init_bounds))
			else:
				rv.append(self.output_zonos[i].deep_copy())
		rv.append(self.output_zonos[-1].deep_copy())
		return rv

	def check_feasible(self, overflow, networks):
		assert self.active
		Timers.tic('is_feasible')
		feasible = self.lpi.minimize(None,fail_on_unsat=False, use_exact=False)
		if feasible is None:
			self.active=False
			GLOBAL_STATE.WRONG+=1
			GLOBAL_STATE.FINISHED_FRAC+=self.workload
			Timers.toc('is_feasible')
			return False
		else:
			Timers.toc('is_feasible')
			return True

	def admits_refinement(self):
		return len(self.overapprox_nodes)>0

	def refine(self):
		branching_list = self.get_branching_list()
		rv = ZonoState(self.network_count, do_branching=branching_list)
		rv.from_init_zono(self.initial_zono)
		rv.workload = self.workload
		return [rv]

	def get_branching_list(self):
		refine_node = self.overapprox_nodes[0]
		previous_branching = copy.copy(self.branching)
		previous_branching.append(SplitPoint(refine_node.network,
		                                     refine_node.layer,
		                                     refine_node.index, SplitDecision.BOTH))
		previous_branching.sort()
		return list(reversed(previous_branching))

	def __str__(self):
		return "ZONOTOPE\n"+\
		f"PAST BRANCHING: {self.branching}\n"+\
		f"NEXT BRANCHING: {self.do_branching}\n"+\
		f"OVERAPPROX: {len(self.overapprox_nodes)}"


def status_update():
	Timers.tic('status_update')
	if GLOBAL_STATE.FINISHED_FRAC>0:
		total = GLOBAL_STATE.WRONG+GLOBAL_STATE.RIGHT
		expected = int(total / GLOBAL_STATE.FINISHED_FRAC)
		percentage = float(total)/float(expected)*100
		print(
		f"\rRefined: {GLOBAL_STATE.REFINED} | Max Refine Depth: {GLOBAL_STATE.MAX_REFINE_COUNT} | Wrong: {GLOBAL_STATE.WRONG} | Right: {GLOBAL_STATE.RIGHT} | Total: {total} | Total Zonos Considered: {total+GLOBAL_STATE.REFINED} | Expected {expected} ({percentage}%)",end="")
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
