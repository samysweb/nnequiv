import copy
import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.network import NeuralNetwork, ReluLayer
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
WRONG = 0
RIGHT = 0
FINISHED_FRAC = 0.0

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
		assert len(self.branching_neurons) > 0
		rv = self.branching_neurons[0]
		self.branching_neurons = self.branching_neurons[1:]
		return rv

	def process_layer(self, zono):
		Timers.tic('layer_bounds_process_layer')
		self.output_bounds = zono.box_bounds()

		if self.branching_neurons is None:
			self.branching_neurons = np.nonzero(np.logical_and(self.output_bounds[:, 0] < -Settings.SPLIT_TOLERANCE,
	            self.output_bounds[:, 1] > Settings.SPLIT_TOLERANCE))[0]
		elif self.branching_neurons.size>0:
			self.branching_neurons = self.branching_neurons[np.logical_and(self.output_bounds[self.branching_neurons, 0] < -Settings.SPLIT_TOLERANCE,
	            self.output_bounds[self.branching_neurons, 1] > Settings.SPLIT_TOLERANCE)]
		else:
			self.branching_neurons = []

		new_zeros = []
		for i in range(0,len(self.output_bounds)): # TODO(steuber): Speedup possible!
			if self.output_bounds[i,1] < -Settings.SPLIT_TOLERANCE:
				new_zeros.append(i)
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
		self.branching_precise=[]


	def from_init_zono(self, init: Zonotope):
		self.zono = init.deep_copy()
		self.initial_zono = init.deep_copy()
		if self.lpi is None:
			self.lpi = LpInstance()
			for i, (lb, ub) in enumerate(self.zono.init_bounds):
				self.lpi.add_double_bounded_cols([f"i{i}"], lb, ub)
		self.split_heuristic = TrivialHeuristic(len(init.init_bounds))

	def from_state(self, state):
		self.zono = state.zono.deep_copy()
		self.initial_zono = state.initial_zono.deep_copy()
		self.cur_network = state.cur_network
		self.cur_layer = state.cur_layer
		self.layer_bounds = LayerBounds()
		self.split_heuristic = state.split_heuristic.copy()
		self.lpi = LpInstance(other_lpi=state.lpi)
		self.branching = copy.copy(state.branching)
		self.branching_precise = copy.deepcopy(state.branching_precise)
		state.workload/=2
		self.workload=state.workload
		for x in range(0, self.cur_network):
			self.output_zonos[x] = state.output_zonos[x].deep_copy()

	def contract_domain(self, row, bias, index):
		self.zono.contract_domain(row,bias)
		zeros = self.layer_bounds.process_layer(self.zono)
		self.set_to_zero(zeros)

	def do_first_relu_split(self, networks: [NeuralNetwork]):
		Timers.tic('do_first_relu_split')
		network = networks[self.cur_network]
		assert isinstance(network.layers[self.cur_layer], ReluLayer)

		index = self.layer_bounds.pop_branch()
		row = self.zono.mat_t[index]
		bias = self.zono.center[index]
		child = ZonoState(self.network_count, state=self)
		pos, neg = self, child

		pos.contract_domain(-row, bias, index)
		neg.contract_domain(row, -bias, index)
		pos.lpi.add_dense_row(-row, bias)
		neg.lpi.add_dense_row(row, -bias)
		pos.branching.append((self.cur_network, self.cur_layer, index, True))
		neg.branching.append((self.cur_network, self.cur_layer, index, False))
		neg.branching_precise[-1][index]=False

		neg.zono.mat_t[index] = 0.0
		neg.zono.center[index] = 0.0

		neg.propagate_up_to_split(networks)

		Timers.toc('do_first_relu_split')
		return neg

	def propagate_layer(self, networks: [NeuralNetwork]):
		Timers.tic('propagate_layer')
		network = networks[self.cur_network]
		layer = network.layers[self.cur_layer]
		assert not isinstance(layer, ReluLayer)
		layer.transform_zono(self.zono)
		self.branching_precise.append(np.array([]))
		self.branching_precise.append(np.array([True] * self.zono.mat_t.shape[0]))
		Timers.toc('propagate_layer')

	def next_layer(self):
		if self.layer_bounds.remaining_splits() <= 0:
			self.cur_layer += 1
			self.layer_bounds.clear_output_bounds()

	def propagate_up_to_split(self, networks: [NeuralNetwork]):
		Timers.tic('propagate_up_to_split')
		start_layer = self.cur_layer
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
		return (start_layer!=self.cur_layer),self.zono.mat_t,self.zono.center

	def set_to_zero(self, zeros):
		Timers.tic('set_to_zero')
		if zeros is not None:
			self.branching_precise[-1][zeros]=False
			self.zono.mat_t[zeros] = 0.0
			self.zono.center[zeros] = 0.0
		Timers.toc('set_to_zero')

	def is_finished(self, networks: [NeuralNetwork]):
		if self.cur_network < self.network_count and self.cur_layer >= len(networks[self.cur_network].layers):
			self.cur_layer = 0
			self.output_zonos[self.cur_network] = self.zono
			new_zono = Zonotope(
				self.initial_zono.center,
				self.initial_zono.mat_t,
				init_bounds=self.zono.init_bounds
			)
			# new_in_star = self.initial_star.copy()
			# new_in_star.lpi = LpInstance(self.star.lpi)
			self.cur_network += 1
			self.from_init_zono(new_zono)

		if self.network_count <= self.cur_network:
			return True
		else:
			return False

	def is_feasible(self, networks):
		global WRONG, RIGHT, FINISHED_FRAC
		for i, (lb, ub) in enumerate(self.zono.init_bounds):
			self.lpi.set_col_bounds(i, lb, ub)
		feasible = self.lpi.minimize(None,fail_on_unsat=False)
		FINISHED_FRAC+=self.workload
		if feasible is None:
			# TODO(steuber): Stats on wrong paths
			WRONG+=1
			return False
		else:
			RIGHT+=1
			return True

def status_update():
	global WRONG, RIGHT, FINISHED_FRAC
	if FINISHED_FRAC>0:
		print(
		f"\rWrong: {WRONG} | Right: {RIGHT} | Total: {WRONG + RIGHT} | Expected {int((WRONG + RIGHT) / FINISHED_FRAC)}")


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
