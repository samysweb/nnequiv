import copy
import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.network import NeuralNetwork, ReluLayer
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope


class LayerBounds:
	def __init__(self,branching_decisions=[[]]):
		self.output_bounds = None
		self.branching_neurons = None
		# TODO(steuber): Remove? (only debugging)
		self.branching_decisions = branching_decisions

	def clear_output_bounds(self):
		self.output_bounds = None
		self.branching_neurons = None
		self.branching_decisions.append([])

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

	def process_layer(self, zono, start_with=0, include_start=True):
		if len(self.branching_decisions[-1])==0:
			self.branching_decisions[-1]+=[None]*zono.mat_t.shape[0]
		if self.output_bounds is None or start_with!=0:
			self.output_bounds = zono.box_bounds()
			self.branching_neurons = np.nonzero(np.logical_and(self.output_bounds[start_with:, 0] < -Settings.SPLIT_TOLERANCE,
			                                                   self.output_bounds[start_with:, 1] > Settings.SPLIT_TOLERANCE))[0]+start_with
			if not include_start and self.branching_neurons[0]==start_with:
				self.branching_neurons=self.branching_neurons[1:]

			for i,b in enumerate(self.output_bounds[start_with:]):
				if b[1] < -Settings.SPLIT_TOLERANCE:
					self.branching_decisions[-1][i+start_with]=False
				elif b[0] > Settings.SPLIT_TOLERANCE:
					self.branching_decisions[-1][i + start_with] = True
				else:
					self.branching_decisions[-1][i + start_with]=None

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
		self.layer_bounds.branching_decisions = copy.deepcopy(state.layer_bounds.branching_decisions)
		self.split_heuristic = state.split_heuristic.copy()
		self.lpi = LpInstance(other_lpi=state.lpi)
		self.branching = copy.copy(state.branching)
		for x in range(0, self.cur_network):
			self.output_zonos[x] = state.output_zonos[x].deep_copy()

	def contract_domain(self, row, bias, index):
		self.zono.contract_domain(row,bias)
		self.layer_bounds.process_layer(self.zono,start_with=index,include_start=False)

	def do_first_relu_split(self, networks: [NeuralNetwork]):
		Timers.tic('do_first_relu_split')
		network = networks[self.cur_network]
		assert isinstance(network.layers[self.cur_layer], ReluLayer)

		index = self.layer_bounds.pop_branch()
		row = self.zono.mat_t[index]
		bias = self.zono.center[index]
		child = ZonoState(self.network_count, state=self)
		child.layer_bounds.branching_decisions[-1][index]=not child.layer_bounds.branching_decisions[-1][index]
		pos, neg = self, child

		pos.contract_domain(-row, bias, index)
		neg.contract_domain(row, -bias, index)
		pos.lpi.add_dense_row(-row, -bias)
		neg.lpi.add_dense_row(row, -bias)
		pos.branching.append((self.cur_network, self.cur_layer, index, True))
		neg.branching.append((self.cur_network, self.cur_layer, index, False))

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
				if self.layer_bounds.remaining_splits() > 0:
					break

				self.next_layer()
			else:
				self.propagate_layer(networks)
				self.next_layer()
				self.layer_bounds.process_layer(self.zono)
		Timers.toc('propagate_up_to_split')
		return (start_layer!=self.cur_layer),self.zono.mat_t,self.zono.center

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
		for i, (lb, ub) in enumerate(self.zono.init_bounds):
			self.lpi.set_col_bounds(i, lb, ub)
		feasible = self.lpi.minimize(None,fail_on_unsat=False)
		if feasible is None:
			# TODO(steuber): Stats on wrong paths
			return False
		else:
			print(f"Feasible: {feasible}")
			print(f"Net 1: {networks[0].execute(np.array(feasible,dtype=np.float32), save_branching=True)}")
			print(f"Net 2: {networks[1].execute(np.array(feasible,dtype=np.float32), save_branching=True)}")
			print(f"Zon 1: {np.dot(self.output_zonos[0].mat_t,feasible) + self.output_zonos[0].center}")
			print(f"Zonotope: {self.output_zonos[0].mat_t}")
			print(f"Branching: {self.layer_bounds.branching_decisions}")
			print(f"Branches: {str(self.branching)}")
			return True


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
