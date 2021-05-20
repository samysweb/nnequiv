import copy

import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.network import NeuralNetwork
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from nnequiv.zono_state import ZonoState, BranchDecision, LayerBounds


class OverapproxNode:
	def __init__(self, cur_network, cur_layer, index, coefficient_index):
		self.cur_network = cur_network
		self.cur_layer = cur_layer
		self.index = index
		self.coefficient_index = coefficient_index

class OverapproxZonoState(ZonoState):
	def __init__(self, network_count, state=None):
		super().__init__(network_count, state=state)
		if state is None:
			self.overapprox_nodes = []

	def allows_refinement(self):
		return len(self.overapprox_nodes) > 0

	def from_state(self, state):
		super().from_state(state)
		self.overapprox_nodes = state.overapprox_nodes

	def get_child(self):
		pass

	def is_finished(self, networks: [NeuralNetwork]):
		Timers.tic('is_finished')
		if not self.active:
			Timers.toc('is_finished')
			return True
		if self.cur_network < self.network_count and self.cur_layer >= len(networks[self.cur_network].layers):
			self.cur_layer = 0
			self.output_zonos[self.cur_network] = self.zono
			new_dims = self.zono.mat_t.shape[1] - self.initial_zono.mat_t.shape[1]
			init_center = self.initial_zono.center
			init_mat = self.initial_zono.mat_t
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
			dim_count = self.output_zonos[-1].mat_t.shape[1]
			for i in range(len(self.output_zonos) - 1):
				missing_dims = dim_count - self.output_zonos[i].mat_t.shape[1]
				if missing_dims > 0:
					self.output_zonos[i].mat_t = np.pad(
						self.output_zonos[i].mat_t,
						((0, 0), (0, missing_dims))
					)
				self.output_zonos[i].pos1_gens = None
				self.output_zonos[i].neg1_gens = None
				self.output_zonos[i].init_bounds_nparray = None
			Timers.toc('is_finished')
			return True
		else:
			Timers.toc('is_finished')
			return False

	def overapprox(self, index, networks: [NeuralNetwork]):
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
		self.lpi.add_double_bounded_cols([f"i{dim}"], 0.0, new_dim_u)
		self.zono.mat_t[index, dim] = 1.0
		self.overapprox_nodes.append(OverapproxNode(self.cur_network, self.cur_layer, index, dim))
		Timers.toc('overapprox')

class EgoCacheNode:
	def __init__(self, zono_state, fallback):
		self.zono_state = zono_state
		self.fallback = fallback

class EgoZonoState(OverapproxZonoState):
	def __init__(self, network_count, state=None):
		super().__init__(network_count, state=state)
		if state is None:
			self.overapprox_mode=False
			self.ego_cache = []
			self.fallback = None

	def from_state(self, state):
		super().from_state(state)
		self.overapprox_mode=state.overapprox_mode
		# TODO(steuber): need for deepcopy?
		self.ego_cache = copy.copy(state.ego_cache)

	def get_child(self):
		return EgoZonoState(self.network_count, state=self)

	def refine(self, refine_index):
		Timers.tic('ego_refine')
		assert self.overapprox_mode
		print("[refine] larger scope failed...")
		self.fallback.overapprox_mode=False
		Timers.tic('ego_refine')
		return [self.fallback]

	def produce_cache_state(self, bounds):
		Timers.tic('ego_cache_state')
		rv = EgoZonoState(self.network_count)
		rv.zono = self.zono.deep_copy()
		rv.ego_cache = self.ego_cache

		rv.branching = copy.deepcopy(self.branching)
		rv.initial_zono = self.initial_zono.deep_copy()
		rv.cur_network = self.cur_network
		rv.cur_layer = self.cur_layer
		rv.workload = self.workload
		for x in range(0, rv.cur_network):
			rv.output_zonos[x] = self.output_zonos[x].deep_copy()
		rv.layer_bounds = bounds
		rv.lpi = LpInstance(other_lpi=self.lpi)
		Timers.tic('ego_cache_state')
		return rv

	def do_first_relu_split(self, networks: [NeuralNetwork], branch_decision=None, index=None):
		bounds = copy.deepcopy(self.layer_bounds)
		if index is None:
			index = self.layer_bounds.pop_branch()
		if index is None:
			return None
		if self.overapprox_mode:
			self.overapprox(index, networks)
			return None
		zono_cache = self.produce_cache_state(bounds)
		rv = super().do_first_relu_split(networks, index=index)
		if self.active:
			if rv is not None:
				self.ego_cache.append(EgoCacheNode(zono_cache, rv))
			return None
		else:
			assert rv is not None
			rv.ego_cache=self.ego_cache
			rv.overapprox_mode=self.overapprox_mode
			return rv

	def wrap_up(self):
		Timers.tic('ego_wrap_up')
		if len(self.ego_cache)==0:
			return []
		node = self.ego_cache.pop()
		rv = node.zono_state
		rv.fallback = node.fallback
		rv.overapprox_mode = True
		print("[wrap_up] retrying with larger scope")
		Timers.toc('ego_wrap_up')
		return [rv]



class CegarZonoState(OverapproxZonoState):
	def __init__(self, network_count, state=None, branch_on=[]):
		super().__init__(network_count, state=state)
		if state is None:
			self.branch_on = branch_on

	def refine(self, refine_index):
		Timers.tic('cegar_refine')
		node = self.overapprox_nodes[refine_index]
		branches = []
		added_node = False
		new_branch_decision = BranchDecision(node.cur_network, node.cur_layer, node.index, BranchDecision.BOTH)
		for branch in reversed(self.branching):
			if branch < new_branch_decision \
					and not added_node:
				branches.append(new_branch_decision)
				added_node = True
			branches.append(branch)
		if not added_node:
			branches.append(new_branch_decision)
		rv = CegarZonoState(self.network_count, branch_on=branches)
		rv.workload = self.workload
		rv.from_init_zono(self.initial_zono)
		Timers.toc('cegar_refine')
		return [rv]

	def from_state(self, state):
		super().from_state(state)
		self.branch_on = copy.deepcopy(state.branch_on)

	def get_child(self):
		return CegarZonoState(self.network_count, state=self)

	def do_first_relu_split(self, networks: [NeuralNetwork], branch_decision=None, index=None):
		if index is None:
			index = self.layer_bounds.pop_branch()
		if index is None:
			return None
		if self.do_overapprox(index):
			self.depth += 1
			self.overapprox(index, networks)
			return None
		if branch_decision is None:
			branch_decision = self.branch_on.pop()
		return super().do_first_relu_split(networks, branch_decision=branch_decision, index=index)

	def do_overapprox(self, index):
		Timers.tic('do_overapprox')
		cur_branch = BranchDecision(self.cur_network, self.cur_layer, index, None)
		while len(self.branch_on) > 0 and self.branch_on[-1] < cur_branch:
			self.branch_on.pop()
		if len(self.branch_on) == 0 or (not self.branch_on[-1] == cur_branch):
			Timers.toc('do_overapprox')
			return True
		Timers.toc('do_overapprox')
		return False
