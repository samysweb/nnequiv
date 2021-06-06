import copy
from collections import namedtuple

import numpy as np

from nnenum.network import NeuralNetwork
from nnenum.timerutil import Timers
from nnequiv.equivalence_properties import EquivalenceProperty
from nnequiv.global_state import GLOBAL_STATE
from nnequiv.refinement import Refinement
from nnequiv.zono_state import ZonoState

class OverapproxNodeSplits:
	def __init__(self, splits, overapprox):
		self.split_nodes = splits
		self.overapprox_nodes=overapprox
		self.exact_sets=0

	def inc(self):
		self.exact_sets+=1

	def __str__(self):
		return str((self.split_nodes, self.overapprox_nodes, self.exact_sets))

	def __repr__(self):
		return self.__str__()

class EnumerationStackElement:
	def __init__(self, state: ZonoState):
		self.state = state

	def get_state(self):
		return self.state

	def is_finished(self,networks):
		return self.state.is_finished(networks)

	def advance_zono(self, networks):
		Timers.tic('advance_zono')
		assert self.state.active
		new_el = self.state.do_first_relu_split(networks)
		if self.state.active:
			self.state.propagate_up_to_split(networks)
		if new_el is None:  # We crossed a layer => new EnumerationStackElement
			Timers.toc('advance_zono')
			return None
		else:
			Timers.toc('advance_zono')
			return EnumerationStackElement(new_el)


class StateManager:
	def __init__(self, init: ZonoState, property: EquivalenceProperty, networks: [NeuralNetwork]):
		self.enumeration_stack = [EnumerationStackElement(init)]
		self.property = property
		self.networks = networks

	def get_networks(self):
		return self.networks

	def done(self):
		return len(self.enumeration_stack) == 0

	def peek(self):
		assert len(self.enumeration_stack) > 0
		return self.enumeration_stack[-1]

	def pop(self):
		assert len(self.enumeration_stack) > 0
		popped = self.enumeration_stack.pop()
		return popped

	def push(self, el: EnumerationStackElement):
		assert el.state.active
		self.enumeration_stack.append(el)

	def check(self, el: EnumerationStackElement):
		Timers.tic('StateManager.check')
		assert el.state.active
		equiv, data = self.property.check(el.state)
		valid, result = self.valid_result(el, equiv, data)
		if not valid and self.property.allows_fallback(el.state):
			equiv, data = self.property.fallback_check(el.state)
			valid, result = self.valid_result(el, equiv, data)
		if not valid:
			assert el.state.admits_refinement()
			new_zonos = el.state.refine()
			for z in new_zonos:
				z.propagate_up_to_split(self.networks)
				self.push(EnumerationStackElement(z))
				GLOBAL_STATE.REFINED+=1
			result=True
		else:
			if not el.state.do_exact:
				branching_list = copy.copy(el.state.branching)
				branching_list.sort()
				exactCounter = OverapproxNodeSplits(len(el.state.branching),len(el.state.overapprox_nodes))
				rv = ZonoState(el.state.network_count, do_branching=list(reversed(branching_list)))
				rv.from_init_zono(el.state.initial_zono)
				rv.do_exact = True
				rv.branches = len(el.state.branching)
				rv.exactCounter = exactCounter
				rv.propagate_up_to_split(self.networks)
				GLOBAL_STATE.EXACT_COUNTERS.append(exactCounter)
				self.push(EnumerationStackElement(rv))
				GLOBAL_STATE.RIGHT += 1
				GLOBAL_STATE.FINISHED_FRAC += el.state.workload
				GLOBAL_STATE.MAX_REFINE_COUNT = max(GLOBAL_STATE.MAX_REFINE_COUNT, len(el.state.branching))
			else:
				el.state.exactCounter.inc()
		Timers.toc('StateManager.check')
		return result

	def valid_result(self, el: EnumerationStackElement, equiv, data):
		"""
		Check whether result returned by property check is valid
		:param el: The EnumerationStackElement that was checked
		:param equiv: Whether the two networks were equivalent
		:param data: The data returned by the check
		:return: A tuple rv. rv[0] is True iff the result by check is valid. rv[0] is true iff the result is valid and
			equivalence was shown.
		"""
		Timers.tic('StateManager.valid_result')
		if not equiv:
			# TODO(steuber): Make float types explicit?
			input_size = self.networks[0].get_num_inputs()
			r1 = self.networks[0].execute(np.array(data[1][:input_size],dtype=np.float32))
			r2 = self.networks[1].execute(np.array(data[1][:input_size],dtype=np.float32))
			if not self.property.check_out(r1, r2):
				print(f"\n[NEQUIV] {data[0]}\n")
				print(r1)
				print(r2)
				# We found a counter-example -- that's it
				Timers.toc('StateManager.valid_result')
				return (True, False)
			else:
				# print(f"\n[NEED_FALLBACK] {data[0]}\n")
				Timers.toc('StateManager.valid_result')
				return (False, False)
		else:
			#print(f"\n[EQUIV] {data[0]}\n")
			Timers.toc('StateManager.valid_result')
			return (True, True)