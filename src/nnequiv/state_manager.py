import numpy as np

from nnenum.network import NeuralNetwork
from nnenum.timerutil import Timers
from nnequiv.equivalence_properties import EquivalenceProperty
from nnequiv.global_state import GLOBAL_STATE
from nnequiv.refinement_strategies import RefinementStrategy
from nnequiv.zono_state import ZonoState


class EnumerationStackElement:
	def __init__(self, state: ZonoState):
		self.state = state

	def get_state(self):
		return self.state

	def is_finished(self, networks):
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
	def __init__(self, init: ZonoState, property: EquivalenceProperty, networks: [NeuralNetwork],
	             strategy: RefinementStrategy):
		self.enumeration_stack = [EnumerationStackElement(init)]
		self.property = property
		self.strategy = strategy
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
		GLOBAL_STATE.MAX_DEPTH = max(GLOBAL_STATE.MAX_DEPTH, el.state.depth)
		equiv, data = self.property.check(el.state)
		valid, result = self.valid_result(el, equiv, data)
		if not valid:
			equiv, data = self.property.fallback_check(el.state)
			valid, result = self.valid_result(el, equiv, data)
			if not valid:
				Timers.tic('StateManager.check.refine')
				assert el.state.allows_refinement()
				refinement_index = self.strategy.get_index(el.state, self.property, equiv, data)
				for next_zono in el.state.refine(refinement_index):
					next_zono.propagate_up_to_split(self.networks)
					self.enumeration_stack.append(EnumerationStackElement(next_zono))
				result = True
				Timers.toc('StateManager.check.refine')
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
			r1 = self.networks[0].execute(np.array(data[1], dtype=np.float32))
			r2 = self.networks[1].execute(np.array(data[1], dtype=np.float32))
			found, nndata = self.property.check_out(r1, r2)
			if not found:
				print(f"\n[NEQUIV] {nndata}\n")
				print(r1)
				print(r2)
				# We found a counter-example -- that's it
				GLOBAL_STATE.VALID_DEPTH.append(el.state.depth)
				GLOBAL_STATE.RIGHT += 1
				GLOBAL_STATE.FINISHED_FRAC += el.state.workload
				Timers.toc('StateManager.valid_result')
				return (True, False)
			else:
				# print(f"\n[NEED_FALLBACK] {data[0]}\n")
				GLOBAL_STATE.NEED_REFINEMENT += 1
				Timers.toc('StateManager.valid_result')
				return (False, False)
		else:
			print(f"\n[EQUIV] {data[0]}\n", end="")
			GLOBAL_STATE.VALID_DEPTH.append(el.state.depth)
			GLOBAL_STATE.RIGHT += 1
			if hasattr(el.state, "overapprox_nodes"):
				print(f"[EQUIV_SUMMARIZES] {len(el.state.overapprox_nodes)}")
				GLOBAL_STATE.TREE_PARTS.append(len(el.state.overapprox_nodes))
			GLOBAL_STATE.FINISHED_FRAC += el.state.workload
			Timers.toc('StateManager.valid_result')
			return (True, True)
