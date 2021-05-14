import copy

from nnenum.network import NeuralNetwork
from nnenum.timerutil import Timers
from nnequiv.equivalence_properties import EquivalenceProperty
from nnequiv.global_state import GLOBAL_STATE
from nnequiv.refinement import Refinement
from nnequiv.zono_state import ZonoState


class EnumerationStackElement:
	def __init__(self, state: ZonoState, hyperplane, rhs, next = []):
		# TODO(steuber): Move matrix/center handling from ZonoState into this class
		self.hyperplane = hyperplane
		self.rhs = rhs
		self.state = state
		self.next = next

	def get_next(self):
		if len(self.next) > 0:
			#TODO(steuber): Is this correct?
			return self.next[-1]
		else:
			return None

	def get_state(self):
		return self.state

	def is_finished(self,networks):
		return self.state.is_finished(networks)

	def advance_zono(self, networks):
		Timers.tic('advance_zono')
		new_el = None
		new_el, new_hyp, new_bias, old_hyp, old_bias = self.state.do_first_relu_split(networks)
		process = True
		if new_el is not None:
			if not self.state.active:
				self.state=new_el
				new_el = None
				process = False
			else:
				self.next.append(EnumerationStackElement(new_el, new_hyp, new_bias))
		if process:
			self.state.propagate_up_to_split(networks)
		if new_el is None:  # We crossed a layer => new EnumerationStackElement
			Timers.toc('advance_zono')
			return None
		else:
			Timers.toc('advance_zono')
			return EnumerationStackElement(self.state, old_hyp, old_bias, [])


class StateManager:
	def __init__(self, init: ZonoState, property: EquivalenceProperty, networks: [NeuralNetwork]):
		self.enumeration_stack = [EnumerationStackElement(init,None,None)]
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
		cur_pop = popped
		while True:
			next = cur_pop.get_next()
			if next is not None:
				self.push(next)
				break
			elif len(self.enumeration_stack)>0:
				cur_pop = self.enumeration_stack.pop()
			else:
				break

		return popped

	def push(self, el: EnumerationStackElement):
		self.enumeration_stack.append(el)

	def check(self, el: EnumerationStackElement, refine=True):
		if el.state.active:
			equiv, data = self.property.check(el.state)
			GLOBAL_STATE.VALID_DEPTH.append(el.state.depth)
			GLOBAL_STATE.RIGHT+=1
			GLOBAL_STATE.FINISHED_FRAC += el.state.workload
			if not equiv:
				r1 = self.networks[0].execute(data[1])
				r2 = self.networks[1].execute(data[1])
				if not self.property.check_out(r1,r2):
					print(f"\n[NEQUIV] {data[0]}\n")
					# We found a counter-example -- that's it
					return
				else:
					refinement = Refinement(el, self.property, self.networks, self.enumeration_stack)
					refinement.loop()
			else:
				print(f"\n[EQUIV] {data[0]}\n")
				return