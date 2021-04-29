from nnenum.network import NeuralNetwork
from nnequiv.equivalence_properties import EquivalenceProperty
from nnequiv.zono_state import ZonoState


class EnumerationStackElement:
	def __init__(self, state: ZonoState, mat, center, next: [ZonoState] = []):
		# TODO(steuber): Move matrix/center handling from ZonoState into this class
		self.mat = mat
		self.center = center
		self.state = state
		self.next = next

	def get_next(self):
		if len(self.next) > 0:
			#TODO(steuber): Is this correct?
			return EnumerationStackElement(self.next[-1], self.mat, self.center, self.next[:-1])
		else:
			return None

	def get_state(self):
		return self.state

	def is_finished(self,networks):
		return self.state.is_finished(networks)

	def advance_zono(self, networks):
		new_el = self.state.do_first_relu_split(networks)
		if new_el is not None:
			self.next.append(new_el)
		self.state.next_layer()
		crossed, mat, center = self.state.propagate_up_to_split(networks)
		if crossed:  # We crossed a layer => new EnumerationStackElement
			return EnumerationStackElement(self.state, mat, center, [])
		else:
			return EnumerationStackElement(self.state, self.mat, self.center, [])


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

	def check(self, el: EnumerationStackElement):
		self.property.check(el)
