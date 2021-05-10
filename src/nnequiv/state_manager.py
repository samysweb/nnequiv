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

	# def is_valid(self, zono_state):
	# 	for el in self.enumeration_stack:
	# 		if el.hyperplane is not None:
	# 			min_val = zono_state.output_zonos[1].minimize_box(el.hyperplane)
	# 			if min_val>el.rhs:
	# 				# Invalid Zonotope
	# 				return False
	#
	# 	return True
	#
	# def split(self,el):
	# 	split_dim = el.state.split_heuristic.get_split()
	# 	up = el.state.output_zonos[1].init_bounds[split_dim][1]
	# 	low = el.state.output_zonos[1].init_bounds[split_dim][0]
	# 	mid = low + (up - low) / 2
	# 	return split_dim, low, mid, up
	#
	# def update(self,state,i,low,up):
	# 	state.output_zonos[1].init_bounds_nparray = None
	# 	state.output_zonos[1].init_bounds[i]=(low,up)
	# 	if state.output_zonos[1].neg1_gens is not None:
	# 		state.output_zonos[1].neg1_gens[i] = low
	# 		state.output_zonos[1].pos1_gens[i] = up
	#
	# def refine(self, el, networks, data):
	# 	Timers.tic("refine_zono")
	# 	check_state = ZonoState(2,state=el.state)
	# 	i, low,mid,up = self.split(el)
	# 	self.update(check_state,i,low,mid)
	# 	other_split_dirs=[(i,low,mid,up)]
	# 	cur_i=0
	# 	cur_low=low
	# 	cur_up=up
	# 	finished = 0
	# 	split_num=1
	# 	while True:
	# 		if not self.is_valid(check_state):
	# 			finished+=1
	# 			if len(other_split_dirs)==0:
	# 				break
	# 			i,low,mid,up = other_split_dirs.pop()
	# 			if i == cur_i:
	# 				self.update(check_state,i,mid,up)
	# 			else:
	# 				self.update(check_state,cur_i,cur_low,cur_up)
	# 				cur_i=i
	# 				cur_low = low
	# 				cur_up = up
	# 				self.update(check_state,i,mid,up)
	# 			print("infeasible")
	# 			continue
	# 		equiv, data = self.property.check(check_state)
	# 		if not equiv:
	# 			r1 = self.networks[0].execute(data[1])
	# 			r2 = self.networks[1].execute(data[1])
	# 			if not self.property.check_out(r1, r2):
	# 				finished+=1
	# 				print(f"\n[NEQUIV] {data[0]}\n")
	# 				# We found a counter-example -- that's it
	# 				Timers.toc("refine_zono")
	# 				return False
	# 			else:
	# 				split_num+=1
	# 				if i%100 == 0:
	# 					print(f"Splitting {i} ({data[0]})")
	# 				i, low, mid, up = self.split(el)
	# 				self.update(check_state,i,low,mid)
	# 				other_split_dirs.append((i,low,mid,up))
	# 				continue
	# 		else:
	# 			finished+=1
	# 			print(f"\n[EQUIV] {data[0]}\n")
	# 			if len(other_split_dirs)==0:
	# 				break
	# 			i,low,mid,up = other_split_dirs.pop()
	# 			self.update(check_state,i,mid,up)
	# 	Timers.toc("refine_zono")
	# 	print(f"Finished: {finished}")
	# 	print(f"Split Num: {split_num}")
	# 	return True
	# 	#TODO(steuber): Refinement loop