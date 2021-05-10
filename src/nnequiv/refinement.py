from nnequiv.zono_state import ZonoState


class Refinement:
	def __init__(self, el, property, networks, enumeration_stack):
		self.property = property
		self.zono_state = ZonoState(len(networks),state=el.state)
		self.split_dirs = []
		self.networks = networks
		self.enumeration_stack = enumeration_stack
		self.cur_i = None
		self.cur_low = None
		self.cur_high = None
		self.split_num=0
		self.finished=0
		self.done=False
		self.found_concrete = False
		self.forward()

	def is_valid(self):
		for el in self.enumeration_stack:
			if el.hyperplane is not None:
				min_val = self.zono_state.output_zonos[1].minimize_box(el.hyperplane)
				if min_val>el.rhs:
					# Invalid Zonotope
					return False

		return True

	def forward(self):
		self.split_num+=1
		i, low, mid, high = self.split()
		self.cur_i = i
		self.cur_low = low
		self.cur_high = high
		self.split_dirs.append((i,low,mid,high))
		self.update(i,low,mid)

	def backtrack(self):
		if len(self.split_dirs)==0:
			self.done=True
			return
		i,low,mid,high = self.split_dirs.pop()
		if self.cur_i == i:
			self.update(i,mid,high)
		else:
			# Reset current level
			self.update(self.cur_i,self.cur_low,self.cur_high)
			self.cur_i = i
			self.cur_low = low
			self.cur_high = high
			# Enter other branch from level higher up
			self.update(i,mid,high)

	def step(self):
		if not self.is_valid():
			self.finished+=1
			print(f"invalid")
			self.backtrack()
		else:
			equiv, data = self.property.check(self.zono_state)
			if not equiv:
				r1 = self.networks[0].execute(data[1])
				r2 = self.networks[1].execute(data[1])
				if not self.property.check_out(r1, r2):
					self.finished += 1
					print(f"\n[NEQUIV] {data[0]}\n")
					# We found a counter-example -- that's it
					self.done=True
				else:
					self.forward()
			else:
				self.finished+=1
				self.found_concrete = True
				print(f"\n[EQUIV] {data[0]}\n")
				self.backtrack()

	def loop(self):
		while not self.done:
			self.step()
		assert self.found_concrete
		print(f"Finished: {self.finished}")
		print(f"Splits: {self.split_num}")

	def update(self,i,low,up):
		self.zono_state.output_zonos[1].init_bounds_nparray = None
		self.zono_state.output_zonos[1].init_bounds[i]=(low,up)
		if self.zono_state.output_zonos[1].neg1_gens is not None:
			self.zono_state.output_zonos[1].neg1_gens[i] = low
			self.zono_state.output_zonos[1].pos1_gens[i] = up

	def split(self):
		split_dim = self.zono_state.split_heuristic.get_split()
		up = self.zono_state.output_zonos[1].init_bounds[split_dim][1]
		low = self.zono_state.output_zonos[1].init_bounds[split_dim][0]
		mid = low + (up - low) / 2
		return split_dim, low, mid, up
