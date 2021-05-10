from nnequiv.zono_state import ZonoState


class Refinement:
	def __init__(self, el, property, networks, enumeration_stack):
		self.property = property
		self.zono_state = ZonoState(len(networks),state=el.state)
		self.zono_state_backup = ZonoState(len(networks), state=el.state)
		self.split_dirs = []
		self.networks = networks
		self.enumeration_stack = enumeration_stack
		self.split_num=0
		self.finished=0
		self.done=False
		self.found_concrete = False
		self.cur_load=1
		self.finished_load=0
		self.invalid=0
		self.valid=0
		self.forward()

	def is_valid(self):
		for el in self.enumeration_stack:
			if el.hyperplane is not None:
				min_val = self.zono_state.output_zonos[1].minimize_box(el.hyperplane)
				if min_val-el.rhs > 1e-6 :
					# Invalid Zonotope
					return False

		return True

	def forward(self):
		self.split_num+=1
		i, low, mid, high = self.split()
		self.cur_load/=2
		self.split_dirs.append((i,low,mid,high,True))
		self.update(i,low,mid)

	def backtrack(self):
		while len(self.split_dirs)>0:
			i,low,mid,high,descend = self.split_dirs.pop()
			if descend:
				self.update(i,mid,high)
				self.split_dirs.append((i,low,mid,high,False))
				break
			else:
				# Reset current level
				self.cur_load*=2
				self.update(i,low,high)
		if len(self.split_dirs)==0:
			self.done=True

	def step(self):
		if not self.is_valid():
			self.finished+=1
			self.invalid+=1
			self.finished_load+=self.cur_load
			self.backtrack()
		else:
			equiv, data = self.property.check(self.zono_state)
			if not equiv:
				r1 = self.networks[0].execute(data[1])
				r2 = self.networks[1].execute(data[1])
				if not self.property.check_out(r1, r2):
					self.finished += 1
					self.finished_load += self.cur_load
					print(f"\n[NEQUIV] {data[0]}\n")
					# We found a counter-example -- that's it
					self.done=True
				else:
					self.forward()
			else:
				self.finished+=1
				self.valid+=1
				self.finished_load += self.cur_load
				self.found_concrete = True
				print(f"\n[EQUIV] {data[0]}\n")
				self.backtrack()

	def loop(self):
		counter = 0
		while not self.done:
			self.step()
			counter+=1
			if counter%100 == 0:
				self.status_update()
		self.status_update()
		print(f"Finished: {self.finished}")
		print(f"Splits: {self.split_num}")
		assert self.found_concrete

	def status_update(self):
		total = self.valid + self.invalid
		expected = int(total / self.finished_load)
		percentage = float(total) / float(expected) * 100
		print(
			f"\r[REFINE] Invalid: {self.invalid} | Valid: {self.valid} | Total: {total} | Expected {expected} ({percentage}%)",
			end="")

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
