import numpy as np


class RefinementStrategy:
	def get_index(self, state, property, equiv, data):
		pass


class RefineFirst(RefinementStrategy):
	def __init__(self):
		pass

	def get_index(self, state, property, equiv, data):
		# bias, init_bounds, mat = self.build_out_zono(el.state)
		# vals = np.sum(np.abs(mat[:, self.input_size:]), axis=0)
		# max_index = np.argmax(vals)
		# min_index = np.argmin(vals)
		return 0


class RefineMax(RefinementStrategy):
	def __init__(self):
		pass

	def get_index(self, state, property, equiv, data):
		bias, init_bounds, mat = property.build_out_zono(state)
		vals = np.sum(np.abs(mat[:, property.input_size:]), axis=0)
		max_index = np.argmax(vals)
		return max_index
