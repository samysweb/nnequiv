import numpy as np

from nnenum.timerutil import Timers


class EquivalenceProperty:
	def check(self, zono):
		"""
		Approximative check of equivalence
		:param zono: ZonoState to check for equivalence
		:return: Tuple rv: rv[0] is True if equivalence was found. rv[1] contains additional data
		"""
		pass

	def fallback_check(self, zono):
		"""
		Fallback check for cases where approximative check was insufficient
		:param zono:
		:return:
		"""
		pass

	def check_out(self, r1, r2):
		"""
		Check if property holds for the given output vectors
		:param r1: Output Vector 1
		:param r2: Output Vector 2
		:return: True iff the property holds for r1 and r2
		"""
		pass

	def has_fallback(self, zono):
		pass

	def compute_deviation(self, vec, bias, mat_row, init_bounds, minmax, dtype):
		vec_size = vec.shape[0]
		alpha_row = mat_row[vec_size:]
		ib = np.array(init_bounds, dtype=dtype)
		min_factors = np.where(alpha_row <= 0, ib[vec_size:, 1-minmax], ib[vec_size:, minmax])
		alpha_dev = min_factors.dot(alpha_row)
		res_vec = np.concatenate((vec,min_factors))
		return (np.dot(vec, mat_row[:vec_size])+alpha_dev+bias, res_vec)
