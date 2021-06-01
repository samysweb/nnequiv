import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.timerutil import Timers
from .property import EquivalenceProperty
from ..zono_state import ZonoState


class Top1Equivalence(EquivalenceProperty):

	def __init__(self, input_size):
		self.input_size = input_size

	def check(self, zono: ZonoState):
		return self.fallback_check(zono)

	def has_fallback(self, zono):
		return False

	def fallback_check(self, zono):
		Timers.tic('check_top1_fallback')
		output_zonos = zono.get_output_zonos()
		mat0 = output_zonos[0].mat_t
		mat1 = output_zonos[1].mat_t
		bias0 = output_zonos[0].center
		bias1 = output_zonos[1].center
		ib = np.array(output_zonos[1].init_bounds, dtype=output_zonos[1].dtype)
		for j in range(mat0.shape[0]):
			lp = LpInstance(other_lpi=zono.lpi)
			# print(f"Trying {j}: ", end="")
			current_bias0 = bias0 - bias0[j]
			current_mat0 = mat0 - mat0[j]
			for k in range(mat0.shape[0]):
				if k == j:
					continue
				if np.any(current_mat0[k]):
					alpha_min = lp.compute_residual(current_mat0[k,self.input_size:], ib[self.input_size:])
					lp.add_dense_row(current_mat0[k,:self.input_size], -current_bias0[k]-alpha_min)
				else:
					print(f"[TOP1_CHECK] Skipping row {k}")
			if not lp.is_feasible():
				continue
			current_bias1 = bias1 - bias1[j]
			current_mat1 = mat1 - mat1[j]
			for k in range(mat1.shape[0]):
				# if k==j:
				# 	continue
				max_vec = lp.minimize(-current_mat1[k,:self.input_size])
				max_val = self.compute_deviation(max_vec, current_bias1[k], current_mat1[k],ib,1, zono.zono.dtype)
				if max_val > 0:
					Timers.toc('check_top1_fallback')
					return False, (k, max_vec[:self.input_size])

		Timers.toc('check_top1_fallback')
		return True, (None, None)

	def check_out(self, r1, r2):
		return np.argmax(r1) == np.argmax(r2), None
