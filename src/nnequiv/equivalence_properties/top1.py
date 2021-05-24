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

	def fallback_check(self, zono):
		Timers.tic('check_top1_fallback')
		mat0 = zono.output_zonos[0].mat_t
		mat1 = zono.output_zonos[1].mat_t
		bias0 = zono.output_zonos[0].center
		bias1 = zono.output_zonos[1].center
		for j in range(mat0.shape[0]):
			lp = LpInstance(other_lpi=zono.lpi)
			# print(f"Trying {j}: ", end="")
			current_bias0 = bias0 - bias0[j]
			current_mat0 = mat0 - mat0[j]
			for k in range(mat0.shape[0]):
				if k == j:
					continue
				if np.any(current_mat0[k]):
					lp.add_dense_row(current_mat0[k], -current_bias0[k])
				else:
					print(f"[TOP1_CHECK] Skipping row {k}")
			if not lp.is_feasible():
				continue
			current_bias1 = bias1 - bias1[j]
			current_mat1 = mat1 - mat1[j]
			for k in range(mat1.shape[0]):
				# if k==j:
				# 	continue
				max_vec = lp.minimize(-current_mat1[k])
				max_val = current_bias1[k] + np.dot(current_mat1[k], max_vec)
				if max_val > 0:
					Timers.toc('check_top1_fallback')
					return False, (k, max_vec[:self.input_size])

		Timers.toc('check_top1_fallback')
		return True, (None, None)

	def check_out(self, r1, r2):
		return np.argmax(r1) == np.argmax(r2), None
