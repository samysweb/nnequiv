import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from .property import EquivalenceProperty
from ..zono_state import ZonoState


class Top1Equivalence(EquivalenceProperty):

	def check(self, zono : ZonoState):
		Timers.tic('check_top1_fallback')
		mat0 = zono.output_zonos[0].mat_t
		mat1 = zono.output_zonos[1].mat_t
		bias0 = zono.output_zonos[0].center
		bias1 = zono.output_zonos[1].center
		for j in range(mat0.shape[0]):
			lp = LpInstance(other_lpi=zono.lpi)
			print(f"Trying {j}: ",end="")
			current_bias0 = bias0 - bias0[j]
			current_mat0 = mat0 - mat0[j]
			for k in range(mat0.shape[0]):
				if k==j:
					continue
				lp.add_dense_row(current_mat0[k],-current_bias0[k])
			if not lp.is_feasible():
				print(f"Skipped")
				continue
			else:
				print(f"Checking...")
			current_bias1 = bias1 - bias1[j]
			current_mat1 = mat1 - mat1[j]
			for k in range(mat1.shape[0]):
				# if k==j:
				# 	continue
				max_vec =lp.minimize(-current_mat1[k])
				max_val = current_bias1[k]+np.dot(current_mat1[k],max_vec)
				if max_val>0:
					Timers.toc('check_top1_fallback')
					return False,(k,max_vec)

		Timers.toc('check_top1_fallback')
		return True, (None, None)

	def fallback_check(self, zono):
		raise NotImplementedError()

	def allows_fallback(self, state):
		return False

	def check_out(self, r1, r2):
		return np.argmax(r1)==np.argmax(r2)