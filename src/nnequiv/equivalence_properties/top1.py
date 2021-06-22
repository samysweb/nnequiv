import numpy as np

from nnenum.lpinstance import LpInstance, UnsatError
from nnenum.timerutil import Timers
from .property import EquivalenceProperty
from ..zono_state import ZonoState


class Top1Equivalence(EquivalenceProperty):

	def check(self, zono: ZonoState, use_exact=False):
		Timers.tic('check_top1_fallback')
		output_zonos = zono.get_output_zonos()
		mat0 = np.array(output_zonos[0].mat_t,dtype=np.float64)
		mat1 = np.array(output_zonos[1].mat_t,dtype=np.float64)
		bias0 = np.array(output_zonos[0].center,dtype=np.float64)
		bias1 = np.array(output_zonos[1].center,dtype=np.float64)
		lp_col_num = zono.lpi.get_num_cols()
		for j in range(mat0.shape[0]):
			lp = LpInstance(other_lpi=zono.lpi)
			#print(f"Trying {j}: ", end="")
			current_bias0 = bias0 - bias0[j]
			current_mat0 = mat0 - mat0[j]
			for k in range(mat0.shape[0]):
				if k == j:
					continue
				lp_row = current_mat0[k, :lp_col_num]
				alpha_row = current_mat0[k, lp_col_num:]
				ib = np.array(output_zonos[-1].init_bounds, dtype=output_zonos[-1].dtype)
				alpha_min = lp.compute_residual(alpha_row, ib[lp_col_num:])
				#print(lp_row)
				if np.count_nonzero(lp_row)!=0:
					lp.add_dense_row(lp_row, -current_bias0[k] - alpha_min)
				else:
					print("Skipping 0 row")
			if not lp.is_feasible():
				#print(f"Skipped")
				continue
			#else:
				#print(f"Checking...")
			current_bias1 = bias1 - bias1[j]
			current_mat1 = mat1 - mat1[j]
			for k in range(mat1.shape[0]):
				# if k==j:
				# 	continue
				try:
					max_vec = lp.minimize(-current_mat1[k, :lp_col_num], use_exact=use_exact)
				except UnsatError:
					Timers.toc('check_top1_fallback')
					raise UnsatError
				max_val = self.compute_deviation(max_vec, current_bias1[k], current_mat1[k], output_zonos[-1].init_bounds, 1,
				                                 output_zonos[-1].dtype)
				#max_val = current_bias1[k] + np.dot(current_mat1[k], max_vec)
				if max_val > 0:
					Timers.toc('check_top1_fallback')
					return False, (k, max_vec)

		Timers.toc('check_top1_fallback')
		return True, (None, None)

	def fallback_check(self, zono):
		raise NotImplementedError()

	def allows_fallback(self, state):
		return False

	def check_out(self, r1, r2):
		return np.argmax(r1) == np.argmax(r2)
