import numpy as np

from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from .property import EquivalenceProperty
from ..zono_state import ZonoState


class EpsilonEquivalence(EquivalenceProperty):
	def __init__(self, epsilon, networks=[]):
		self.epsilon = epsilon

	def check(self, zono: ZonoState):
		Timers.tic('check_epsilon')
		output_zonos = zono.get_output_zonos()
		mat = output_zonos[0].mat_t - output_zonos[1].mat_t
		bias = output_zonos[0].center - output_zonos[1].center
		out = Zonotope(bias, mat, output_zonos[1].init_bounds)
		final_bounds_orig = out.box_bounds()
		outsize = mat.shape[0]
		final_bounds = np.abs(final_bounds_orig)
		pos = np.argmax(final_bounds)
		pos1 = pos // 2
		pos2 = pos % 2
		eps = final_bounds[pos1, pos2]
		if eps > self.epsilon:
			Timers.tic('check_epsilon_nonequiv_treatment')
			rv = None
			too_low = final_bounds_orig[:, 1] < -self.epsilon
			if too_low.any():
				# Definitely not equivalent!
				pos1 = too_low.nonzero()[0][0]
				pos2 = 1
			too_high = final_bounds_orig[:, 0] > self.epsilon
			if too_high.any():
				pos1 = too_high.nonzero()[0][0]
				pos2 = 0
			init_bounds_nparray = np.array(output_zonos[1].init_bounds, dtype=zono.zono.dtype)
			if pos2 == 0:
				# Return lower bound
				rv = np.where(out.mat_t[pos1] <= 0, init_bounds_nparray[:, 1], init_bounds_nparray[:, 0])
			else:
				# Return upper bound
				rv = np.where(out.mat_t[pos1] <= 0, init_bounds_nparray[:, 0], init_bounds_nparray[:, 1])
			Timers.toc('check_epsilon_nonequiv_treatment')
			Timers.toc('check_epsilon')
			return False, (eps, rv)
		else:
			Timers.toc('check_epsilon')
			return True, (eps, None)

	def fallback_check(self, zono):
		Timers.tic('check_epsilon_fallback')
		input_size = zono.lpi.get_num_cols()
		output_zonos = zono.get_output_zonos()
		mat = output_zonos[0].mat_t - output_zonos[1].mat_t
		bias = output_zonos[0].center - output_zonos[1].center
		max_eps = 0.0
		for i in range(mat.shape[0]):
			min_vec = zono.lpi.minimize(mat[i, :input_size])
			min_val = self.compute_deviation(min_vec, bias[i], mat[i], output_zonos[-1].init_bounds, 0,
			                                 output_zonos[-1].dtype)
			if min_val > self.epsilon or min_val < -self.epsilon:
				Timers.toc('check_epsilon_fallback')
				return False, (min_val, min_vec)
			max_vec = zono.lpi.minimize(-mat[i, :input_size])
			max_val = self.compute_deviation(max_vec, bias[i], mat[i], output_zonos[-1].init_bounds, 1,
			                                 output_zonos[-1].dtype)
			if max_val > self.epsilon or max_val < -self.epsilon:
				Timers.toc('check_epsilon_fallback')
				return False, (max_val, max_vec)
			max_eps = max(max_eps, abs(max_val), abs(min_val))
		Timers.toc('check_epsilon_fallback')
		return True, (max_eps, None)

	def allows_fallback(self, state):
		return True

	def check_out(self, r1, r2):
		return (np.abs(r1 - r2) < self.epsilon).all()
