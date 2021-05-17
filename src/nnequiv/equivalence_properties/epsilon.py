import numpy as np

from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from .property import EquivalenceProperty
from ..zono_state import ZonoState, BranchDecision


class EpsilonEquivalence(EquivalenceProperty):
	def __init__(self, epsilon, input_size, networks=[]):
		self.epsilon = epsilon
		self.input_size = input_size

	def check(self, zono: ZonoState):
		Timers.tic('check_epsilon')
		bias, init_bounds, mat = self.build_out_zono(zono)
		out = Zonotope(bias, mat, init_bounds)
		final_bounds_orig = out.box_bounds()
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
			init_bounds_nparray = np.array(init_bounds, dtype=zono.zono.dtype)
			if pos2 == 0:
				# Return lower bound
				rv = np.where(out.mat_t[pos1] <= 0, init_bounds_nparray[:, 1], init_bounds_nparray[:, 0])
			else:
				# Return upper bound
				rv = np.where(out.mat_t[pos1] <= 0, init_bounds_nparray[:, 0], init_bounds_nparray[:, 1])
			Timers.toc('check_epsilon_nonequiv_treatment')
			Timers.toc('check_epsilon')
			return False, (eps, rv[:self.input_size])
		else:
			Timers.toc('check_epsilon')
			return True, (eps, None)

	def build_out_zono(self, zono):
		Timers.tic('build_out_zono')
		outdim = zono.output_zonos[1].mat_t.shape[0]
		if self.input_size != zono.output_zonos[0].mat_t.shape[1] or self.input_size != \
				zono.output_zonos[1].mat_t.shape[1]:
			init_bounds = zono.output_zonos[1].init_bounds[:self.input_size]
			mat0 = zono.output_zonos[0].mat_t
			mat1 = zono.output_zonos[1].mat_t[:, :self.input_size]
			if len(zono.output_zonos[0].init_bounds) > self.input_size:
				init_bounds = np.concatenate((init_bounds, zono.output_zonos[0].init_bounds[self.input_size:]))
				mat1 = np.hstack((mat1,
				                  np.zeros((outdim, zono.output_zonos[0].mat_t.shape[1] - self.input_size))))
			if len(zono.output_zonos[1].init_bounds) > self.input_size:
				init_bounds = np.concatenate((init_bounds, zono.output_zonos[1].init_bounds[self.input_size:]))
				mat0 = np.hstack((mat0,
				    np.zeros((outdim, zono.output_zonos[1].mat_t.shape[1] - self.input_size))))
				mat1 = np.hstack((mat1,
					 zono.output_zonos[1].mat_t[:, self.input_size:]))
			mat = mat0 - mat1
			bias = zono.output_zonos[0].center - zono.output_zonos[1].center
		else:
			mat = zono.output_zonos[0].mat_t - zono.output_zonos[1].mat_t
			bias = zono.output_zonos[0].center - zono.output_zonos[1].center
			init_bounds = zono.output_zonos[1].init_bounds
		Timers.toc('build_out_zono')
		return bias, init_bounds, mat

	def compute_deviation(self, zono, vec, i, bias, mat, init_bounds, dev):
		Timers.tic('compute_deviation')
		if self.input_size == zono.output_zonos[0].mat_t.shape[1] and self.input_size == \
				zono.output_zonos[1].mat_t.shape[1]:
			Timers.toc('compute_deviation')
			return bias[i] + np.dot(mat[i, :self.input_size], vec)
		ib = np.array(init_bounds[self.input_size:], dtype=zono.zono.dtype)
		vals = np.where(mat[i, self.input_size:] <= 0, ib[:, 1 - dev], ib[:, dev])
		invec = np.concatenate((vec, vals))
		rv = bias[i] + np.dot(mat[i], invec)
		Timers.toc('compute_deviation')
		return rv

	def fallback_check(self, zono):
		Timers.tic('check_epsilon_fallback')
		bias, init_bounds, mat = self.build_out_zono(zono)
		max_eps = 0.0
		for i in range(mat.shape[0]):
			min_vec = zono.lpi.minimize(mat[i, :self.input_size])
			min_val = self.compute_deviation(zono, min_vec, i, bias, mat, init_bounds, 0)
			if min_val > self.epsilon or min_val < -self.epsilon:
				Timers.toc('check_epsilon_fallback')
				return False, (min_val, min_vec[:self.input_size])
			max_vec = zono.lpi.minimize(-mat[i, :self.input_size])
			max_val = self.compute_deviation(zono, max_vec, i, bias, mat, init_bounds, 1)
			if max_val > self.epsilon or max_val < -self.epsilon:
				Timers.toc('check_epsilon_fallback')
				return False, (max_val, max_vec[:self.input_size])
			max_eps = max(max_eps, abs(max_val), abs(min_val))
		Timers.toc('check_epsilon_fallback')
		return True, (max_eps, None)

	def refine_resubmit(self, el, equiv, data):
		Timers.tic('refine_resubmit')
		bias, init_bounds, mat = self.build_out_zono(el.state)
		vals = np.sum(np.abs(mat[:, self.input_size:]), axis=0)
		max_index = np.argmax(vals)
		# min_index = np.argmin(vals)
		node = el.state.overapprox_nodes[max_index]
		branches = []
		added_node = False
		new_branch_decision = BranchDecision(node.cur_network, node.cur_layer, node.index, BranchDecision.BOTH)
		for branch in reversed(el.state.branching):
			if branch < new_branch_decision \
					and not added_node:
				branches.append(new_branch_decision)
				added_node = True
			branches.append(branch)
		if not added_node:
			branches.append(new_branch_decision)
		rv = ZonoState(el.state.network_count, branch_on=branches)
		rv.workload = el.state.workload
		rv.from_init_zono(el.state.initial_zono)
		Timers.toc('refine_resubmit')
		return [rv]

	def check_out(self, r1, r2):
		val = np.abs(r1 - r2)
		return (val < self.epsilon).all(), np.max(val)
