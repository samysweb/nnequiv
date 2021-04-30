import numpy as np

from nnenum.zonotope import Zonotope
from .property import EquivalenceProperty
from ..zono_state import ZonoState


class EpsilonEquivalence(EquivalenceProperty):
	def __init__(self, epsilon, networks=[]):
		self.epsilon = epsilon

	def check(self, zono : ZonoState):
		mat = zono.state.output_zonos[0].mat_t - zono.state.output_zonos[1].mat_t
		bias = zono.state.output_zonos[0].center - zono.state.output_zonos[1].center
		out = Zonotope(bias, mat, zono.state.output_zonos[1].init_bounds)
		final_bounds = out.box_bounds()
		outsize = mat.shape[0]
		eps = abs(np.max(final_bounds))
		if eps > self.epsilon:
			print(f"[NEQUIV] {eps}")
			return False, eps
		else:
			print(f"[EQUIV] {eps}")
			return True, eps
