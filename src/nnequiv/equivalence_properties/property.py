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
