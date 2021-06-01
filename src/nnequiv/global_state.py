class GlobalState:
	def __init__(self):
		self.RIGHT = 0
		self.WRONG = 0
		self.NEED_REFINEMENT = 0
		self.MAX_DEPTH = 0
		self.REFINEMENT_AVG=None
		self.REFINEMENT_AVG_N=0
		self.OVERAPPOXED_RIGHT=0
		self.FINISHED_FRAC = 0
		self.VALID_DEPTH = []
		self.INVALID_DEPTH = []
		self.VALID_DEPTH_DECISION = []
		self.PASSED_FIRST = 0


GLOBAL_STATE = GlobalState()
