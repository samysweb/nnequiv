class GlobalState:
	def __init__(self):
		self.RIGHT = 0
		self.WRONG = 0
		self.FINISHED_FRAC = 0
		self.VALID_DEPTH=[]
		self.INVALID_DEPTH=[]
		self.VALID_DEPTH_DECISION=[]


GLOBAL_STATE = GlobalState()