class GlobalState:
	def __init__(self):
		self.RIGHT = 0
		self.WRONG = 0
		self.FINISHED_FRAC = 0
		self.REFINED=0
		self.REFINE_LIMIT=0
		self.REFINE_DEPTH=[]
		self.REFINE_BRANCHING=[]


GLOBAL_STATE = GlobalState()