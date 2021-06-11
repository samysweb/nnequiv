import signal

from nnenum.network import NeuralNetwork
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from nnequiv.global_state import GLOBAL_STATE
from nnequiv.state_manager import StateManager
from nnequiv.zono_state import ZonoState, status_update


def make_init_zs(init, networks):
	zono_state = ZonoState(len(networks))
	zono_state.from_init_zono(init)

	zono_state.propagate_up_to_split(networks)

	return zono_state

def check_equivalence(network1 : NeuralNetwork, network2 : NeuralNetwork, input : Zonotope, equiv):
	if not Settings.TIMING_STATS:
		Timers.disable()

	Timers.tic('network_equivalence')
	assert network1.get_input_shape() == network2.get_input_shape(), "Networks must have same input shape"
	assert network1.get_output_shape() == network2.get_output_shape(), "Networks must have same output shape"
	network1.check_io()
	network2.check_io()
	networks = [network1, network2]
	init = make_init_zs(input, networks)

	manager = StateManager(init, equiv, networks)

	main_loop(manager)

	Timers.toc('network_equivalence')



class GracefulKiller:
	kill_now = False
	def __init__(self):
		signal.signal(signal.SIGINT, self.exit_gracefully)
		signal.signal(signal.SIGTERM, self.exit_gracefully)

	def exit_gracefully(self,signum, frame):
		print("\nEXITING...")
		Timers.tocRec()
		Timers.print_stats()
		self.kill_now = True



def main_loop(manager : StateManager):
	counter = 0
	killer = GracefulKiller()
	while not manager.done() and not killer.kill_now:
		cur_state = manager.pop()
		if not cur_state.state.active:
			continue
		if cur_state.is_finished(manager.get_networks()):
			if not cur_state.state.check_feasible(None, None):
				continue
			if not manager.check(cur_state):
				print(f"NETWORKS NOT EQUIVALENT")
				break
		else:
			newStackEl = cur_state.advance_zono(manager.get_networks())
			if newStackEl is not None and newStackEl.state.active:
				manager.push(newStackEl)
			if cur_state.state.active:
				manager.push(cur_state)
		counter+=1
		if counter%5000==1:
			status_update()
	status_update()
