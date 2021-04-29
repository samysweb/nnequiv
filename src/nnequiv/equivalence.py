from nnenum.network import NeuralNetwork
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from nnequiv.state_manager import StateManager
from nnequiv.zono_state import ZonoState


def make_init_zs(init, networks):
	zono_state = ZonoState(len(networks))
	zono_state.from_init_zono(init)

	zono_state.propagate_up_to_split(networks)

	return zono_state

def check_equivalence(network1 : NeuralNetwork, network2 : NeuralNetwork, input : Zonotope, equiv):
	Timers.reset()
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







def main_loop(manager : StateManager):
	while not manager.done():
		cur_state = manager.peek()
		if cur_state.is_finished(manager.get_networks()):
			manager.check(cur_state)
			manager.pop()
		else:
			manager.push(cur_state.advance_zono(manager.get_networks()))