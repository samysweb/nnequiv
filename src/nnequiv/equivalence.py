
import numpy as np
import time


from nnenum.timerutil import Timers
from nnenum.settings import Settings
from nnenum.network import NeuralNetwork
from nnenum.lp_star import LpStar
from nnenum.enumerate import make_init_ss, SharedState, PrivateState
from nnenum.prefilter import LpCanceledException

from nnequiv.worker import EquivWorker
from nnequiv.lp_star_state import EquivStarState



def make_init_ss(init, network1, network2, start_time):
    'make the initial star state'

    network_inputs = network1.get_num_inputs()
    network_outputs = network1.get_num_outputs()

    ss = EquivStarState(2)
    ss.from_init_star(init)

    assert len(ss.star.init_bias) == network_inputs, f"init_bias len: {len(ss.star.init_bias)}" + \
        f", network inputs: {network_inputs}"

    ss.should_try_overapprox = False

    # propagate the initial star up to the first split
    timer_name = Timers.stack[-1].name if Timers.stack else None

    try: # catch lp timeout
        Timers.tic('propagate_up_to_split')
        ss.propagate_up_to_split([network1, network2], start_time)
        Timers.toc('propagate_up_to_split')
    except LpCanceledException:
        while Timers.stack and Timers.stack[-1].name != timer_name:
            Timers.toc(Timers.stack[-1].name)

        ss = None

    return ss


def check_equivalence(network1 : NeuralNetwork, network2 : NeuralNetwork, input : LpStar):

    Timers.reset()
    if not Settings.TIMING_STATS:
        Timers.disable()
    start_time = time.perf_counter()

    assert network1.get_input_shape() == network2.get_input_shape(), "Networks must have same input shape"
    assert network1.get_output_shape() == network2.get_output_shape(), "Networks must have same output shape"
    network1.check_io()
    network2.check_io()

    init_star_state = make_init_ss(input, network1 ,network2, start_time)

    shared = EquivSharedState(network1, network2, None, 1, start_time)
    shared.push_init(init_star_state)

    worker_func(0, shared)

def worker_func(worker_index, shared):
    'worker function during verification'

    np.seterr(all='raise') # raise exceptions on floating-point errors instead of printing warnings

    if shared.multithreaded:
        assert False, "No multithreadding support yet"
        # reinit_onnx_sessions(shared.network)
        # Timers.stack.clear() # reset inherited Timers
        # tag = f" (Process {worker_index})"
    else:
        tag = ""

    timer_name = f'worker_func{tag}'

    Timers.tic(timer_name)

    priv = PrivateState(worker_index)
    priv.start_time = shared.start_time
    w = EquivWorker(shared, priv)
    w.main_loop()
    Timers.toc(timer_name)


class EquivSharedState(SharedState):

    def __init__(self, network1, network2, spec, num_workers, start_time):
        self._fully_initialized=False
        super(EquivSharedState, self).__init__(network1, spec, num_workers, start_time)
        self.network1 = network1
        self.network2 = network2
        self._fully_initialized=True
        self.freeze_attrs()
    
    def freeze_attrs(self):
        if self._fully_initialized:
            self._frozen = True
