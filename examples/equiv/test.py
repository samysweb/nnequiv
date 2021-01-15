import sys
import numpy as np

from nnequiv.equivalence import check_equivalence
from nnenum.settings import Settings
from nnenum.onnx_network import load_onnx_network

from nnenum.lp_star import LpStar

def load_networks(net1File : str, net2File : str):
    network1 = load_onnx_network(net1File)
    network2 = load_onnx_network(net2File)
    return network1, network2

def generateBox(inputShape, low=0.0, high=0.1):
    assert len(inputShape)==1, "Multidimensional inputs are not yet supported"
    generator = np.identity(inputShape[0], dtype = np.float32)
    bias = np.zeros(inputShape[0], dtype = np.float32)
    bounds = []
    for i in range(0,inputShape[0]):
        bounds.append((low,high))
    return LpStar(generator, bias, box_bounds=bounds)

def main():
    Settings.TIMING_STATS = True
    # TODO(steuber): Check out implications of this setting
    Settings.CHECK_SINGLE_THREAD_BLAS=False
    Settings.BRANCH_MODE=Settings.BRANCH_EXACT
    Settings.SPLIT_TOLERANCE = 1e-8
    Settings.NUM_LP_PROCESSES = 1 # if > 1, then force multiprocessing during lp step
    Settings.PARALLEL_ROOT_LP = True # near the root of the search, use parallel lp, override NUM_LP_PROCESES if true
    Settings.EAGER_BOUNDS = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = False
    # Settings.GLPK_RESET_BEFORE_MINIMIZE = True # Necessary for trustworthy solutions?

    net1File = sys.argv[1]
    net2File = sys.argv[2]

    network1, network2 = load_networks(net1File, net2File)
    

    input = generateBox(network1.get_input_shape())

    
    check_equivalence(network1, network2, input)
    print("")




if __name__ == "__main__":
    main()