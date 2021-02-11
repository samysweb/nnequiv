import sys

import numpy as np
import logging

from nnenum.lp_star import LpStar
from nnenum.onnx_network import load_onnx_network
from nnenum.settings import Settings
from nnequiv.equivalence import check_equivalence
from nnequiv.equivalence_properties import EpsilonEquivalence
from nnenum.timerutil import Timers

from properties import PROPERTY

logging.basicConfig(level=logging.INFO)


def load_networks(net1File: str, net2File: str):
	network1 = load_onnx_network(net1File)
	network2 = load_onnx_network(net2File)
	return network1, network2


def generateBox(inputShape, index):
	assert len(inputShape) == 1, "Multidimensional inputs are not yet supported"
	generator = np.identity(inputShape[0], dtype=np.float32)
	bias = np.zeros(inputShape[0], dtype=np.float32)
	#bounds = [(0,0.1)]*5
	#[(0,0.2),(0.085, 0.089),(0.01, 0.015),(0.09, 0.1),(0,0.006)]
	bounds = []
	# for i in range(inputShape[0]):
	#	bounds.append((low, high))
	for i in range(len(PROPERTY[index][1])):
		bounds.append((PROPERTY[index][1][i],PROPERTY[index][0][i]))
	return LpStar(generator, bias, box_bounds=bounds)


def main():
	Settings.TIMING_STATS = True
	# TODO(steuber): Check out implications of this setting
	Settings.CHECK_SINGLE_THREAD_BLAS = True
	Settings.BRANCH_MODE = Settings.BRANCH_EXACT
	Settings.SPLIT_TOLERANCE = 1e-8
	Settings.NUM_PROCESSES = 4  # if > 1, then force multiprocessing during lp step
	Settings.PARALLEL_ROOT_LP = True  # near the root of the search, use parallel lp, override NUM_LP_PROCESES if true
	Settings.EAGER_BOUNDS = True

	net1File = sys.argv[1]
	net2File = sys.argv[2]

	network1, network2 = load_networks(net1File, net2File)

	input = generateBox(network1.get_input_shape(),"1010")

	check_equivalence(network1, network2, input, EpsilonEquivalence(0.05, networks=[network1,network2]))
	print("")
	Timers.print_stats()
	print("")



if __name__ == "__main__":
	main()
