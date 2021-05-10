import sys

import numpy as np
import logging

from nnenum.lp_star import LpStar
from nnenum.onnx_network import load_onnx_network
from nnenum.settings import Settings
from nnenum.zonotope import Zonotope
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
	# assert len(inputShape) == 1, f"Multidimensional inputs are not yet supported: {inputShape}"
	inshape = np.prod(inputShape)
	print(f"INPUT SHAPE: {inshape}")
	generator = np.identity(inshape, dtype=np.float32)
	bias = np.zeros(inshape, dtype=np.float32)
	bounds = []
	for i in range(len(PROPERTY[index][1])):
		bounds.append((PROPERTY[index][1][i],PROPERTY[index][0][i]))
	return Zonotope(bias, generator, init_bounds=bounds)


def main():
	Settings.TIMING_STATS = True
	# TODO(steuber): Check out implications of this setting
	Settings.SPLIT_TOLERANCE = 1e-8

	net1File = sys.argv[1]
	net2File = sys.argv[2]
	property = sys.argv[3]
	epsilon = float(sys.argv[4])

	network1, network2 = load_networks(net1File, net2File)

	input = generateBox(network1.get_input_shape(),property)

	check_equivalence(network1, network2, input, EpsilonEquivalence(epsilon, networks=[network1,network2]))
	print("")
	Timers.print_stats()
	print("")



if __name__ == "__main__":
	main()
