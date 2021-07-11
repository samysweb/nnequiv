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
from nnequiv.equivalence_properties.milptop1 import MilpTop1Equivalence
from nnequiv.equivalence_properties.top1 import Top1Equivalence

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
	Timers.reset()
	Timers.tic('main')
	Timers.tic('main_init')
	Settings.TIMING_STATS = True
	# TODO(steuber): Check out implications of this setting
	Settings.SPLIT_TOLERANCE = 1e-8
	Settings.PARALLEL_ROOT_LP = False
	Settings.NUM_PROCESSES = 1

	net1File = sys.argv[1]
	net2File = sys.argv[2]
	property = sys.argv[3]
	strategy = sys.argv[5]
	if strategy not in Settings.EQUIV_STRATEGIES and not strategy == "CEGAR_OPTIMAL":
		print(f"ERROR: Strategy {strategy} unknown", file=sys.stderr)
		return
	elif strategy == "CEGAR_OPTIMAL":
		Settings.EQUIV_OVERAPPROX_STRAT = "CEGAR"
		Timers.toc('main_init')
		main_cegar_optimal(net1File, net2File, property)
	else:
		Settings.EQUIV_OVERAPPROX_STRAT = strategy
		if strategy.startswith("REFINE_UNTIL"):
			Settings.EQUIV_OVERAPPROX_STRAT_REFINE_UNTIL=True
		Timers.toc('main_init')
		main_normal(net1File, net2File, property)
	main_time = Timers.toc('main')
	print(f"\n[MAIN_TIME] {main_time}")
	print("")
	Timers.print_stats()
	print("")


def main_normal(net1File, net2File, property):
	Timers.tic('net_load')
	network1, network2 = load_networks(net1File, net2File)
	Timers.toc('net_load')
	Timers.tic('property_create')
	if sys.argv[4] == "top":
		equivprop = Top1Equivalence()
	elif sys.argv[4] == "mtop":
		equivprop = MilpTop1Equivalence()
	else:
		epsilon = float(sys.argv[4])
		equivprop = EpsilonEquivalence(epsilon, networks=[network1, network2])
	Timers.toc('property_create')
	Timers.tic('generate_box')
	input = generateBox(network1.get_input_shape(), property)
	Timers.toc('generate_box')
	check_equivalence(network1, network2, input, equivprop)

def main_cegar_optimal(net1File, net2File, property):
	Timers.tic('net_load')
	network1, network2 = load_networks(net1File, net2File)
	Timers.toc('net_load')
	Timers.tic('property_create')
	if sys.argv[4] == "top":
		equivprop = Top1Equivalence()
	elif sys.argv[4] == "mtop":
		equivprop = MilpTop1Equivalence()
	else:
		epsilon = float(sys.argv[4])
		equivprop = EpsilonEquivalence(epsilon, networks=[network1, network2])
	Timers.toc('property_create')

	Timers.tic('cegar_run')
	Timers.tic('generate_box')
	input = generateBox(network1.get_input_shape(), property)
	Timers.toc('generate_box')
	check_equivalence(network1, network2, input, equivprop)
	cegar_time = Timers.toc('cegar_run')
	print(f"\n[CEGAR_TIME] {cegar_time}\n")

	Timers.tic('optimal_run')
	Settings.EQUIV_OVERAPPROX_STRAT = "OPTIMAL"
	Timers.tic('generate_box')
	input = generateBox(network1.get_input_shape(), property)
	Timers.toc('generate_box')
	check_equivalence(network1, network2, input, equivprop)
	optimal_time = Timers.toc('optimal_run')
	print(f"\n[OPTIMAL_TIME] {optimal_time}\n")


if __name__ == "__main__":
	main()
