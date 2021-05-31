import numpy as np

from nnenum.timerutil import Timers
from nnequiv.overapprox import OverapproxZonoState
from nnequiv.zono_state import ZonoState


class RefinementStrategy:
	def get_index(self, state, property, equiv, data):
		pass


class RefineFirst(RefinementStrategy):
	def __init__(self):
		pass

	def get_index(self, state, property, equiv, data):
		# bias, init_bounds, mat = self.build_out_zono(el.state)
		# vals = np.sum(np.abs(mat[:, self.input_size:]), axis=0)
		# max_index = np.argmax(vals)
		# min_index = np.argmin(vals)
		return 0

class RefineNewTopDown(RefinementStrategy):
	def __init__(self):
		pass

	def get_index(self, state:OverapproxZonoState, property, equiv, data):
		Timers.tic('refine_new_top_down_get_index')
		input_size = state.initial_zono.mat_t.shape[1]
		alpha_count = len(state.overapprox_nodes)
		counter_example = np.array(data[1])
		counter_example_correct = np.array(counter_example)
		correct_alphas = np.zeros((alpha_count,))
		alpha_delta = np.zeros((alpha_count,))
		bias, init_bounds, mat = property.build_out_zono(state)
		assert input_size+alpha_count == counter_example.shape[0], f"mismatch {input_size} {alpha_count} {counter_example.shape[0]}"
		for node in state.overapprox_nodes:
			hyperplane_dim = node.hyperplane.shape[0]
			assert node.coefficient_index >= hyperplane_dim
			relu_in_res = np.dot(node.hyperplane,counter_example_correct[:hyperplane_dim]) + node.bias
			relu_out_res = max(0.0, relu_in_res)
			relaxed_res = node.factor*relu_in_res + counter_example_correct[node.coefficient_index]
			alpha_delta[node.coefficient_index-input_size] = relu_out_res-relaxed_res
			counter_example_correct[node.coefficient_index] +=alpha_delta[node.coefficient_index-input_size]
			correct_alphas[node.coefficient_index-input_size] = counter_example_correct[node.coefficient_index]
		rv = np.dot(mat,counter_example)+bias
		delta_change = np.dot(rv, mat[:,input_size:])*alpha_delta
		delta_change_abs = np.abs(delta_change)
		delta_change_sorted = np.argsort(-delta_change_abs)
		Timers.toc('refine_new_top_down_get_index')
		return delta_change_sorted[:min(30,len(delta_change_sorted))]

class RefineMax(RefinementStrategy):
	def __init__(self):
		pass

	def get_index(self, state, property, equiv, data):
		bias, init_bounds, mat = property.build_out_zono(state)
		vals = np.sum(np.abs(mat[:, property.input_size:]), axis=0)
		max_index = np.argmax(vals)
		return max_index
