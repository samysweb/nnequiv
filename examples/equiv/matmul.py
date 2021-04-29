import numpy as np
import onnx

from onnx import numpy_helper

model = onnx.load("../ReluDiff-ICSE2020-Artifact/DiffNN-Code/nnet/ACASXU_run2a_1_1_batch_2000.onnx")
onnx.checker.check_model(model)
names = [i.name for i in model.graph.initializer]
print(names)
initializers = model.graph.initializer
load_weights = {}
for i in initializers:
	load_weights[i.name] = onnx.numpy_helper.to_array(i)

weights={}
for k in load_weights.keys():
	weights[k] = np.array(load_weights[k])

path = [(0, 1, 14, False), (0, 3, 3, True), (0, 3, 17, True), (0, 3, 30, True), (0, 5, 10, False), (0, 5, 13, False), (0, 9, 8, True)] #, (1, 1, 14, False), (1, 5, 13, False), (1, 7, 5, False), (1, 9, 8, True)]
actual = []
for p in path:
	actual.append((int((p[1]-1)/2), p[2], p[3]))
print(actual)

# Adjust matricies:
for m in actual:
	if not m[2]:
		weights["W"+str(m[0])][m[1]]=0.0
		weights["B" + str(m[0])][m[1]] = 0.0

W = np.identity(5)
B = np.zeros(5)

for i in range(7):
	W = np.dot(weights["W"+str(i)],W)
	B = np.dot(weights["W"+str(i)],B) + weights["B"+str(i)]

print(W)
