class EpsilonEquivalence:
	def __init__(self, epsilon, networks=None):
		self.epsilon = epsilon
		self.networks = networks

	def check(self, output_stars):
		mat = output_stars[0].a_mat - output_stars[1].a_mat
		bias = output_stars[0].bias - output_stars[1].bias
		output_stars[1].a_mat = mat
		output_stars[1].bias = bias
		for i in range(output_stars[1].a_mat.shape[0]):
			lower = output_stars[1].minimize_output(i)
			if (lower < 0 and lower < -self.epsilon)\
					or (lower > 0 and lower > self.epsilon):
				if self.networks is not None:
					(input,res) = output_stars[1].construct_last_io()
					outputs=[]
					for net in self.networks:
						outputs.append(net.execute(input.astype(net.layers[0].dtype)))
				raise NotEpsilonEquivalentException(lower,outputs=outputs)
			upper = output_stars[1].minimize_output(i,maximize=True)
			if (upper > 0 and upper > self.epsilon) or (upper < 0 and upper < -self.epsilon):
				if self.networks is not None:
					(input, res) = output_stars[1].construct_last_io()
					outputs = []
					for net in self.networks:
						inVal = input.astype(net.layers[0].dtype)
						outputs.append(net.execute(inVal))
				raise NotEpsilonEquivalentException(upper, outputs=outputs)
			#print(f"[EQUIV] {lower}, {upper}")

class NotEpsilonEquivalentException(Exception):
	def __init__(self, distance,outputs=None):
		self.distance = distance
		self.outputs = outputs

	def __str__(self):
		str = f"Networks are not equivalent: Distance of {self.distance} found!\n"
		if self.outputs is not None:
			str += "Outputs obtained: "
			for out in self.outputs:
				str+=f"{out}\n"
		return str