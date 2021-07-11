import math

import numpy as np
import swiglpk as glpk
import gurobipy as grb
from gurobipy.gurobipy import LinExpr
from gurobipy import GRB

from nnenum.lpinstance import LpInstance, UnsatError, SwigArray
from nnenum.gurobiInstance import LpInstance as GurobiIntance
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from .property import EquivalenceProperty
from ..zono_state import ZonoState

StoreI = 0

class MilpTop1Equivalence(EquivalenceProperty):

	EPSILON = 1e-10

	STOP = 1e-15

	def get_gurobi_instance(self, zono: ZonoState):
		Timers.tic('copy_lp')
		milp = GurobiIntance(other_lpi=zono.lpi)
		#var_bounds = zono.lpi._get_col_bounds()
		#for i, (l, u) in enumerate(var_bounds):
		#	milp.add_double_bounded_cols([f"i{i}"], l, u)

		#lp_rows = zono.lpi.get_num_rows()
		#lp_cols = zono.lpi.get_num_cols()

		#inds_row = SwigArray.get_int_array(lp_cols + 1)
		#vals_row = SwigArray.get_double_array(lp_cols + 1)

		#for row in range(1, lp_rows + 1):
		#	data = np.zeros((lp_cols,))
		#	got_len = glpk.glp_get_mat_row(zono.lpi.lp, row, inds_row, vals_row)
		#	for i in range(1, got_len + 1):
		#		data[inds_row[i] - 1] = vals_row[i]
		#	limit = glpk.glp_get_row_ub(zono.lpi.lp, row)
		#	milp.add_dense_row(data, limit, normalize=False)
		invars = milp.lp.getVars()
		Timers.toc('copy_lp')
		Timers.tic('define_out')
		output_zonos = zono.get_output_zonos()
		out1 = milp.lp.addVars(output_zonos[0].mat_t.shape[0],lb=-float('inf'))
		out2 = milp.lp.addVars(output_zonos[1].mat_t.shape[0],lb=-float('inf'))
		milp.lp.update()
		for i,outk in enumerate(out1.keys()):
			# Add output dimensions
			milp.lp.addLConstr(LinExpr(output_zonos[0].mat_t[i], invars)-out1[outk],
			                   GRB.EQUAL,
			                   0.0)
			milp.lp.addLConstr(LinExpr(output_zonos[1].mat_t[i], invars),
			                   GRB.EQUAL,
			                   out2[i])
		#milp.lp.printAttr(['lb', 'ub'])
		Timers.toc('define_out')
		return milp, invars, out1, out2

	def check(self, zono: ZonoState, use_exact=False):
		global StoreI
		Timers.tic('check_top1_fallback')
		Timers.tic('create_milp')
		output_zonos = zono.get_output_zonos()
		milp, invars, outvars1, outvars2 = self.get_gurobi_instance(zono)
		Timers.toc('create_milp')
		Timers.tic('constrain_milp')
		maxvars1 = milp.lp.addVars(output_zonos[0].mat_t.shape[0],vtype=GRB.BINARY)
		net1top1 = milp.lp.addVar(name="net1top1",lb=-float('inf'))
		maxvars2 = milp.lp.addVars(output_zonos[1].mat_t.shape[0], vtype=GRB.BINARY)
		net2top1 = milp.lp.addVar(name="net2top1",lb=-float('inf'))
		#net2topreference = milp.lp.addVar(name="net2topref",lb=-float('inf'))

		milp.lp.update()
		for i, var_index in enumerate(maxvars1.keys()):
			#Net 1
			constraint = outvars1[var_index] - net1top1
			bool_var =maxvars1[var_index]
			milp.lp.addGenConstrIndicator(bool_var, True,
			                              constraint,
			                              GRB.EQUAL,
			                              0.0)

			milp.lp.addLConstr(outvars1[var_index] - net1top1, '<=', 0.0)

			constraint = outvars1[var_index] - net1top1 + MilpTop1Equivalence.EPSILON
			milp.lp.addGenConstrIndicator(bool_var, False,
			                              constraint,
			                              GRB.LESS_EQUAL,
			                              0.0)

			#Net 2
			# constraint = outvars2[var_index] - net2topreference
			# milp.lp.addGenConstrIndicator(bool_var, True,
			#                               constraint,
			#                               GRB.EQUAL,
			#                               0.0)

			constraint = outvars2[var_index] - net2top1
			bool_var = maxvars2[var_index]
			milp.lp.addGenConstrIndicator(bool_var, True,
			                              constraint,
			                              GRB.EQUAL,
			                              0.0)
			milp.lp.addLConstr(outvars2[var_index] - net2top1, '<=', 0.0)
			constraint = outvars2[var_index] - net2top1 + MilpTop1Equivalence.EPSILON
			milp.lp.addGenConstrIndicator(bool_var, False,
			                              constraint,
			                              GRB.LESS_EQUAL,
			                              0.0)

			milp.lp.addGenConstrIndicator(maxvars1[var_index],
			                              True,
			                              maxvars2[var_index],
			                              GRB.EQUAL,
			                              0)


		maxvars1_list = [maxvars1[k] for k in maxvars1.keys()]
		milp.lp.addLConstr(LinExpr(np.ones(len(maxvars1_list)),maxvars1_list), GRB.EQUAL, 1.0)
		maxvars2_list = [maxvars2[k] for k in maxvars2.keys()]
		milp.lp.addLConstr(LinExpr(np.ones(len(maxvars2_list)), maxvars2_list), GRB.EQUAL, 1.0)
		Timers.toc('constrain_milp')
		#milp.lp.setObjective(net2top1-net2topreference, GRB.MAXIMIZE)
		milp.lp.setObjective(0)
		milp.lp.update()
		Timers.tic('check_feasible')
		#print("TUNING")
		if Settings.DO_TUNING:
			milp.lp.tune()
		milp.lp.setParam('GomoryPasses', math.floor(1.5 * output_zonos[0].mat_t.shape[0]))
		milp.lp.optimize()
		Timers.toc('check_feasible')
		if milp.lp.getAttr('Status')==GRB.INFEASIBLE:
			# Infeasible => No Counterexample => Top1 Equiv
			if Settings.DO_TUNING:
				for i in range(min(milp.lp.tuneResultCount, 1)):
					milp.lp.getTuneResult(i)
					milp.lp.write('tune-inf' + str(StoreI) + '.prm')
					StoreI += 1
			Timers.toc('check_top1_fallback')
			return True, (None, None)
		else:
			# Feasible => Found Counterexample
			input_vals = np.zeros((output_zonos[0].mat_t.shape[1],))
			for i, var in enumerate(invars):
				#print(var.X)
				input_vals[i]=var.X
			if Settings.DO_TUNING:
				for i in range(min(milp.lp.tuneResultCount, 1)):
					milp.lp.getTuneResult(i)
					milp.lp.write('tune-feas' + str(StoreI) + '.prm')
					StoreI += 1
			Timers.toc('check_top1_fallback')
			return False, (None, input_vals)
		print(milp.lp.getAttr('ObjBound'))
		# print("Top1")
		# for k in maxvars1.keys():
		# 	print(maxvars1[k].X)
		# print("Top2")
		# for k in maxvars2.keys():
		# 	print(maxvars2[k].X)

	def fallback_check(self, zono):
		raise NotImplementedError()

	def allows_fallback(self, state):
		return False

	def check_out(self, r1, r2):
		return np.argmax(r1) == np.argmax(r2)
