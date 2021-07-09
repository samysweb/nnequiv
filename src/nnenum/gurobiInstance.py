
import gurobipy as grb
import numpy as np

from nnenum.settings import Settings


class LpInstance:
    lp_time_limit_sec = 15

    def __init__(self, other_lpi=None):
        if other_lpi is None:
            self.lp = grb.Model('model')
        else:
            self.lp = other_lpi.lp.copy()

        self.lp.setParam('TimeLimit', Settings.GLPK_TIMEOUT)

    def __del__(self):
        if hasattr(self, 'lp') and self.lp is not None:
            if not isinstance(self.lp, tuple):
                del self.lp
            self.lp = None

    def serialize(self, check_constraints_sense=False):
        model = self.lp
        sense = model.getAttr('ModelSense') # 1: minimize, -1: maximize
        numVars = model.getAttr('NumVars')

        if check_constraints_sense:
            for c in model.getConstrs():
                if not c.sense == grb.GRB.LESS_EQUAL:
                    raise ValueError(f'Constraint sense {c.sense} not implemented!' +
                                     ' Method assumes only <=')

        A = model.getA()
        b = np.array(model.getAttr('RHS'))

        lbs = []
        ubs = []
        objs = []
        types = []
        for v in model.getVars():
            lbs.append(v.lb)
            ubs.append(v.ub)
            objs.append(v.obj)
            types.append(v.vtype)

        self.lp = (A, b, lbs, ubs, objs, types, sense, numVars)

    def deserialize(self):
        assert isinstance(self.lp, tuple)

        A, b, lbs, ubs, objs, types, sense, numVars, modelName = self.lp
        model = grb.Model(modelName)

        x = model.addMVar(shape=numVars, lb=lbs, ub=ubs, obj=objs, vtype=types, name='x')
        model.addConstr(A @ x <= b, name='c')  # TODO: what about >= or == constraints?
        model.setAttr('ModelSense', sense)
        model.update()

        self.lp = model

    def set_col_bounds(self, col, lb, ub):
        self.lp.getVars()[col].lb = lb
        self.lp.getVars()[col].ub = ub


    def get_num_rows(self):
        return self.lp.getAttr('NumConstrs')

    def get_num_cols(self):
        return self.lp.getAttr('NumVars')

    def add_rows_less_equal(self, rhs_vec):
        raise NotImplementedError('What is this supposed to do?'  +
                                  ' Add constraints without variables' +
                                  ' and just rhs makes no sense???')

    def get_types(self):
        '''get constraint types. (==, <=, >=)'''
        types = []
        for c in self.lp.getConstrs():
            types.append(c.sense)

        return types

    def add_positive_cols(self, names):
        raise NotImplementedError()

    def add_cols(self, names):
        raise NotImplementedError()

    def add_double_bounded_cols(self, names, lb, ub):
        '''add variables to the model with common lower and upper bound'''
        model = self.lp
        for name in names:
            model.addVar(lb, ub, name=name)

        model.update()

    def compute_residual(self, alpha_row, bounds):
        min_factors = np.where(alpha_row <= 0, bounds[:, 1], bounds[:, 0])
        alpha_min = min_factors.dot(alpha_row)
        return alpha_min

    def add_dense_row(self, vec, rhs, normalize=True):
        '''add constraint row <= rhs, where row is dense array'''

        if normalize:
            norm = np.linalg.norm(vec)
            vec = vec / norm
            rhs = rhs / norm

        # model.addMContr() requires Matrix as input
        lhs = np.expand_dims(vec, axis=0)
        rhs = np.array([rhs])
        self.lp.addMConstr(lhs, self.lp.getVars(), '<=', rhs)

        # TODO: could be removed, when it is ensured that optimize() is called next
        self.lp.update()

    def set_constraints_csr(self, data, glpk_indices, indptr, shape):
        raise NotImplementedError()

    def get_rhs(self, row_indices=None):
        '''get rhs vector of the constraints
        row_indices - a list of row indices, None=all
        returns an np.array of rhs values for the requested indices
        '''
        rhss = []
        for c in self.lp.getConstrs():
            rhss.append(c.RHS)

        rhss = np.array(rhss)

        if not row_indices is None:
            row_indices = np.array(row_indices)
            rhss = rhss[row_indices]

        return rhss

    def set_rhs(self, rhs_vec):
        raise NotImplementedError()

    def get_constraints_csr(self):
        '''get lp matrix as a scipy.sparse.csr_matrix'''
        return self.lp.getA()

    def is_feasible(self):
        '''check if the lp is feasible

        returns a feasible point or None'''
        # TODO: doesn't this return a boolean???
        return self.minimize(None, fail_on_unsat=False, use_exact=False) is not None

    def contains_point(self, pt, tol=1e-9):
        A = self.get_constraints_csr()
        b = self.get_rhs()
        vec = A @ pt

        # np.all() return True, iff every element is true
        return (vec - tol <= b).all()

    def set_minimize_direction(self, direction):
        direction = np.array(direction)
        self.lp.setMObjective(None, direction, 0, xc=self.lp.getVars())
        self.lp.update()

    def reset_basis(self, basis_type='std'):
        raise NotImplementedError()

    def minimize(self, direction_vec, fail_on_unsat=True, use_exact=False):
        '''minimize the lp, returning a list of assignments to each of the variables.

        if direction_vec is not None, will first assign optimization direction

        !!!fail_on_unsat and use_exact are not supported right now!!!

        returns None if UNSAT, otherwise the optimization result'''

        assert not isinstance(self.lp, tuple), 'self.lp was a tuple. Did you call lpi.deserialize()?'

        if direction_vec is None:
            direction_vec = np.zeros(self.lp.getAttr('NumVars'))

        self.set_minimize_direction(direction_vec)

        # if Settings.GLPK_RESET_BEFORE_MINIMIZE:
        #    self.reset_basis()

        if use_exact:
            raise NotImplementedError('Gurobi has no exact simplex method!')
        else:
            self.lp.optimize()
            opt_status = self.lp.getAttr('Status')

            if fail_on_unsat:
                assert opt_status == grb.GRB.OPTIMAL, f'[Gurobi] Optimization failed with status code {opt_status}!'

            if opt_status == grb.GRB.OPTIMAL:
                sol = np.array(self.lp.getAttr('X'))
            else:
                sol = None

        return sol
