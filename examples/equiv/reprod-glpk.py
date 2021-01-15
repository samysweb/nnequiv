import numpy as np
from swiglpk import *

from nnenum.settings import Settings


def set_minimize_direction(lp, direction):
    '''set the optimization direction'''
    print("Objective Function: ",end="")
    for i, d in enumerate(direction):
        col = int(1 + i)
        print(f"{i}: {d}; ",end="")
        glp_set_obj_coef(lp, col, float(d))
    print("")

def printRes(lp):
    Z = glp_get_obj_val(lp)
    x1 = glp_get_col_prim(lp, 1)
    x2 = glp_get_col_prim(lp, 2)
    x3 = glp_get_col_prim(lp, 3)
    x4 = glp_get_col_prim(lp, 4)
    x5 = glp_get_col_prim(lp, 5)
    print("Z = %g; x1 = %g; x2 = %g; x3 = %g; x4 = %g, x5 = %g\n" % (Z, x1, x2, x3, x4, x5))
    return (x1,x2,x3,x4,x5)


def get_lp_params(alternate_lp_params=False):
    'get the lp params object'
    return None

    # if not hasattr(get_lp_params, 'obj'):
    #     params = glp_smcp()
    #     glp_init_smcp(params)
    #
    #     # params.msg_lev = glpk.GLP_MSG_ERR
    #     params.msg_lev = GLP_MSG_ERR
    #     params.meth = GLP_PRIMAL if Settings.GLPK_FIRST_PRIMAL else glpk.GLP_DUAL
    #
    #     params.tm_lim = int(Settings.GLPK_TIMEOUT * 1000)
    #     params.out_dly = 2 * 1000  # start printing to terminal delay
    #     params.msg_lev = GLP_MSG_ALL
    #
    #     get_lp_params.obj = params
    #
    #     # make alternative params
    #     params2 = glp_smcp()
    #     glp_init_smcp(params2)
    #     params2.meth = GLP_DUAL if Settings.GLPK_FIRST_PRIMAL else glpk.GLP_PRIMAL
    #     params2.msg_lev = GLP_MSG_ON
    #
    #     params2.tm_lim = int(Settings.GLPK_TIMEOUT * 1000)
    #     params2.out_dly = 1 * 1000  # start printing to terminal status after 1 secs
    #
    #     get_lp_params.alt_obj = params2

    # if alternate_lp_params:
    #     # glpk.glp_term_out(glpk.GLP_ON)
    #     rv = get_lp_params.alt_obj
    # else:
    #     # glpk.glp_term_out(glpk.GLP_OFF)
    #     rv = get_lp_params.obj
    #
    # return rv


def check_result(x1,x2,x3,x4,x5, rows, biases):
    rowNum = 1
    input = [x5,x4,x3,x2,x1]
    for row, bias in zip(rows, biases):
        sum = 0.0
        colNum = 0
        for val in row:
            sum+= val * input[colNum]
            colNum+=1
        if sum > bias:
            print(f"Found error! Diff: {sum-bias}")
        rowNum += 1

lp = glp_create_prob()
glp_set_prob_name(lp, "reprod")
glp_set_obj_dir(lp, GLP_MIN)

# Instance 349
rows = [[0.04864749684929848, -0.01757809706032276, -0.17345963418483734, -0.9826385378837585, -0.0407162681221962],
     [0.4536316990852356, -0.1982513815164566, -0.5648610591888428, 0.6599723696708679, -0.016828561201691628], # 2
     [-0.05371265485882759, -0.09199603646993637, 0.06838224083185196, 0.991743266582489, -0.020513953641057014],
     [0.8469746112823486, -0.3263065218925476, 0.32752013206481934, 0.06199650466442108, -0.25503942370414734], # 4
     [-0.06426270306110382, -0.054490767419338226, 0.982328474521637, 0.15956541895866394, 0.04970501735806465], # 5
     [-0.27481338381767273, 0.23421965539455414, 0.30147749185562134, 0.747488260269165, -0.46903231739997864],
     [0.16181759536266327, -0.041498687118291855, 0.056704867631196976, -0.9841556549072266, 0.017749551683664322],
     [0.14756521582603455, -0.22544756531715393, -0.8337231874465942, -0.48134520649909973, 0.024706074967980385],
     [0.13338515162467957, 0.0015116985887289047, 0.6001701951026917, 0.47933512926101685, -0.6262904405593872],
     [-0.5257245898246765, 0.4158400893211365, 0.15986143052577972, 0.691771924495697, 0.2158391922712326],
     [-0.0926082506775856, 0.011573733761906624, -0.07388503104448318, 0.9861070513725281, -0.11586035788059235],
     [0.602049708366394, -0.4677243232727051, -0.2943362295627594, -0.41899874806404114, -0.3956971764564514],
     [-0.06426925212144852, -0.05448075756430626, 0.9823279976844788, 0.1595783829689026, 0.04967666417360306], # Similar to 5
     [0.8470733165740967, -0.32620829343795776, 0.3273952007293701, 0.06201716139912605, -0.25499242544174194], # Similar to 4
     [-0.45361119508743286, 0.1982090324163437, 0.5648705363273621, -0.659990668296814, 0.01684524491429329]] # Opposite of 2
biases = [-0.06928084790706635, 0.009916311129927635, 0.08646398037672043, -0.006437735166400671, 0.06931018829345703, 0.10731283575296402, -0.0713353231549263, -0.09997757524251938, 0.044224679470062256, 0.1301191747188568, 0.07082317024469376, -0.11029086261987686, 0.06929231435060501, -0.006437124218791723, -0.009919529780745506]

def diff(v1, v2):
    diff1 = np.linalg.norm(v1-v2)
    diff2 = np.linalg.norm(v1+v2)
    return min(diff1, diff2)

print(f"Norm Diff: {diff(np.array(rows[1]),np.array(rows[14]))}")
print(f"Distance Diff: {diff(np.array(biases[1]),np.array(biases[14]))}")

# Instance 428
# rows = [[0.04864749684929848, -0.01757809706032276, -0.17345963418483734, -0.9826385378837585, -0.0407162681221962],
#         [0.4536316990852356, -0.1982513815164566, -0.5648610591888428, 0.6599723696708679, -0.016828561201691628],
#         [-0.05371265485882759, -0.09199603646993637, 0.06838224083185196, 0.991743266582489, -0.020513953641057014],
#         [0.8469746112823486, -0.3263065218925476, 0.32752013206481934, 0.06199650466442108, -0.25503942370414734],
#         [-0.06426270306110382, -0.054490767419338226, 0.982328474521637, 0.15956541895866394, 0.04970501735806465],
#         [-0.27481338381767273, 0.23421965539455414, 0.30147749185562134, 0.747488260269165, -0.46903231739997864],
#         [0.16181759536266327, -0.041498687118291855, 0.056704867631196976, -0.9841556549072266, 0.017749551683664322],
#         [0.14756521582603455, -0.22544756531715393, -0.8337231874465942, -0.48134520649909973, 0.024706074967980385],
#         [0.13338515162467957, 0.0015116985887289047, 0.6001701951026917, 0.47933512926101685, -0.6262904405593872],
#         [-0.5257245898246765, 0.4158400893211365, 0.15986143052577972, 0.691771924495697, 0.2158391922712326],
#         [-0.0926082506775856, 0.011573733761906624, -0.07388503104448318, 0.9861070513725281, -0.11586035788059235],
#         [0.602049708366394, -0.4677243232727051, -0.2943362295627594, -0.41899874806404114, -0.3956971764564514],
#         [-0.06426925212144852, -0.05448075756430626, 0.9823279976844788, 0.1595783829689026, 0.04967666417360306],
#         [0.8470733165740967, -0.32620829343795776, 0.3273952007293701, 0.06201716139912605, -0.25499242544174194],
#         [-0.45361119508743286, 0.1982090324163437, 0.5648705363273621, -0.659990668296814, 0.01684524491429329],
#         [-0.17116691172122955, 0.2331039011478424, 0.852286159992218, 0.4352318346500397, -0.02336525358259678],
#         [0.14350329339504242, -0.03413015976548195, 0.07424727827310562, -0.9861066937446594, 0.017968198284506798],
#         [0.15337520837783813, -0.10587888956069946, -0.29048600792884827, -0.9331469535827637, 0.10059934109449387]]
# biases = [-0.06928084790706635, 0.009916311129927635, 0.08646398037672043, -0.006437735166400671, 0.06931018829345703, 0.10731283575296402, -0.0713353231549263, -0.09997757524251938, 0.044224679470062256, 0.1301191747188568, 0.07082317024469376, -0.11029086261987686, 0.06929231435060501, -0.006437124218791723, -0.009919529780745506, 0.0977737084031105, -0.07022736221551895, -0.09404632449150085]

glp_add_cols(lp, 5);
glp_set_col_name(lp, 1, "x1")
glp_set_col_bnds(lp, 1, GLP_DB, 0.0, 0.1)
glp_set_col_name(lp, 2, "x2")
glp_set_col_bnds(lp, 2, GLP_DB, 0.0, 0.1)
glp_set_col_name(lp, 3, "x3")
glp_set_col_bnds(lp, 3, GLP_DB, 0.0, 0.1)
glp_set_col_name(lp, 4, "x4")
glp_set_col_bnds(lp, 4, GLP_DB, 0.0, 0.1)
glp_set_col_name(lp, 5, "x5")
glp_set_col_bnds(lp, 5, GLP_DB, 0.0, 0.1)

assert(len(biases)==len(rows))

# rows = rows[:-1]
# biases = biases[1:]

glp_add_rows(lp, len(rows));
rowNum = 1
index = 1
rowI = intArray(1+len(rows)*len(rows[0]))
colI = intArray(1+len(rows)*len(rows[0]))
valI = doubleArray(1+len(rows)*len(rows[0]))
for row, bias in zip(rows,biases):
    glp_set_row_bnds(lp, rowNum, GLP_UP, 0.0, bias)
    colNum = 0
    for val in row:
        rowI[index]=rowNum
        colI[index]=5-colNum
        valI[index]=val
        colNum += 1
        index+=1
    rowNum += 1
glp_load_matrix(lp, index-1, rowI, colI, valI)

#glp_set_obj_coef(lp, 1, 0.0)
#glp_set_obj_coef(lp, 2, 0.0)
#glp_set_obj_coef(lp, 3, 0.0)
#glp_set_obj_coef(lp, 4, 0.0)
#glp_set_obj_coef(lp, 5, 1.0)

set_minimize_direction(lp,[0.,0.,0.,0.,1.])
res = glp_simplex(lp, get_lp_params())
print("Result: %d"%(res))
x1,x2,x3,x4,x5 = printRes(lp)
check_result(x1,x2,x3,x4,x5, rows, biases)



set_minimize_direction(lp,[1.,0.,0.,0.,0.])
res = glp_simplex(lp, get_lp_params())
print("Result: %d"%(res))
x1,x2,x3,x4,x5 = printRes(lp)
check_result(x1,x2,x3,x4,x5, rows, biases)


set_minimize_direction(lp,[0.,0.,0.,0.,1.])
res = glp_simplex(lp, get_lp_params())
print("Result: %d"%(res))
x1,x2,x3,x4,x5 = printRes(lp)
check_result(x1,x2,x3,x4,x5, rows, biases)






