from line_eq import line_eq
from maker import make_line_eq
from maker import make_var, make_const, make_prod
from maker import make_pwr, make_plus
from linprog import line_intersection, get_line_coeffs, maximize_obj_fun, minimize_obj_fun, opt_prob_1a, opt_prob_1b,opt_prob_1c
from maker import make_point2d, make_e_expr
from const import const
from var import var
from prod import prod
from pwr import pwr
from plus import plus
from tof import tof
from deriv import deriv
from consts import is_const_line
import sys
from finalexamquestions import demand_elasticity, net_change, consumer_surplus, fit_regression_line, taylor_poly, newton_raphson,pell_approx_sqrt, pell, linear_programming_prob1, linear_programming_prob2
from finalexamquestions import problem_2_deriv, problem_1_deriv, arm_tumor_test, problem_4_rate_change, problem_02, problem_1_decay, problem_2_arbitrary_sol, problem_3_unique_sol
from derivtest import find_infl_pnts
import numpy as np
import math


def test_01():
    print(demand_elasticity(const(10.0)))
    print(demand_elasticity(const(20.0)))
    print(demand_elasticity(const(30.0)))

def test_02():
    fex = make_plus(make_plus(make_prod(const(0.03),make_pwr('x', 2.0)), make_prod(const(-2.0), make_pwr('x', 1.0))),const(25))
    print(fex)
    print(net_change(fex, const(20), const(25)))

def test_03():
    dexpr = make_plus(make_const(50), make_prod(make_const(-0.06), make_pwr('x', 2.0)))
    print(dexpr)
    print(consumer_surplus(dexpr, const(20)))

def test_04():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    rlf = fit_regression_line(x, y)


    assert rlf(1) == 2.0
    assert rlf(2) == 3.0
    assert rlf(3) == 4.0
    assert rlf(4) == 5.0
    assert rlf(10) == 11.0
    assert rlf(101) == 102.0
    print('Problem 04: Unit Test 01: pass')

def gt21_02(x):
    ''' ground truth for 2nd taylor of fexpr2_01. '''
    fexpr2_01 = make_prod(make_pwr('x', 1.0),
                          make_e_expr(make_pwr('x', 1.0)))
    f0 = tof(fexpr2_01)
    f1 = tof(deriv(fexpr2_01))
    f2 = tof(deriv(deriv(fexpr2_01)))
    return f0(2.0) + f1(2.0)*(x - 2.0) + (f2(2.0)/2)*(x - 2.0)**2

def test_05():
    fexpr2_01 = make_prod(make_pwr('x', 1.0),
                          make_e_expr(make_pwr('x', 1.0)))
    print(gt21_02(2.001))
    fex = taylor_poly(fexpr2_01, make_const(2.001), make_const(2))
    print(fex)
    fex_tof = tof(fex)
    print(fex_tof(2.001))

def test_06():
    ''' Approximating x^3 + x - 1. '''
    fexpr = make_pwr('x', 3.0)
    fexpr = make_plus(fexpr, make_pwr('x', 1.0))
    fexpr = make_plus(fexpr, make_const(-1.0))
    print(newton_raphson(fexpr, make_const(1.0), make_const(10000)))

def test_07():
    fexpr = make_plus(make_pwr('x', 2.0), make_const(-3.0))
    print(newton_raphson(fexpr, make_const(1.0), make_const(10000)))

def test_08():
    print(pell(10))

def test_09():
    print(linear_programming_prob2())

def test_10():
    problem_1_deriv()

def test_11():
    f1 = make_prod(make_const(-1.0), make_pwr('x', 3.0))
    f2 = make_prod(make_const(8.5), make_pwr('x', 2.0))
    f3 = make_prod(make_const(0.0), make_pwr('x', 0.0))
    f4 = make_plus(f1, f2)
    f5 = make_plus(f4, f3)
    f6 = make_plus(f5, make_const(100.0))
    print(f6)

    ips = find_infl_pnts(f6)
    err = 0.0001
    assert len(ips) == 1
    ip = ips[0]
    # assert abs(ip.get_x().get_val() - 1.0) <= err
    # assert abs(ip.get_y().get_val() - 3.0) <= err
    print("inflection points: ", ip)

def test_12():
    yt = make_prod(make_const(0.003 * math.pi),
                   make_pwr('r', 3.0))
    arm_tumor_test(yt, make_const(10.3), make_const(-1.75))

if __name__ =='__main__':
    # problem_4_rate_change()
    # problem_02()
    # problem_1_decay()
    # problem_2_arbitrary_sol(0.3)
    problem_3_unique_sol(0.3, 6.0)