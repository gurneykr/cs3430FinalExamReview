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
from finalexamquestions import demand_elasticity, net_change, consumer_surplus, fit_regression_line, taylor_poly, newton_raphson,pell_approx_sqrt, pell
import numpy as np


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

if __name__ =='__main__':
    test_08()