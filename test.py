from line_eq import line_eq
from maker import make_line_eq
from maker import make_var, make_const, make_prod
from maker import make_pwr, make_plus
from linprog import line_intersection, get_line_coeffs, maximize_obj_fun, minimize_obj_fun, opt_prob_1a, opt_prob_1b,opt_prob_1c
from maker import make_point2d
from const import const
from var import var
from prod import prod
from pwr import pwr
from plus import plus
from tof import tof
from consts import is_const_line
import sys
from finalexamquestions import demand_elasticity, net_change, consumer_surplus

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


if __name__ =='__main__':
    test_03()