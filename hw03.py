#!/usr/bin/python

from prod import prod
from const import const
from plus import plus
from pwr import pwr
from quot import quot
from tof import tof
from deriv import deriv
from derivtest import loc_xtrm_1st_drv_test
from maker import make_prod, make_const, make_pwr, make_plus
import math

#######################################
# module: hw03.py
# Krista Gurney
# YOUR A01671888
#######################################

# place all necessary imports here.
#
# I placed the updated version of maker.py
# Use it as you see fit.

def dydt_given_x_dxdt(yt, x, dxdt):
    yt_deriv = deriv(yt)
    yt_fn = tof(yt_deriv)(x.get_val())
    result = yt_fn * dxdt.get_val()
    return const(result)

def oil_disk_test():
    yt = make_prod(make_const(0.02 * math.pi),
                    make_pwr('r', 2.0))
    print(yt)
    dydt = dydt_given_x_dxdt(yt, make_const(150.0), make_const(20.0))
    assert not dydt is None
    assert isinstance(dydt, const)
    print(dydt)

def arm_tumor_test():
    yt = make_prod(make_const(0.003 * math.pi),
                   make_pwr('r', 3.0))
    print(yt)
    dydt = dydt_given_x_dxdt(yt, make_const(10.3), make_const(-1.75))
    assert not dydt is None
    assert isinstance(dydt, const)
    print(dydt)


def maximize_revenue(demand_expr, constraint):
    priceExpr = demand_expr
    priceExprFn = tof(priceExpr)
    revenueExpr = mult_x(priceExpr)

    extrema = loc_xtrm_1st_drv_test(revenueExpr)

    for i, j in extrema:
        if i == "max":
            x = j.get_x().get_val()
            if constraint(j.get_x().get_val()):
                price = priceExprFn(j.get_x().get_val())
                max_revenue = price * j.get_x().get_val()

    return [(const(x)), (const(max_revenue)), (const(price))]


def mult_x(expr):  #1/12x^2 - 10x + 300

    if isinstance(expr, plus):
        if isinstance(expr.get_elt2(), const):
            return plus(mult_x(expr.get_elt1()), prod(expr.get_elt2(), make_pwr('x', 1.0)))
        else:
            return plus(mult_x(expr.get_elt1()), mult_x(expr.get_elt2()))
    elif isinstance(expr, pwr):
        if isinstance(expr.get_deg(), const):
            return pwr(expr.get_base(), const(expr.get_deg().get_val()+1))
        else:
            return pwr(mult_x(expr.get_base()), mult_x(expr.get_deg()))
    elif isinstance(expr, prod):
        return prod(mult_x(expr.get_mult1()), mult_x(expr.get_mult2()))
    elif isinstance(expr, quot):
        return quot(mult_x(expr.get_num()), mult_x(expr.get_denom()))
    else:
        return expr


def max_rev_test():
    print("***Max Revenue Test 01 *****")
    e1 = make_prod( make_const(1.0/12.0), make_pwr('x', 2.0))
    e2 = make_prod(make_const(-10.0), make_pwr('x', 1.0))
    sum1 = make_plus(e1, e2)
    demand_expr = make_plus(sum1, make_const(300.0))
    num_units, rev, price = maximize_revenue(demand_expr, constraint=lambda x: 0 <= x <= 60)
    print('x = ', num_units.get_val())
    print("rev= ", rev.get_val())
    print('price = ', price.get_val())
    print("Max Revenue Test: pass")


if __name__ == "__main__":
    oil_disk_test()
