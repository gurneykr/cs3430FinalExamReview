#!/usr/bin/python

#########################################
# module: derivtest.py
#Krista Gurney
# A01671888
#########################################

from const import const
from pwr import pwr
from prod import prod
from plus import plus
from tof import tof
from deriv import deriv
from maker import make_const, make_prod, make_pwr, make_plus, make_point2d
from poly12 import find_poly_2_zeros, find_poly_1_zeros

def findDegree(expr):
    return evalExpr(expr)

def evalExpr(expr):
    if isinstance(expr, plus):
        results = evalExpr(expr.get_elt1())
        if results > -1:
            return results

        results = evalExpr(expr.get_elt2())
        if results >= 0:
            return results
    elif isinstance(expr, pwr):
        theFn = tof(expr.get_deg())
        return theFn(0)
    elif isinstance(expr, prod):
        results = evalExpr(expr.get_mult1())
        if results > -1:
            return results
        results = evalExpr(expr.get_mult2())
        if results > -1:
            return results
    elif isinstance(expr, const):
        return -1
    else:
        raise Exception('Exception Found: ', type(expr))

    return -2

def loc_xtrm_1st_drv_test(expr):
    exprFn = tof(expr)
    derivativeExpr = deriv(expr)
    derivfn = tof(derivativeExpr)

    degree = findDegree(derivativeExpr)

    critical_points = []

    if degree == 2:
        xvalues = find_poly_2_zeros(derivativeExpr)
        for x in xvalues:
            y = exprFn(x.get_val())

            critical_points.append(make_point2d(x.get_val(), y))
    elif degree == 1:
        x = find_poly_1_zeros(derivativeExpr)
        y = exprFn(x.get_val())

        critical_points.append(make_point2d(x.get_val(), y))
    elif degree == 0:
        # The derivative is just a constant so all values will be just the constant
        # f` = 5
        x = derivativeExpr.get_val();
    else:
        raise Exception("Not a first or second degree polynomial degree=", degree)

    maxima = None
    minima = None
    for p in critical_points:
        x = p.get_x().get_val()
        less = derivfn(x - 0.5)
        more = derivfn(x + 0.5)
        if less < 0 and more > 0:
            y = exprFn(x)
            minima = make_point2d(x, y)
        elif less > 0 and more < 0:
            y = exprFn(x)
            maxima = make_point2d(x, y)

    if not maxima and not minima:
        return None
    else:
        results = []
        if maxima:
            results.append(("max", maxima))
        if minima:
            results.append(("min", minima))

    return results

def loc_xtrm_2nd_drv_test(expr):
    # Get the extrema from the first derivative
    first_xtrema = loc_xtrm_1st_drv_test(expr)

    # Take the second derivative, put those extrema values in
    expr2 = deriv(deriv(expr))
    fn2 = tof(expr2)

    results = []
    for ex in first_xtrema:
        value = fn2(ex[1].get_x().get_val())

        if value < 0:  # If the results are negative, then we have a local max
            point = make_point2d(ex[1].get_x().get_val(), ex[1].get_y().get_val())
            results.append(("max", point))

        else:  # If the results are positive, then we have a local min
            point = make_point2d(ex[1].get_x().get_val(), ex[1].get_y().get_val())
            results.append(("min", point))

    return results


def find_infl_pnts(expr):
    #find the second derivative
    second_drv = deriv(deriv(expr))
    degree = findDegree(second_drv)
    expr_tof = tof(expr)
    inflection_points = []
    if degree == 2:
        zeros = find_poly_2_zeros(second_drv)
        for x in zeros:
            y = expr_tof(x.get_val())
            inflection_points.append(make_point2d(x.get_val() , y))
    else:
        x = find_poly_1_zeros(second_drv)
        y = expr_tof(x.get_val())
        inflection_points.append(make_point2d(x.get_val(), y))

    return inflection_points



# def test_04():
#     f1 = make_prod(make_const(27.0), make_pwr('x', 3.0))
#     f2 = make_prod(make_const(-27.0), make_pwr('x', 2.0))
#     f3 = make_prod(make_const(9.0), make_pwr('x', 1.0))
#     f4 = make_plus(f1, f2)
#     f5 = make_plus(f4, f3)
#     f6 = make_plus(f5, make_const(-1.0))
#     drv = deriv(f6)
#     assert not drv is None
#     xtrma = loc_xtrm_2nd_drv_test(f6)
#     assert xtrma is None
#
# def test_01():
#     # -2x^2 + 3x + 1
#     f2 = make_prod(make_const(-2.0), make_pwr('x', 2.0))
#     f3 = make_prod(make_const(3.0), make_pwr('x', 1.0))
#     f4 = make_plus(f2, f3)
#     poly = make_plus(f4, make_const(1.0))
#     loc_xtrm_1st_drv_test(poly)
#
# if __name__ == '__main__':
#     test_04()

