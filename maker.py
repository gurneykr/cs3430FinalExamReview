#!/usr/bin/python

from var import var
from pwr import pwr
from const import const
from plus import plus
from prod import prod
from quot import quot
from point2d import point2d
from ln import ln
from absv import absv
from line_eq import line_eq
import math

def make_var(var_name):
    return var(name=var_name)

def make_pwr(var_name, d):
    return pwr(base=var(name=var_name), deg=const(val=d))

def make_pwr_expr(expr, deg):
    if isinstance(deg, const):
        return pwr(base=expr, deg=deg)
    else:
        return pwr(base=expr, deg=const(val=deg))

def make_const(val):
    return const(val=val)

def make_point2d(xv, yv):
    return point2d(x=make_const(xv),
                   y=make_const(yv))

def make_e_expr(d):
    if isinstance(d, float):
        return pwr(base=make_const(math.e), deg=const(val=d))
    elif isinstance(d, const):
        return pwr(base=make_const(math.e), deg=d)
    elif isinstance(d, pwr) or isinstance(d, plus) or \
         isinstance(d, prod) or isinstance(d, quot):
        return pwr(base=make_const(math.e), deg=d)
    else:
        raise Exception('make_e_expr: case 1: ' + str(d))

def make_ln(expr):
    return ln(expr=expr)

def make_quot(nexpr, dexpr):
    return quot(num=nexpr, denom=dexpr)

def make_prod(mult_expr1, mult_expr2):
    return prod(mult1=mult_expr1, mult2=mult_expr2)

def make_plus(elt_expr1, elt_expr2):
    return plus(elt1=elt_expr1, elt2=elt_expr2)

def make_absv(expr):
    return absv(expr)

def make_line_eq(lhs, rhs):
    return line_eq(lhs=lhs, rhs=rhs)



