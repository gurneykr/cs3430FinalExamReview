#!/usr/bin/python


###########################################
# module: tof.py
# Krista Gurney
# A01671888
###########################################

from var import var
from const import const
from pwr import pwr
from prod import prod
from plus import plus
from quot import quot
from ln import ln
from absv import absv
import math


def tof(expr):
    if isinstance(expr, const):
        return const_tof(expr)
    elif isinstance(expr, pwr):
        return pwr_tof(expr)
    elif isinstance(expr, prod):
        return prod_tof(expr)
    elif isinstance(expr, plus):
        return plus_tof(expr)
    elif isinstance(expr, quot):
        return quot_tof(expr)
    elif isinstance(expr, ln):
        return ln_tof(expr)
    elif isinstance(expr, absv):
        return absv_tof(expr)
    else:
        raise Exception('tof: ' + str(expr))

## here is how you can implement converting
## a constant to a function.
def const_tof(c):
    assert isinstance(c, const)
    def f(x):
        return c.get_val()
    return f

def pwr_tof(expr):
    assert isinstance(expr, pwr)
    f1 = expr.get_base()
    f2 = expr.get_deg()

    if isinstance(f1, var) and not isinstance(f2, var):
        def f(x):
            temp = tof(f2)(x)
            # if type(x) == const:
            #     print("found it")
            return math.pow(x, tof(f2)(x))
        return f
    elif isinstance(f1, var) and isinstance(f2, var):
        def f(x):
            return math.pow(x, x)
        return f
    elif not isinstance(f1, var) and isinstance(f2, var):
        def f(x):
            return math.pow(tof(f1)(x), x)
        return f
    else:
        def f(x):
            return math.pow(tof(f1)(x), tof(f2)(x))
        return f

def prod_tof(expr):
    assert isinstance(expr, prod)
    f1 = expr.get_mult1()
    f2 = expr.get_mult2()

    if isinstance(f1, var) and not isinstance(f2, var):
        def f(x):
            return x * tof(f2)(x)
        return f
    elif isinstance(f1, var) and isinstance(f2, var):
        def f(x):
            return x * x
        return f
    elif not isinstance(f1, var) and isinstance(f2, var):
        def f(x):
            return tof(f1)(x) * x
        return f
    else:
        def f(x):
            return tof(f1)(x) * tof(f2)(x)
        return f

def plus_tof(expr):
    assert isinstance(expr, plus)
    f1 = expr.get_elt1()
    f2 = expr.get_elt2()

    if isinstance(f1, var) and not isinstance(f2, var):
        def f(x):
            return x + tof(f2)(x)
        return f
    elif isinstance(f1, var) and isinstance(f2, var):
        def f(x):
            return x + x
        return f
    elif not isinstance(f1, var) and isinstance(f2, var):
        def f(x):
            return tof(f1)(x) + x
        return f
    else:
        def f(x):
            return tof(f1)(x) + tof(f2)(x)
        return f

def quot_tof(expr):
    assert isinstance(expr, quot)
    f1 = expr.get_num()
    f2 = expr.get_denom()

    if isinstance(f1, var) and not isinstance(f2, var):
        def f(x):
            return x / tof(f2)(x)
        return f
    elif isinstance(f1, var) and isinstance(f2, var):
        def f(x):
            return x / x
        return f
    elif not isinstance(f1, var) and isinstance(f2, var):
        def f(x):
            return tof(f1)(x) / x
        return f
    else:
        def f(x):
            return tof(f1)(x) / tof(f2)(x)
        return f

def ln_tof(expr):
    assert isinstance(expr, ln)
    g = expr.get_expr()
    def f(x):
        return math.log(tof(g)(x))
    return f

def absv_tof(expr):
    assert isinstance(expr, absv)
    g = expr.get_expr()
    def f(x):
        return abs(tof(g)(x))
    return f