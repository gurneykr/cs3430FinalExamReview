#!/usr/bin/python

######################################
# module: defintegralapprox.py
# Krista Gurney
# A01671888
######################################

# modify these as you see fit.
import numpy as np
from const import const
from maker import make_plus, make_e_expr, make_prod, make_const, make_pwr, make_pwr_expr
from tof import tof
import matplotlib.pyplot as plt

 
def midpoint_rule(fexpr, a, b, n):
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(n, const)

    area = 0
    fex_tof = tof(fexpr)
    partition = (b.get_val() - a.get_val())/ n.get_val()

    a = int(a.get_val())
    b = int(b.get_val())

    for i in np.arange(a, b, partition):
        mid = i + (partition / 2)
        area += fex_tof(mid) * partition

    return const(area)

def trapezoidal_rule(fexpr, a, b, n):
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(n, const)
    area = 0
    fex_tof = tof(fexpr)
    partition = (b.get_val() - a.get_val())/ n.get_val()

    a = int(a.get_val())
    b = int(b.get_val())

    for i in np.arange(a, b, partition):
        area += partition * ((fex_tof(i)+fex_tof(i+partition))/2)

    return const(area)
def simpson_rule(fexpr, a, b, n):
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(n, const)

    #Simpson = (2M+T)/3
    T = trapezoidal_rule(fexpr, a, b, n)
    M = midpoint_rule(fexpr, a, b, n)

    return const((2*M.get_val() + T.get_val())/3)

  


