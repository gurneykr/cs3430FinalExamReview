#!/usr/bin/python

###################################
# module: linprog.py
# Krista Gurney
# A01671888
###################################

from line_eq import line_eq
from maker import make_line_eq
from maker import make_var, make_const, make_prod
from maker import make_pwr, make_plus
from maker import make_point2d
from const import const
from var import var
from prod import prod
from pwr import pwr
# from poly12 import is_pwr_1
from plus import plus
from tof import tof
from consts import is_const_line
import sys



### sample line equations
lneq1 = make_line_eq(make_var('y'),
                     make_const(2))
lneq2 = make_line_eq(make_var('y'),
                     make_var('x'))
lneq3 = make_line_eq(make_var('y'),
                     make_var('y'))
lneq4 = make_line_eq(make_var('y'),
                     make_prod(make_const(2.0),
                               make_pwr('x', 1.0)))
lneq5 = make_line_eq(make_var('y'),
                     make_prod(make_const(5.0),
                               make_pwr('y', 1.0)))
lneq6 = make_line_eq(make_var('y'),
                     make_plus(make_prod(make_const(5.0),
                                         make_pwr('x', 1.0)),
                               make_const(4.0)))
lneq7 = make_line_eq(make_var('y'),
                     make_plus(make_prod(make_const(5.0),
                                         make_pwr('y', 1.0)),
                               make_const(4.0)))
lneq8 = make_line_eq(make_var('y'),
                     make_plus(make_prod(make_const(3.0),
                                         make_pwr('x', 1.0)),
                               make_const(-4.0)))


def line_intersection(lneq1, lneq2):
    # Case 1: 2 const lines
    if is_const_line(lneq1):
        if is_const_line(lneq2):
            if lneq1.get_lhs().get_name() == 'x':
                x = lneq1.get_rhs().get_val()
                y = lneq2.get_rhs().get_val()
            elif lneq1.get_lhs().get_name() == 'y':
                y = lneq1.get_rhs().get_val()
                x = lneq2.get_rhs().get_val()
            else:
                raise Exception('line_intersection: ' + str(lneq1))
        else:
            y = lneq1.get_rhs().get_val()
            x = tof(lneq2.get_rhs())(y)
    elif is_const_line(lneq2):
        #Case 2: 1 const line y = 1 ;y = x -1
        y = lneq2.get_rhs().get_val()
        x = tof(lneq1.get_rhs())(y)
    elif isinstance(lneq1.get_rhs(), pwr):#y = 1x; y = -1x +6
        eq1_coeff = get_line_coeffs(lneq1)
        eq2_coeff = get_line_coeffs(lneq2)
        if isinstance(lneq2.get_rhs(), plus):
            if isinstance(lneq2.get_rhs().get_elt2(), const):
                eq2_const = lneq2.get_rhs().get_elt2().get_val()
                x = eq2_const/(eq1_coeff - eq2_coeff)
                y = tof(lneq1.get_rhs())(x)
    elif isinstance(lneq1.get_rhs(), plus):#y = -0.2x+10; y =0.2x+5
        eq1_coeff = get_line_coeffs(lneq1)
        eq2_coeff = get_line_coeffs(lneq2)
        if isinstance(lneq2.get_rhs(), plus):
            x = (lneq2.get_rhs().get_elt2().get_val() + lneq1.get_rhs().get_elt2().get_val())/ (eq1_coeff - eq2_coeff)
            y = tof(lneq1.get_rhs())(x)
        else:
            raise Exception("Unknown plus equation")
    elif isinstance(lneq1.get_rhs(), prod):#y = 0.5x; y = -0.75x +3
        eq1_coeff = get_line_coeffs(lneq1)
        eq2_coeff = get_line_coeffs(lneq2)
        if isinstance(lneq2.get_rhs(), plus):
            eq2_const = lneq2.get_rhs().get_elt2().get_val()
            x = eq2_const/(eq1_coeff - eq2_coeff)
            y = tof(lneq1.get_rhs())(x)
        elif isinstance(lneq2.get_rhs(), pwr):#y = -x, y = x
            x = 0.0
            y = 0.0
        else:
            raise Exception("Unknown prod equation")

    else:
        raise Exception('line_intersection: ' + 'unknown equations')

    return make_point2d(x, y)

def get_line_coeffs(lneq):
    if isinstance(lneq.get_rhs(), prod):
        if isinstance(lneq.get_rhs().get_mult1(), const):
            return lneq.get_rhs().get_mult1().get_val()
        else:
            raise Exception("Unknown product")
    elif isinstance(lneq.get_rhs(), pwr):
        return 1.0
    elif isinstance(lneq.get_rhs(), plus):
        if isinstance(lneq.get_rhs().get_elt1(), prod):
            if isinstance(lneq.get_rhs().get_elt1().get_mult1(), const):
                return lneq.get_rhs().get_elt1().get_mult1().get_val()
            else:
                raise Exception('Unknown mult1')
        else:
            raise Exception('Unknown prod')
    else:
        raise Exception('Unknown line equation')



def maximize_obj_fun(f, corner_points):
    currentMax = 0
    for points in corner_points:
        max = f(points.get_x().get_val(), points.get_y().get_val())
        if max > currentMax:
            currentMax = max
            max_point = make_point2d(points.get_x().get_val(), points.get_y().get_val())

    return max_point

def minimize_obj_fun(f, corner_points):
    currentMin = 1000000
    for points in corner_points:
        min = f(points.get_x().get_val(), points.get_y().get_val())
        if min < currentMin:
            currentMin = min
            min_point = make_point2d(points.get_x().get_val(), points.get_y().get_val())

    return min_point


## write your answer to problem 1a as x, y, mv
def opt_prob_1a():
    f1 = lambda x, y: 2 * x + y
    ln1 = make_line_eq(make_var('x'), make_const(1.0))
    ln2 = make_line_eq(make_var('y'), make_const(1.0))
    ln3 = make_line_eq(make_var('x'), make_const(5.0))
    ln4 = make_line_eq(make_var('y'), make_const(5.0))
    ln5 = make_line_eq(make_var('y'), make_plus(make_prod(make_const(-1.0),
                                                          make_pwr('x', 1.0)),
                                                make_const(6.0)))

    cp_1 = line_intersection(ln1, ln5)
    cp_2 = line_intersection(ln1, ln2)
    cp_3 = line_intersection(ln4, ln5)

    corner_points = [cp_1, cp_2, cp_3]

    max_xy = maximize_obj_fun(f1, corner_points)
    max_val = f1(max_xy.get_x().get_val(), max_xy.get_y().get_val())
    print(max_xy, max_val)

## write your answer to problem 1b as x, y, mv
def opt_prob_1b():
    f1 = lambda x, y: x/2 + y
    ln1 = make_line_eq(make_var('y'), make_const(2.0))
    ln2 = make_line_eq(make_var('x'), make_const(0.0))
    ln3 = make_line_eq(make_var('y'), make_pwr('x', 1.0))
    ln4 = make_line_eq(make_var('y'), make_plus(make_prod(make_const(-1.0),
                                                          make_pwr('x', 1.0)),
                                                make_const(6.0)))

    cp_1 = line_intersection(ln3, ln4)
    cp_2 = line_intersection(ln1, ln3)
    cp_3 = line_intersection(ln1, ln4)

    corner_points = [cp_1, cp_2, cp_3]

    min_xy = minimize_obj_fun(f1, corner_points)
    min_val = f1(min_xy.get_x().get_val(), min_xy.get_y().get_val())
    print(min_xy, min_val)


## write your answer to problem 1c as x, y, mv
def opt_prob_1c():
    f1 = lambda x, y: 3 * x - 2*y
    ln1 = make_line_eq(make_var('y'), make_prod(make_const(-1.0), make_pwr('x', 1.0)))
    ln2 = make_line_eq(make_var('y'), make_pwr('x', 1.0))
    ln3 = make_line_eq(make_var('y'), make_plus(make_prod(make_const(1.0 / 2.0),
                                                          make_pwr('x', 1.0)),
                                                make_const(5.0 / 4.0)))

    cp_1 = line_intersection(ln1, ln2)#y = -x; y = x
    cp_2 = line_intersection(ln1, ln3)
    cp_3 = line_intersection(ln2, ln3)

    corner_points = [cp_1, cp_2, cp_3]

    max_xy = maximize_obj_fun(f1, corner_points)
    max_val = f1(max_xy.get_x().get_val(), max_xy.get_y().get_val())
    print(max_xy, max_val)



  
  


