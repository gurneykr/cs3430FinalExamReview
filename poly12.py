#!/usr/bin/python

############################################
# module: poly12.py
# Krista Gurney
# A01671888
############################################

from prod import prod
from const import const
from pwr import pwr
from plus import plus
from var import var
from tof import tof
import math

def evaluate_expression(expr, abc):

    # ((((1/3)*3(x^(3-1)) +-(2)*(x^(2-1)) + (3)*(x^(1-1)) + 0)
    if abc['a'] != None and abc['b'] != None and abc['c'] != None:
        return abc

    if isinstance(expr, plus):
        # a + b^1
        results = evaluate_expression(expr.get_elt1(), abc)
        if results['degrees'] == 2:
            results['a'] = 1
            results['degrees'] = -1
        elif results['degrees'] == 1:
            results['b'] = 1
            results['degrees'] = -1
        elif results['degrees'] == 0:
            results['c'] = 1
            results['degrees'] = -1

        results = evaluate_expression(expr.get_elt2(), abc)  # should be 5
        if results['degrees'] == 2:
            results['a'] = 2
            results['degrees'] = -1
        elif results['degrees'] == 1:
            results['b'] = 1
            results['degrees'] = -1
        elif results['degrees'] == 0:
            results['c'] = 1
            results['degrees'] = -1

        if isinstance(expr.get_elt2(), const):
            results['c'] = expr.get_elt2().get_val()

        return results

    elif isinstance(expr, prod):  # 3 * x^1
        evaluate_expression(expr.get_mult1(), abc)
        results = evaluate_expression(expr.get_mult2(), abc)

        if results['degrees'] == 2:
            results['a'] = tof(expr.get_mult1())(0)
            results['degrees'] = -1
        elif results['degrees'] == 1:
            results['b'] = tof(expr.get_mult1())(0)
            results['degrees'] = -1
        elif results['degrees'] == 0:
            results['c'] = tof(expr.get_mult1())(0)
            results['degrees'] = -1

        return results

    elif isinstance(expr, pwr): # x^1
        if isinstance(expr.get_base(), var):
            abc['degrees'] = tof(expr.get_deg())(0)  # should be a 1
            return abc
        else:
            evaluate_expression(pwr.get_base(), abc)
    return abc

def find_poly_1_zeros(expr):
    abc = {'a': 0, 'b': None, 'c': None, 'degrees': -1}

    evaluate_expression(expr, abc)
    #bx + c => -c/b
    return const(-1 *abc['c'] / abc['b'])

def find_poly_2_zeros(expr):
    abc = {'a': None, 'b': None, 'c': None, 'degrees': -1}

    results = evaluate_expression(expr, abc)

    a = results['a']
    b = results['b'] or 0.0
    c = results['c'] or 0.0

    top1 = b * (-1)
    top2 = (b * b) - (4 * a * c)
    root = math.sqrt(top2)

    topPos = top1 + root
    topNeg = top1 - root

    bottom = 2 * a
    result1 = topPos / bottom
    result2 = topNeg / bottom

    return (const(result1), const(result2))
