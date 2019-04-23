#!/usr/bin/python

#####################################
# module: antideriv.py
# Krista Gurney
# A01671888
#####################################

from const import const
from pwr import pwr
from var import var
from plus import plus
from prod import prod
from quot import quot
from tof import tof
from maker import make_const, make_pwr, make_prod, make_plus, make_ln, make_absv
import math
from consts import is_e_const, is_pi_const, is_zero_const

def antideriv(i):
    ## CASE 1: i is a constant
    if isinstance(i, const):
        #3 => 3x^1
        return prod(i, make_pwr('x', 1.0))
    ## CASE 2: i is a pwr
    elif isinstance(i, pwr):
        #x^d => 1/(d+1) * x^(d+1)
        b = i.get_base()
        d = i.get_deg()
        ## CASE 2.1: b is var and d is constant.
        if isinstance(b, var) and isinstance(d, const):
            if d.get_val() == -1:
                return make_ln(make_absv(pwr(b, const(1.0))))
            else:
                r = const(d.get_val() + 1.0)
                return prod(quot(const(1.0), r), pwr(b, r))
        ## CASE 2.2: b is e
        elif is_e_const(b):# e^(kx) => 1/k  * e(kx)
            if isinstance(d, prod):
                k = d.get_mult1()
                return prod(quot(const(1.0), k), i)
            else:
                raise Exception('antideriv: unknown case')
        # ## CASE 2.3: b is a sum
        elif isinstance(b, plus):#(1+x)^-3
            r = const(d.get_val()+1.0)
            if isinstance(d, const) and d.get_val() == -1:
                return make_ln(make_absv(b))
            elif isinstance(b.get_elt1(), prod):#(3x+2)^4 => 1/3 * anti(
                if isinstance(d, const) and d.get_val() < 0:
                    return prod(quot(const(-1.0), b.get_elt1().get_mult1()), pwr(b, r))
                else:
                    return prod(quot(const(1.0), prod(b.get_elt1().get_mult1(), r)), pwr(b, r))
            else:
                return prod(quot(const(1.0), r), pwr(b, r))
        else:
            raise Exception('antideriv: unknown case')
    ### CASE 3: i is a sum, i.e., a plus object.
    elif isinstance(i, plus):
        return plus(antideriv(i.get_elt1()), antideriv(i.get_elt2()))
    ### CASE 4: is is a product, i.e., prod object,
    ### where the 1st element is a constant.
    elif isinstance(i, prod):
        return prod(i.get_mult1(), antideriv(i.get_mult2()))
    else:
        raise Exception('antideriv: unknown case')

def antiderivdef(expr, a, b):
    assert isinstance(a, const)
    assert isinstance(b, const)

    anti_deriv = antideriv(expr)
    anti_deriv_tof = tof(anti_deriv)

    return const(anti_deriv_tof(b.get_val())- anti_deriv_tof(a.get_val()))
