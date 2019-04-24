from const import const
from maker import make_const, make_pwr, make_pwr_expr, make_plus, make_prod, make_quot, make_e_expr, make_ln, make_absv
from tof import tof
from riemann import riemann_approx, riemann_approx_with_gt, plot_riemann_error
from deriv import deriv
from antideriv import antideriv, antiderivdef
from defintegralapprox import midpoint_rule, trapezoidal_rule, simpson_rule
import math
import numpy as np
import matplotlib.pyplot as plt

######SECTION 2 - DIFFERENTIATION



########## SECTION 3 - THEORY OF THE FIRM #########

#PROBLEM 1


#PROBLEM 2

#PROBLEM 3 - demand elasticity f(p) = 100 - 2p
def demand_elasticity(p):
    assert isinstance(p, const)
    demand_eq = make_plus(make_const(100), make_prod(make_const(-2.0), make_pwr('p', 1.0)))
    fx_drv = tof(deriv(demand_eq))
    fx = tof(demand_eq)

    #E(p) = (-p*f'(p)) / f(p)

    num = (-1.0*p.get_val()) * fx_drv(p.get_val())
    denom = fx(p.get_val())

    return num / denom

#PROBLEM 4 - net change
def net_change(mrexpr, pl1, pl2):
    assert isinstance(pl1, const)
    assert isinstance(pl2, const)

    #F(b) - F(a)
    a = pl1
    b = pl2
    F = tof(antideriv(mrexpr))

    return F(b.get_val()) - F(a.get_val())

#PROBLEM 5 - consumer surplus - lecture 14
# f(x)-B evaluated from a to 0
# where B = f(a)
def consumer_surplus(dexpr, a):
    assert isinstance(a, const)

    B = const(-1 * tof(dexpr)(a.get_val()))

    f = make_plus(dexpr, B)
    surplus = tof(antideriv(f))
    return surplus(a.get_val()) - surplus(0)


######## SECTION 4 RATES OF CHANGE AND 1D FUNCTION OPTIMIZATION #####





#######  SECTION 5 - GROWTH, DECAY, TERMINAL VELOCITY, PARTIAL DIFFERENTIAL EQUATIONS #####





##### SECTION 6 - ANTIDIFFERENTIATION ############



##### SECTION 7 - CURVE FITTING ###########



##### SECTION 8 - NEWTON-RAPHSON AND PELL EQUATIONS ######




##### SECTION 9 - LINEAR PROGRAMMING IN 2 VARIABLES #######



#### SECTION 10 - LINEAR SYSTEMS ###########




#### SECTION 11 - CVIP ###############