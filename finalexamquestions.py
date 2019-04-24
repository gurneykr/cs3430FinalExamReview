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

    ##PROBLEM 2 Riemann sum stuff in defintegralapprox.py


##### SECTION 7 - CURVE FITTING ###########

    ##PROBLEM 1 - Bell curve
def gaussian_pdf(x, sigma=1, mu=0):
    a = 1.0 / (sigma * math.sqrt(2 * math.pi))
    b = math.e ** (-0.5 * (((x - mu) / sigma) ** 2))
    return a * b

def bell_curve_iq_approx(a, b):
    assert isinstance(a, const)
    assert isinstance(b, const)

    iqc = lambda x: gaussian_pdf(x, sigma=16.0, mu=100)
    print(simpson_rule(iqc, a, b, const(6)))

    #PROBLEM 2 - Method of least squares
def fit_regression_line(x, y):
    N = len(x)
    assert len(y) == N
    sum_xy = sum(xy[0]* xy[1] for xy in zip(x, y))
    sum_x = sum(xi for xi in x)
    sum_y = sum(yi for yi in y)
    sum_x_sqr = sum(xi**2 for xi in x)
    A = (1.0*(N*sum_xy - sum_x*sum_y))/(N*sum_x_sqr - sum_x**2)
    B = (sum_y - A*sum_x)/ (1.0*N)
    rlf = lambda x: A*x + B
    return rlf

    #PROBLEM 3 - Taylor polynomials
def taylor_poly(fexpr, a, n):
    assert isinstance(a, const)
    assert isinstance(n, const)

    tof_exp = tof(fexpr)
    ex = fexpr
    result = const(tof_exp(a.get_val()))

    for i in range(1, int(n.get_val())):
        drv = deriv(ex)
        drv_tof = tof(drv)

        inside = const(drv_tof(a.get_val()) / math.factorial(i))
        x = make_plus(make_pwr('x', 1.0), make_prod(const(-1.0), a))

        pw = make_pwr_expr(x, i)

        result = make_plus(result, make_prod(inside, pw))
        ex = drv

    return result


##### SECTION 8 - NEWTON-RAPHSON AND PELL EQUATIONS ######
    #PROBLEM 1 & 2
def newton_raphson(poly_fexpr, g, n):
    tof_expr = tof(poly_fexpr)
    deriv_expr = tof(deriv(poly_fexpr))

    for i in range(n.get_val()):
        x = g.get_val() - (tof_expr(g.get_val()) / deriv_expr(g.get_val()))
        g = const(x)

    return g

    #PROBLEM 4
# finds x and y that give the solution to
#the equation x^2 -ky^2 = +- 1
# def pell(n):
#     pairs = []
#
#     while len(pairs) < n:
#         for x in range(1, 10000):
#             for y in range(1, 1000):
#                 if abs(x**2 - 2*y**2) == 1:
#                     pairs.append((x,y))
#
#     return pairs

def pell(x1, x2, y1, y2):
    pairs = []
    f = lambda x,y: x*x - 2*x*y

    for x in range(x1, x2+1):
        for y in range(y1, y2+1):
            if abs(f(x,y)) == 1:
                if (x,y) not in pairs:
                    pairs.append((x, y))
    return pairs



def pell_approx_sqrt(n, a, b):
    pairs = []
    for i in range(1, n):
        x_n = ((a + math.sqrt(n)*b)**n + (a - math.sqrt(n)*b)**n) /2
        y_n = ((a + math.sqrt(n) * b) ** n + (a - math.sqrt(n) * b) ** n) / (2*math.sqrt(n))
        pairs.append((x_n, y_n))


    return pairs


##### SECTION 9 - LINEAR PROGRAMMING IN 2 VARIABLES #######
#### SECTION 10 - LINEAR SYSTEMS ###########




#### SECTION 11 - CVIP ###############