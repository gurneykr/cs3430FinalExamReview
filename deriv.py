#!/usr/bin/python

####################################
# Krista
# A01671888
####################################

from var import var
from const import const
from pwr import pwr
from prod import prod
from plus import plus
from quot import quot
from ln import ln
from maker import make_ln
from absv import absv

def deriv(expr):
    if isinstance(expr, const):
        return const_deriv(expr)
    elif isinstance(expr, pwr):
        return pwr_deriv(expr)
    elif isinstance(expr, prod):
        return prod_deriv(expr)
    elif isinstance(expr, plus):
        return plus_deriv(expr)
    elif isinstance(expr, quot):
        return quot_deriv(expr)
    elif isinstance(expr, ln):
        return ln_deriv(expr)
    elif isinstance(expr, absv):
        return absv_deriv(expr)
    else:
        raise Exception('deriv:' + repr(expr))

# the derivative of a consant is 0.
def const_deriv(c):
    assert isinstance(c, const)
    return const(val=0.0)

def plus_deriv(s):
    if isinstance(s.get_elt2(), const):
        # Simplified - if the second argument is a constant, then it becomes zero and there is no need to add it.
        return deriv(s.get_elt1())
    else:
        return plus(deriv(s.get_elt1()), deriv(s.get_elt2()))

def pwr_deriv(p):
    assert isinstance(p, pwr)
    b = p.get_base()
    d = p.get_deg()
    if isinstance(b, const):
        if isinstance(d, pwr): #e^(x^1)
            return p
        elif isinstance(d, prod):#85 *e^(-0.5 *t)    => 85 *-0.5 *e^(-0.5 *t)
            return prod(p, deriv(d))
        #e^(2x)
        elif isinstance(d, plus):
            return prod(p, deriv(d))
    if isinstance(b, var):
        if isinstance(d, const):
            return prod(d, pwr(b, const(d.get_val()-1)))
        elif isinstance(d, plus):
            return prod(d, pwr(b, const(d.get_val()-1)))
        else:
            raise Exception('pwr_deriv: case 1: ' + str(p))
    if isinstance(b, pwr):  # think this is (x^2 (^3))
        if isinstance(d, const):
           return prod(b.get_base(), prod(b.get_deg(), const))
        elif isinstance(d, pwr):#e^(x^1)
            return p
        else:
            raise Exception('pwr_deriv: case 2: ' + str(p))
    elif isinstance(b, plus):  # (4x+2)^3 => 3(4x+2)^2 *4
        if isinstance(d, const):
            return prod(prod(d, pwr(b, const(d.get_val()-1))), deriv(b))
        else:
            raise Exception('pwr_deriv: case 3: ' + str(p))
    elif isinstance(b, prod):#(3x)^2 => (2*3*x)^(2-1)
        if isinstance(d, const):
            return pwr( prod(d, prod(b.get_mult1(), b.get_mult2())), const(d.get_val()-1))
        else:
            raise Exception('pwr_deriv: case 4: ' + str(p))
    elif isinstance(b, quot):
        if isinstance(d, const):
            #b*d * deriv(quot)^d-1
            return prod(prod(d, pwr(b, const(d.get_val()-1))) ,deriv(b))
        else:
            raise Exception('power_deriv: case 5: ' + str(p))
    elif isinstance(b, ln):#(lnx)^5
        return prod(prod(d, pwr(b, const(d.get_val() - 1))), deriv(b))
    else:
        raise Exception('power_deriv: case 6: ' + str(p))

def prod_deriv(p):
    assert isinstance(p, prod)
    m1 = p.get_mult1()  # 6
    m2 = p.get_mult2()  # x^3
    if isinstance(m1, const):
        if isinstance(m2, const):
            return const(0)
        elif isinstance(m2, pwr):  # 6*(x^3)=> 6*3*(x^(3-1))
            # 3x^1  becomes (1*3)x^0 => simplified is 3
            if isinstance(m2.get_deg(), const) and m2.get_deg().get_val() == 1:
                return m1
            elif isinstance(m2.get_deg(), const) and m2.get_deg().get_val() == 0:
                return m1
            elif isinstance(m2.get_deg(), prod):
                return prod(m1, deriv(m2))
            else:
                # get 6 * 3
                simplifiedAlt1 = const(m1.get_val() * m2.get_deg().get_val())
                # get x^3-1
                simplifiedExp = const(m2.get_deg().get_val() - 1)
                alt2 = pwr(m2.get_base(), simplifiedExp)
                return prod(simplifiedAlt1, alt2)
        elif isinstance(m2, plus):  # 3*(x+1)
            if isinstance(deriv(m2), const):
                return const(0)
            else:
                return prod(m1, deriv(m2))
        elif isinstance(m2, prod):  # 4*(3x)
            if isinstance(deriv(m2), const):
                return const(0)
            else:
                return prod(m1, deriv(m2))
        elif isinstance(m2, ln):
            return prod(m1, deriv(m2))
        else:
            raise Exception('prod_deriv: case 0' + str(p))
    elif isinstance(m1, plus):
        if isinstance(m2, const):  # (x+1)*3
            if isinstance(deriv(m2), const):
                return const(0)
            else:
                return prod(m1, deriv(m2))
        elif isinstance(m2, plus):#(x+1)(x+2)
            return plus(deriv(m1), deriv(m2))
        else:
            raise Exception('prod_deriv: case 1:' + str(p))
    elif isinstance(m1, pwr):
        if isinstance(m2, const):  # (x^2)*3 => (2x^1)*3
            if isinstance(deriv(m2), const):
                return const(0)
            else:
                return prod(deriv(m1), m2)
        elif isinstance(m2, ln):#(x^3) * (ln x)
            return plus(prod(deriv(m1), m2), prod(m1, deriv(m2)))
        elif isinstance(m2, pwr):#(x^1)*(e^(x^1))
            return plus(prod(deriv(m1), m2), prod(m1, deriv(m2)))
        elif isinstance(m2, prod):#(x^2)(2x)
            return plus(prod(m1, deriv(m2)), prod(m2, deriv(m1)))
        else:
            raise Exception('prod_deriv: case 2:' + str(p))

    elif isinstance(m1, prod) or isinstance(m1, quot):
        if isinstance(m2, const):#(3x)*4
            if isinstance(deriv(m2), const):
                return const(0)
            else:
                return prod(deriv(m1), m2)
        else:
            return plus(prod(m1, deriv(m2)), prod(m2, deriv(m1)))
            # return prod(m1, deriv(m2))
    else:
       raise Exception('prod_deriv: case 4:' + str(p))

def quot_deriv(p):# f/g = (gf'-fg')/g^2 quotient rule
    assert isinstance(p, quot)
    f = p.get_num()
    g = p.get_denom()
    if isinstance(f, const) and isinstance(g, const):
        return const(0)
    else:
        return quot(plus(prod(g, deriv(f)), prod(const(-1),prod(f, deriv(g)))), pwr(g, const(2.0)))

def ln_deriv(p):
    #ln[g(x)] = 1/g(x) * g'(x)
    assert isinstance(p, ln)
    g = p.get_expr()
    if isinstance(g, prod):
        m1 = g.get_mult1()
        m2 = g.get_mult2()
        #(x)(x+1)
        #ln(x) + ln(x+1)
        if isinstance(m1, pwr) and isinstance(m2, pwr):
            return plus(quot(prod(m1.get_deg(), deriv(m1.get_base())), m1.get_base()),
                        quot(prod(m2.get_deg(), deriv(m2.get_base())), m2.get_base()))
        return plus(ln_deriv(make_ln(m1)), ln_deriv(make_ln(m2)))
    else:
        return prod(quot(const(1.0), g), deriv(g))

def absv_deriv(p):
    assert isinstance(p, absv)
    return deriv(p.get_expr())

def logdiff(p):
    # assert isinstance(p, prod)
    # if isinstance(p, prod):
    #     m1 = p.get_mult1()
    #     m2 = p.get_mult2()
    #
    #     if isinstance(m1, plus):#(x+1)(x+2)
    #         if isinstance(m2, plus):
    #             return prod(p, plus(prod(quot(const(1.0), m1),
    #                                  deriv(m1)),
    #                             prod(quot(const(1.0), m2),
    #                                  deriv(m2))))
    #     elif isinstance(m1, pwr):#x(x+1)(x+3)
    #         if isinstance(m2, prod):
    #             #return prod(, logdiff(m2))
    #             return prod(p, ln_deriv(make_ln(p)))

    return prod(p, ln_deriv(make_ln(p)))

'''want:
p *     ((1/m1)*deriv(m1)) + ((1/m2)*deriv(m2))

(((x^1.0)*(((x^1.0)+1.0)*((x^1.0)+2.0)))*    ( ( (1.0/(x^1.0)) * (1.0*(x^0.0)) )+
                                        (( (1.0/((x^1.0)+1.0)) *((1.0*(x^0.0))+0.0))+((1.0/((x^1.0)+2.0))* ((1.0*(x^0.0))+0.0))))


my results
(((x^1.0)*(((x^1.0)+1.0)*((x^1.0)+2.0))) *  ( ( (1.0/(x^1.0)) * (1.0*(x^0.0)) ) +
                                        ((1.0/(((x^1.0)+1.0)*((x^1.0)+2.0)))*((1.0*(x^0.0))+(1.0*(x^0.0))))))
'''
