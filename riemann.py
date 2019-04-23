#!/usr/bin/python

###############################################
# module: riemann.py
# Krista Gurney
# A01671888
###############################################

## modify these imports as you see fit.
import numpy as np
from const import const
from antideriv import antiderivdef, antideriv
from tof import tof
import matplotlib.pyplot as plt
import numpy as np
from maker import make_const

def riemann_approx(fexpr, a, b, n, pp=0):
    '''
    pp=0 - approximate with reimann midpoint
    pp=+1 - approximate with reimann right point
    pp=-1 - approximate with reiman left point
    '''
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(n, const)

    area = 0
    fex_tof = tof(fexpr)
    partition = (b.get_val() - a.get_val())/ n.get_val()

    a = int(a.get_val())
    b = int(b.get_val())

    if pp == -1: #left riemann
        for i in np.arange(a, b, partition):
            area += fex_tof(i)*partition
    elif pp == 1: #right riemann
        for i in np.arange(a+partition, b+partition, partition):
            area += fex_tof(i)*partition
    elif pp == 0: #midpoint riemann
        for i in np.arange(a, b, partition):
            mid = i + (partition/2)
            area += fex_tof(mid)*partition
    return const(area)

def riemann_approx_with_gt(fexpr, a, b, gt, n_upper, pp=0):
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(gt, const)
    assert isinstance(n_upper, const)

    #call riemann approx  n_upper times and calculate the error
    err_list = []
    for i in range(1, n_upper.get_val()+1):
        result = riemann_approx(fexpr, a, b, const(i), pp)
        err = abs(gt.get_val() - result.get_val())
        err_list.append((i, err))
    return err_list

def plot_riemann_error(fexpr, a, b, gt, n):
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(gt, const)
    assert isinstance(n, const)

    err_left = riemann_approx_with_gt(fexpr, a, b, gt, n, -1)
    err_right = riemann_approx_with_gt(fexpr, a, b, gt, n, 1)
    err_midpoint = riemann_approx_with_gt(fexpr, a, b, gt, n, 0)

    xvals = []
    for i in range(1, n.get_val()+1):
        xvals.append(i)

    fig1 = plt.figure(1)
    fig1.suptitle('Riemann Approximation Error')
    plt.xlabel('n')
    plt.ylabel('err')
    plt.ylim([0, 7.5])
    plt.xlim([0, n.get_val()])
    plt.grid()
    plt.plot(xvals, err_left, label='left', c='g')
    plt.plot(xvals, err_right, label='right', c='b')
    plt.plot(xvals, err_midpoint, label='midpoint', c='r')

    plt.legend(loc='best')
    plt.show()




