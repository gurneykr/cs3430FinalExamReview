#!/usr/bin/python

################################################
# module: consts.py
# bugs to vladimir kulyukin via canvas
################################################

import math
from const import const
from line_eq import line_eq
from var import var

def is_e_const(c):
  if isinstance(c, const):
    return c.get_val() == math.e
  else:
    return False

def is_zero_const(c):
  if isinstance(c, const):
    return c.get_val() == 0.0
  else:
    return False

def is_pi_const(c):
  if isinstance(c, const):
    return c.get_val() == math.pi
  else:
    return False

def is_const_line(ln):
  if isinstance(ln, line_eq):
    return isinstance(ln.get_lhs(), var) and isinstance(ln.get_rhs(), const)
  else:
    return False












