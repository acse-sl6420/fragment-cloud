# This python file is to find how to design the data structures of a chromosome
import os, sys

from numpy import random
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_PATH = os.path.abspath(os.path.join(THIS_DIR, ".."))
if sys.path[0] != BASE_PATH:
    sys.path.insert(0, BASE_PATH)

import fcm
import fcm.atmosphere as atm

from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ######### Firstly set a default count of structure group ###########
group_count = 8

# The first chromosome includes the information of structural groups
# use datafram to represent a structural group

"""[Random generate several float numbers which sum up to a given number]
"""
def random_fraction(count, summation, is_even=False):
    if not is_even:
        # random generate several numbers
        random_num = np.random.rand(count)
        # get the ratio of each number
        ratio = summation / sum(random_num)
        result = random_num * ratio
    else:
        result = [summation/count for i in range(count)]
    
    return result


"""[Randomly generate float numbers]
"""
def RA_float(amplifier, lower_bound=0.1, high_bound=1.0,
             count=group_count):
    result = [amplifier * np.random.uniform(lower_bound, high_bound)
              for i in range(count)]
    return result

"""[Randomly generate int numbers]
"""
def RA_int(count, amplifier=1, lower_bound=1, high_bound=16):
    result = [int(amplifier * random.randint(lower_bound, high_bound))
              for i in range(count)]
    return result


"""[Randomly generate tuples which summation of fragments is 1.0]
"""
def RA_tuple(count_max=16, is_even=True):
    # randomly generate the count of this tuple
    count = RA_int(count_max)[0]
    # get the tuple
    result = random_fraction(count, 1, is_even)

    return result


# the index of parameters of structural group
class params_sg():
    mass_fraction = 0
    density = 1
    strength = 2
    pieces = 3
    cloud_mass_frac = 4
    strength_scaler = 5
    fragment_mass_fractions = 6


"""[Generate the structural groups]
"""
def create_structural_group(density, min_strength):
    # The first six parameters are below:
    #   mass_fraction (not suitable for mutation)
    #   Density (the same with the FCMmeteoroid)
    #   Strength (bigger than the one in the FCMmeteoroid)
    #   Pieces (could be a random number)
    #   cloud_mass_frac ( could be a random float number)
    #   strength_scaler ( could be a random float number)
    #
    # The last parameter of structural groups is:
    #   fragment_mass_fractions (tuple)
    param_count = 6
    groups = np.zeros((group_count, param_count))
    fragment_mass_fractions = []

    # random generate the numbers
    groups[:, params_sg.cloud_mass_frac] = random_fraction(group_count, 1)
    groups[:, params_sg.density] = density
    groups[:, params_sg.strength] = RA_float(min_strength, 1.1, 2.0)
    groups[:, params_sg.pieces] = RA_int(group_count)
    groups[:, params_sg.cloud_mass_frac] = RA_float(1, 0.1, 1)
    groups[:, params_sg.strength_scaler] = RA_float(1, 0.1, 1)
    fragment_mass_fractions = [RA_tuple() for i in range(group_count)]

    structural_groups = []
    for i in range(group_count):
        structural_groups.append(fcm.StructuralGroup(
                                groups[i, params_sg.cloud_mass_frac],
                                groups[i, params_sg.density],
                                groups[i, params_sg.strength],
                                int(groups[i, params_sg.pieces]),
                                groups[i, params_sg.cloud_mass_frac],
                                groups[i, params_sg.strength_scaler],
                                fragment_mass_fractions[i]))
    return structural_groups


structural_groups = create_structural_group(10, 10)
