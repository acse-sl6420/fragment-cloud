# This file is to calculate the fitness of every population
import os
import sys

import fcm
import fcm.atmosphere as atm
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import choromosome as chsome
from sklearn.metrics import mean_squared_error

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_PATH = os.path.abspath(os.path.join(THIS_DIR, ".."))
if sys.path[0] != BASE_PATH:
    sys.path.insert(0, BASE_PATH)


# the index class of observed event
class Event():
    chelyabinsk = 0
    kosice = 1
    benesov = 2
    tagish_lake = 3


# the csv file name
file_names = {Event.chelyabinsk: "ChelyabinskEnergyDep_Wheeler-et-al-2018.txt",
              Event.kosice: "KosiceEnergyDep_Wheeler-et-al-2018.txt",
              Event.benesov: "BenesovEnergyDep_Wheeler-et-al-2018.txt",
              Event.tagish_lake: "TagishLakeEnergyDep_Wheeler-et-al-2018.txt"}


def read_event(event):
    """[read the data of event from csv files.]

    Parameters
    ----------
    event : [Class Event]
        [this class includes the observed events.]
    """
    data = pd.read_csv(os.path.join(THIS_DIR, "data", file_names[event]),
                       sep='\t', header=0, index_col=0)
    
    data.columns = ["dEdz [kt TNT / km]", "min. dEdz [kt TNT / km]", "max. dEdz [kt TNT / km]"]
    data['altitude [km]'] = data.index
    data = data.reset_index(drop=True)
    
    return data

def dEdz_fitness(event_pool):
    """[transport error to fitness]

    Parameters
    ----------
    event_pool : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    event_count = len(event_pool)
    fitness_index = 3
    error_sum = 0
    fitness_sum = 0

    for i in range(event_count):
        error_sum += event_pool[i][fitness_index]

    for i in range(event_count):
        event_pool[i][fitness_index] /= error_sum
        event_pool[i][fitness_index] = 1 - event_pool[i][fitness_index]

    # get the summation of fitness
    for i in range(event_count):
        fitness_sum += event_pool[i][fitness_index]

    for i in range(event_count):
        event_pool[i][fitness_index] /= fitness_sum

    return event_pool


def dEdz_error(observation, dEdz):
    """[get the fitness value]

    Parameters
    ----------
    observation : [type]
        [description]
    dEdz : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    dEdz = pd.DataFrame(data=dEdz, columns={'dEdz'})
    dEdz['altitude'] = dEdz.index
    dEdz = dEdz.reset_index(drop=True)
    paired_data = observation.merge(dEdz, left_on='altitude [km]', right_on = 'altitude')

    # get the fitness value by RSME
    fitness = mean_squared_error(paired_data['dEdz [kt TNT / km]'].to_numpy(),
                                 paired_data['dEdz'].to_numpy())

    return fitness


if __name__ == "__main__":
    group_count = 16
    atmosphere = atm.US_standard_atmosphere()
    observation = read_event(Event.tagish_lake)

    structural_groups = chsome.create_structural_group(3.44e3, 3760, group_count)
    chsome.print_structural_groups(structural_groups, group_count)
    meteoroid = chsome.create_FCMmeteoroid(21.3, 81, 3.44e3, 1.14/2, 3760, 0,
                                           structural_groups)
    parameters = chsome.create_FCMparameters(atmosphere, precision=1e-4,
                                             ablation_coeff=2.72e-8,
                                             cloud_disp_coeff=1,
                                             strengh_scaling_disp=0,
                                             fragment_mass_disp=0)

    simudata = fcm.simulate_impact(parameters, meteoroid, 100,
                                   craters=False, dedz=True, final_states=True)

    fitness_value = dEdz_error(observation, simudata.energy_deposition)
    print(fitness_value)
