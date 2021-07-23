# this script is to mute chromosomes
import fcm
import fcm.atmosphere as atm
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import choromosome as cho
import fitness as fit
import pandas as pd


def create_mate_pool(count, group_count, atmosphere, observation):
    """[create a pool to store potential parents]

    Parameters
    ----------
    count : [type]
        [description]
    group_count : [type]
        [description]
    atmosphere : [type]
        [description]
    observation : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # use dataframe to store the events
    event_list = []

    for i in range(count):
        structural_groups = cho.create_structural_group(3.44e3, 3760, group_count)

        meteoroid = cho.create_FCMmeteoroid(21.3, 81, 3.44e3, 1.14/2, 3760, 0,
                                            structural_groups)

        parameters = cho.create_FCMparameters(atmosphere, precision=1e-4,
                                              ablation_coeff=2.72e-8,
                                              cloud_disp_coeff=1,
                                              strengh_scaling_disp=0,
                                              fragment_mass_disp=0)

        result = fcm.simulate_impact(parameters, meteoroid, 100,
                                     craters=False, dedz=True, final_states=True)
        
        # get the error of each events
        error = fit.dEdz_error(observation, result.energy_deposition)

        event_list.append([structural_groups, meteoroid, parameters, error])

    fit.dEdz_fitness(event_list)

    for i in range(count):
        print(event_list[i][3])

    return event_list


def selection(parent_count, pool):
    """[select some events from event pool according to the fitness value]

    Parameters
    ----------
    intcount : [type]
        [description]
    """
    parent_list = []
    pool_count = len(pool)

    # find the highest fitness value and add it to the parents
    parent_list.append(max(pool, key=lambda x: x[3]))

    # choose parent using roulette
    


if __name__ == "__main__":
    atmosphere = atm.US_standard_atmosphere()

    observation = fit.read_event(fit.Event.tagish_lake)

    event_pool = create_mate_pool(10, 2, atmosphere, observation)

    # because the last element of each event error
    # here transport error to event fitness
    # the bigger error, the smaller fitness

    selection(1, event_pool)
