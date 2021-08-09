# This file is to calculate the fitness of every population
import os
import sys

import fcm
import fcm.atmosphere as atm
from enum import Enum
import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pyplot as plt
import chromosome as ch
import tools as t
from sklearn.metrics import mean_absolute_error

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
    event_pool : [dataframe]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    error_sum = 0
    fitness_sum = 0

    # the summation of fitness error
    error_sum = event_pool['fitness_value'].sum()

    # get propotion of fitness value in summation
    event_pool['fitness_value'] = event_pool.apply(lambda x: 1 - (x['fitness_value'] / error_sum), axis=1)

    # get the summation of fitness
    fitness_sum = event_pool['fitness_value'].sum()
    event_pool['fitness_value'] = event_pool.apply(lambda x: x['fitness_value'] / fitness_sum, axis=1)

    return event_pool


def dEdz_error(observation, dEdz):
    """[get the difference betwee observation and dEdz]

    Parameters
    ----------
    observation : [type]
        [description]
    dEdz : [type]
        [description]
    """
    # make observation round to 2 decimal places
    # observation = observation.round({'altitude [km]': 2})
    dEdz = pd.DataFrame(data=dEdz, columns={'dEdz'})
    dEdz['altitude'] = dEdz.index
    dEdz = dEdz.reset_index(drop=True)

    # merge the observation and dEdz at the same altitude
    paired_data = observation.merge(dEdz, left_on='altitude [km]', right_on = 'altitude')
    error = mean_absolute_error(paired_data['dEdz [kt TNT / km]'].to_numpy(),
                                paired_data['dEdz'].to_numpy())
    max_error = (abs(paired_data['dEdz [kt TNT / km]'].to_numpy() - paired_data['dEdz'].to_numpy())).max()
    
    error = 10 * error + max_error
    return error
    

def dEdz_error_poly(reg, poly, observation, dEdz):
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
    mask = np.logical_and(dEdz.index.to_numpy() >= observation.index.min(),
                          dEdz.index.to_numpy() <= observation.index.max())

    dEdz = pd.DataFrame(data=dEdz, columns={'dEdz'})

    X = dEdz.index.to_numpy()[mask].reshape(-1, 1)

    # transform X to 2 degree polynomial
    X2 = poly.transform(X)

    # get the prediction Y
    Y = reg.predict(X2)
    
    # get the absolute_error
    # error = mean_absolute_error(Y, dEdz.to_numpy()[mask].flatten())
    error = sum(Y, dEdz.to_numpy()[mask].flatten())

    return error



if __name__ == "__main__":
    #define the numbers of structural groups
    group_count = 2
    # the count of events
    event_count = 1

    # create a dataframe to store structural_groups
    groups_frame = pd.DataFrame(columns=['mass_fraction', 'density',
                                   'strength', 'pieces',
                                   'cloud_mass_frac', 'strength_scaler',
                                   'fragment_mass_fractions'])
    
    # create a dataframe to store FCMmeteoroids
    meteoroids_frame = pd.DataFrame(columns=['velocity', 'angle',
                                       'density', 'radius',
                                       'strength', 'cloud_mass_frac'])
    # create a dataframe to store paramters
    param_frame = pd.DataFrame(columns=['ablation_coeff', 'cloud_disp_coeff',
                                       'strengh_scaling_disp', 'fragment_mass_disp',
                                       'fitness_value'])

    # ############# define the characteristics ################
    # atmosphere
    atmosphere = atm.US_standard_atmosphere()

    # log uniform distribution
    density = ch.RA_logdis_float(1500, 5000)
    strength = ch.RA_logdis_float(1, 10000)

    # uniform distribution
    cloud_frac = ch.RA_uniform_float(1.0, 1, 0.1, 0.9)[0]

    # genarate structural groups
    structural_group_count = 2

    # #### TODO :the radius temporarily set to 2.5 ########
    radius = 1

    # # ######### the observed tets #########
    observation = read_event(Event.benesov)
    # get total energy deposition in J
    total_energy = t._total_energy(observation)

    # generate the events
    for i in range(event_count):
        # generate structural groups
        ch.groups_generater(groups_frame, density, strength, group_count)
        ch.meteroid_generater(meteoroids_frame, 21.3, 81, density, radius,
                              strength, cloud_frac, total_energy, ra_radius=True,
                              ra_angle=True, ra_velocity=True)
        ch.FCMparameters_generater(param_frame, ablation_coeff=1e-8,
                                   cloud_disp_coeff=2/3.5,
                                   strengh_scaling_disp=0,
                                   fragment_mass_disp=0,
                                   RA_ablation=True)
        
        # simulate this event
        # get the groups list
        groups_list = ch.compact_groups(groups_frame, i, group_count)

        # meteoroid
        me = meteoroids_frame.loc[i]
        meteroid_params = fcm.FCMmeteoroid(me['velocity'], me['angle'],
                                           me['density'], me['radius'],
                                           me['strength'], me['cloud_mass_frac'],
                                           groups_list)
        # parameters
        param = param_frame.loc[i]
        params = fcm.FCMparameters(9.81, 6371, atmosphere, ablation_coeff=2.72e-8,
                                   cloud_disp_coeff=1, strengh_scaling_disp=0,
                                   fragment_mass_disp=0, precision=1e-2)
        # simulate
        simudata = fcm.simulate_impact(params, meteroid_params, 100,
                                       craters=False, dedz=True, final_states=True)

        # get the fitness_value
        param['fitness_value'] = dEdz_error(observation, simudata.energy_deposition)

    dEdz_fitness(param_frame)
