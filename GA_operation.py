# this script is to mute chromosomes
import fcm
import fcm.atmosphere as atm
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.sparse.construct import rand
import chromosome as ch
import fitness as fit
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import tools as t

# get the atmosphere of earth
atmosphere = atm.US_standard_atmosphere()

def linear_regression(observation):
    """[summary]

    Parameters
    ----------
    observation : [type]
        [description]
    """
    X = observation['altitude [km]'].values.reshape(-1, 1)
    y = observation['dEdz [kt TNT / km]'].values

    # second degree polynomial
    poly = PolynomialFeatures(degree=5)
    poly.fit(X)
    X2 = poly.transform(X)
    reg = LinearRegression().fit(X2, y)
    y_pre = reg.predict(X2)

    return reg, poly

def mate_pool_generater(group_count, event_count):
    """[generate the pool of chromosomes to mate]

    Parameters
    ----------
    group_count : [int]
        [the count of structural groups]
    event_count : [int]
        [the count of meteoroid events]
    """

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
                                        'strengh_scaling_disp',
                                        'fragment_mass_disp',
                                        'fitness_value'])

    # ############# define the characteristics of meteoroids ################
    # log uniform distribution
    density = ch.RA_logdis_float(1500, 5000)
    strength = ch.RA_logdis_float(1, 10000)

    # uniform distribution
    cloud_frac = ch.RA_uniform_float(1.0, 1, 0.1, 0.9)[0]

    # ######### the observed tets #########
    observation = fit.read_event(fit.Event.benesov)
    # make observation round to 2 decimal places
    observation = observation.round({'altitude [km]': 2})
    
    total_energy = t._total_energy(observation)
    # generate the events
    for i in range(event_count):
        # generate structural groups
        ch.groups_generater(groups_frame, density, strength, group_count,
                            cloud_frac)
        ch.meteroid_generater(meteoroids_frame, 21.3, 81, density,
                              strength, cloud_frac, total_energy=total_energy,
                              ra_radius=True, ra_velocity=True,
                              ra_angle=True)
        ch.FCMparameters_generater(param_frame, ablation_coeff=2.72e-8,
                                   cloud_disp_coeff=1,
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
        params = fcm.FCMparameters(9.81, 6371, atmosphere,
                                   ablation_coeff=param['ablation_coeff'],
                                   cloud_disp_coeff=1, dh=100,
                                   strengh_scaling_disp=0,
                                   lift_coeff=0, drag_coeff=1,
                                   fragment_mass_disp=0, precision=1e-2)
        # simulate
        simudata = fcm.simulate_impact(params, meteroid_params, 100,
                                       craters=False, dedz=True, final_states=True)
        
        # get the mean square error of dEdz
        param['fitness_value'] = fit.dEdz_error(observation, simudata.energy_deposition)

    # get the fitness of dEdz
    fit.dEdz_fitness(param_frame)

    return groups_frame, meteoroids_frame, param_frame


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

    # choose parents using roulette algorithm
    roulette = np.zeros(pool_count)
    temp = 0
    for i in range(pool_count):
        temp += pool[i][3]
        roulette[i] = temp

    target_count = 0
    while (target_count < parent_count):
        for i in range(pool_count):
            random_prab = random.uniform(0.0, 1)

    return parent_list

if __name__ == "__main__":
    group_dataframe, meteoroids_frame, param_frame = mate_pool_generater(group_count=1, event_count=100)

    print(param_frame)
