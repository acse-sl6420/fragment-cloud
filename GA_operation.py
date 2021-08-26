# this script is to develop the evolvement process of GA
import fcm
import fcm.atmosphere as atm
import numpy as np
import matplotlib.pyplot as plt
from numpy import random, vectorize
from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
from scipy.sparse.construct import rand
import chromosome as ch
import fitness as fit
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import tools as t
import random

# get the atmosphere of earth
atmosphere = atm.US_standard_atmosphere()


def mate_pool_generater(group_count, event_count, observation,
                        velocity, angle,
                        strength_lower_bound=1, strength_higher_bound=10000):
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
    # todo: cloud_mass_frac = 0
    meteoroids_frame = pd.DataFrame(columns=['velocity', 'angle',
                                             'density', 'radius',
                                             'strength', 'cloud_mass_frac'])
    # create a dataframe to store paramters
    param_frame = pd.DataFrame(columns=['ablation_coeff', 'cloud_disp_coeff',
                                        'strengh_scaling_disp',
                                        'fragment_mass_disp',
                                        'fitness_value'])

    total_energy = t._total_energy(observation)
    # generate the events
    for i in range(event_count):
        # log uniform distribution
        # density = ch.RA_logdis_float(1500, 5000)
        density = ch.RA_logdis_float(2500, 3500)
        # density = 3000
        strength = ch.RA_logdis_float(strength_lower_bound,
                                      1000)

        # generate structural groups
        ch.groups_generater(groups_frame, density, strength, group_count,
                            strength_higher_bound)
        ch.meteroid_generater(meteoroids_frame, velocity, angle, density,
                              strength, cloud_mass_frac=0,
                              total_energy=total_energy,
                              ra_radius=True)
        ch.FCMparameters_generater(param_frame, ablation_coeff=2.72e-8,
                                   cloud_disp_coeff=1,
                                   strengh_scaling_disp=0,
                                   fragment_mass_disp=0)

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
                                   cloud_disp_coeff=1,
                                   strengh_scaling_disp=0,
                                   lift_coeff=0, drag_coeff=1,
                                   fragment_mass_disp=0, precision=1e-2)
        # simulate
        simudata = fcm.simulate_impact(params, meteroid_params, 100,
                                       craters=False, dedz=True, final_states=True)

        # get the error of dEdz
        param['fitness_value'] = fit.dEdz_error(observation, simudata.energy_deposition)

    # get the fitness of dEdz
    fit.dEdz_fitness(param_frame)

    return groups_frame, meteoroids_frame, param_frame


def accumulate_probability(pool):
    """[get the accumulative probability of each event]

    Parameters
    ----------
    pool : [Dataframe]
        [store the parameters and fitness]
    """
    pool_count = len(pool)
    
    # get the accumulative probability
    for i in range(pool_count-1):
        pool.loc[i+1, 'fitness_value'] += pool.loc[i, 'fitness_value']
    

def selection(parent_count, pool):
    """[select some events from event pool according to the fitness value]

    Parameters
    ----------
    parent_count : [int]
        [Should be lower than event_count]
    pool : [DataFrame]
        [The last column of it is the information of fitness]

    Returns
    -------
    [type]
        [description]
    """
    pool_count = len(pool)
    assert parent_count <= pool_count, "the parent_count <= pool_count"

    # get the highest fitness in pool
    parents_index = []
    parents_index.append(pool['fitness_value'].idxmax())
    
    # get the accumulative probability
    for i in range(pool_count-1):
        pool.loc[i+1, 'fitness_value'] += pool.loc[i, 'fitness_value']
    
    # generate the parents using Roulette algorithm
    for i in range(parent_count - 1):
        # generate a random float number between 0 to 1
        prob = ch.RA_uniform_float(1.0, 1, 0, 1)
        
        # traverse the list to find the parent
        for i in range(pool_count-1):
            if prob >= pool.loc[i, 'fitness_value'] and prob < pool.loc[i+1, 'fitness_value']:
                parents_index.append(i+1)

    return parents_index


def update_fitness(group_dataframe, meteoroids_frame, param_frame,
                   group_count, observation, fitness_list,
                   precision):
    """[update the fitness after cross-over or mutation]

    Parameters
    ----------
    group_dataframe : [type]
        [description]
    meteoroids_frame : [type]
        [description]
    param_frame : [type]
        [description]
    group_count : [type]
        [description]
    observation : [type]
        [description]
    highest : [type]
        [description]
    is_break : bool, optional
        [description], by default False
    """
    # indicate whether the fitness error is lower than the threshold
    highest = -1
    is_break = False
    for index in range(len(meteoroids_frame)):
        groups_list = ch.compact_groups(group_dataframe, index, group_count)

        # meteoroid
        me = meteoroids_frame.loc[index]
        meteroid_params = fcm.FCMmeteoroid(me['velocity'], me['angle'],
                                           me['density'], me['radius'],
                                           me['strength'], me['cloud_mass_frac'],
                                           groups_list)
        # parameters
        param = param_frame.loc[index]

        params = fcm.FCMparameters(9.81, 6371, atmosphere,
                                   ablation_coeff=param['ablation_coeff'],
                                   cloud_disp_coeff=1,
                                   strengh_scaling_disp=0,
                                   lift_coeff=0, drag_coeff=1,
                                   fragment_mass_disp=0, precision=1e-2)
        # simulate
        simudata = fcm.simulate_impact(params, meteroid_params, 100,
                                       craters=False, dedz=True, final_states=True)
        
        # get the error of dEdz
        param['fitness_value'] = fit.dEdz_error(observation, simudata.energy_deposition)
        fitness_list[index] = param['fitness_value']

        if param['fitness_value'] < precision:
            is_break = True
            highest = index

    # get the fitness of dEdz
    fit.dEdz_fitness(param_frame)

    return is_break, highest


def choose_parent(p_df):
    """[choose parents from the population]

    Parameters
    ----------
    p_df : [DataFrame]
        [The DataFrame stores the population]

    Returns
    -------
    [int]
        [The index of parents]
    """
    # traverse the list to find the parent
    prob1 = ch.RA_uniform_float(1.0, 1, 0, 1, round=".9f")[0]

    parent_1 = 0
    parent_2 = 0
    is_break = False
            
    for i in range(len(p_df) - 1):
        if prob1 >= p_df.loc[i, 'fitness_value'] and prob1 < p_df.loc[i+1, 'fitness_value']:
            parent_1 = i
            break

    prob2 = ch.RA_uniform_float(1.0, 1, 0, 1, round=".9f")[0]
    for i in range(len(p_df) - 1):
        if prob2 >= p_df.loc[i, 'fitness_value'] and prob2 < p_df.loc[i+1, 'fitness_value']:
            parent_2 = i
            break
    return parent_1, parent_2


def cross_over(g_df, m_df, p_df, offspring_count, group_count):
    """[The crossover process of the GA]

    Parameters
    ----------
    g_df : [DataFrame]
        [The DataFrame stores the structrual groups]
    m_df : [DataFrame]
        [The DataFrame stores the meteroids]
    p_df : [DataFrame]
        [The DataFrame stores the FCM_parameters]
    offspring_count : [int]
        [The count of next population]
    group_count : [int]
        [The count of structual group]

    Returns
    -------
    [DataFrame]
        [The DataFrame of structural groups, meteoroids, and FCM_parameters]
    """
    total = 0

    # create a dataframe to store structural_groups
    groups_frame = pd.DataFrame(columns=['mass_fraction', 'density',
                                         'strength', 'pieces',
                                         'cloud_mass_frac', 'strength_scaler',
                                         'fragment_mass_fractions'])
    
    # create a dataframe to store FCMmeteoroids
    # todo: cloud_mass_frac = 0
    meteoroids_frame = pd.DataFrame(columns=['velocity', 'angle',
                                             'density', 'radius',
                                             'strength', 'cloud_mass_frac'])
    # create a dataframe to store paramters
    param_frame = pd.DataFrame(columns=['ablation_coeff', 'cloud_disp_coeff',
                                        'strengh_scaling_disp',
                                        'fragment_mass_disp',
                                        'fitness_value'])

    # choose two parents to cross-over based on probability
    cs_prob = 0.8
    while total < offspring_count:
        temp_prob = ch.RA_uniform_float(1.0, 1, 0, 1)[0]
        p1, p2 = choose_parent(p_df)
        total += 2
        # the probability of cross-over is 80%
        if temp_prob < cs_prob:
            # swap groups
            for j in range(group_count):
                index_1 = p1 * group_count + j
                index_2 = p2 * group_count + j
                last_index = len(groups_frame)
                groups_frame.loc[last_index, ['strength_scaler',
                                              'strength',
                                              'density',
                                              'cloud_mass_frac']] = g_df.loc[index_2, ['strength_scaler',
                                                                                       'strength',
                                                                                       'density',
                                                                                       'cloud_mass_frac']]
                groups_frame.loc[last_index, ['mass_fraction',
                                              'pieces',
                                              'strength_scaler']] = g_df.loc[index_1, ['mass_fraction',
                                                                                       'pieces',
                                                                                       'strength_scaler']]
                groups_frame.at[last_index, 'fragment_mass_fractions'] = g_df.loc[index_1, 'fragment_mass_fractions']

            for j in range(group_count):
                index_1 = p1 * group_count + j
                index_2 = p2 * group_count + j
                last_index = len(groups_frame)
                groups_frame.loc[last_index, ['strength_scaler',
                                              'strength',
                                              'density',
                                              'cloud_mass_frac']] = g_df.loc[index_1, ['strength_scaler',
                                                                                              'strength',
                                                                                              'density',
                                                                                              'cloud_mass_frac']]
                groups_frame.loc[last_index, ['mass_fraction',
                                              'pieces',
                                              'strength_scaler']] = g_df.loc[index_2, ['mass_fraction',
                                                                                       'pieces',
                                                                                       'strength_scaler']]
                groups_frame.at[last_index, 'fragment_mass_fractions'] = g_df.loc[index_2, 'fragment_mass_fractions']
            
            last_index = len(meteoroids_frame)
            meteoroids_frame.loc[last_index, ['strength', 'density']] = m_df.loc[p2, ['strength', 'density']]
            meteoroids_frame.loc[last_index, ['velocity', 'angle', 'radius', 'cloud_mass_frac']] = m_df.loc[p1, ['velocity', 'angle', 'radius', 'cloud_mass_frac']]

            last_index = len(meteoroids_frame)
            meteoroids_frame.loc[last_index, ['strength', 'density']] = m_df.loc[p1, ['strength', 'density']]
            meteoroids_frame.loc[last_index, ['velocity', 'angle', 'radius', 'cloud_mass_frac']] = m_df.loc[p2, ['velocity', 'angle', 'radius', 'cloud_mass_frac']]
            
            last_index = len(param_frame)
            param_frame.loc[last_index, 'ablation_coeff'] = p_df.loc[p2, 'ablation_coeff']
            param_frame.loc[last_index, ['cloud_disp_coeff',
                                         'strengh_scaling_disp',
                                         'fragment_mass_disp']] = p_df.loc[p1, ['cloud_disp_coeff',
                                                                                'strengh_scaling_disp',
                                                                                'fragment_mass_disp']]
            last_index = len(param_frame)
            param_frame.loc[last_index, 'ablation_coeff'] = p_df.loc[p1, 'ablation_coeff']
            param_frame.loc[last_index, ['cloud_disp_coeff',
                                         'strengh_scaling_disp',
                                         'fragment_mass_disp']] = p_df.loc[p2, ['cloud_disp_coeff',
                                                                                'strengh_scaling_disp',
                                                                                'fragment_mass_disp']]                                                                                                                            
        else:
            for j in range(group_count):
                index_1 = p1 * group_count + j
                groups_frame.loc[len(groups_frame)] = g_df.loc[index_1]

            for j in range(group_count):
                index_2 = p2 * group_count + j
                groups_frame.loc[len(groups_frame)] = g_df.loc[index_2]

            meteoroids_frame.loc[len(meteoroids_frame)] = m_df.loc[p1]
            meteoroids_frame.loc[len(meteoroids_frame)] = m_df.loc[p2]
            param_frame.loc[len(param_frame)] = p_df.loc[p1]
            param_frame.loc[len(param_frame)] = p_df.loc[p2]

    return groups_frame, meteoroids_frame, param_frame


def discrete_fitness(pool):
    """[get the probability of each event from a accumulation probability]

    Parameters
    ----------
    pool : [Dataframe]
        [store the parameters and fitness]
    """
    pool_count = len(pool)


    for i in range(pool_count - 1, 0, -1):
        pool.loc[i, 'fitness_value'] = pool.loc[i, 'fitness_value'] - pool.loc[i-1, 'fitness_value']


def mutation(g_df, m_df, p_df, group_count):
    """[Mutation process in the GA]

    Parameters
    ----------
    g_df : [DataFrame]
        [The DataFrame to store the ]
    m_df : [type]
        [description]
    p_df : [type]
        [description]
    group_count : [type]
        [description]
    """
    # the probability of mutation is 3%
    mu_prob = 0.00
    for i in range(len(p_df)):
        temp_prob = ch.RA_uniform_float(1.0, 1, 0, 1)[0]

        if temp_prob > mu_prob:
            continue
        else:
            # mutate the ablation_coeff
            p_df.at[i, 'ablation_coeff'] = ch.RA_uniform_float(1.0, 1, 1e-9, 9e-8,
                                                               round='.9f')[0]


if __name__ == "__main__":
    group_count = 4
    event_count = 10
    offspring_count = 10
    observation = fit.read_event(fit.Event.kosice)
    veloctiy = 21.3
    angle = 81
    group_dataframe, meteoroids_frame, param_frame = mate_pool_generater(group_count,
                                                                         event_count,
                                                                         observation,
                                                                         velocity=veloctiy,
                                                                         angle=angle,
                                                                         strength_lower_bound=1,
                                                                         strength_higher_bound=5000)
    print(param_frame)
    iteration = 1

    # get the accumulate probability of events
    accumulate_probability(param_frame)

    # the index of highest fitness value
    highest = -1
    is_break = False
    fitness_list = np.zeros((iteration, offspring_count))

    for i in range(iteration):
        print("the iteration is ", i)
        # cross-over
        off_gd, off_m, off_p = cross_over(group_dataframe, meteoroids_frame, param_frame, offspring_count, group_count)

        # mutation
        mutation(off_gd, off_m, off_p, group_count)

        # update the fitness
        is_break, highest = update_fitness(off_gd, off_m, off_p, group_count, observation, fitness_list[i], 0.005)

        if is_break is True:
            print("Converged at " + str(i) + " iteration in advance")
            break

        # update the parent dataframe
        group_dataframe = off_gd
        meteoroids_frame = off_m
        param_frame = off_p

        # get the accumulate probability
        accumulate_probability(param_frame)

    discrete_fitness(param_frame)
    print(param_frame)

    if not is_break:
        highest = param_frame['fitness_value'].idxmax()
    else:
        print("The highest index is " + str(highest))

    groups_list = ch.compact_groups(group_dataframe, highest, group_count)
    # print the groups
    for i in range(group_count):
        print(group_dataframe.loc[i + highest * group_count])

    # meteoroid
    me = meteoroids_frame.loc[highest]
    meteroid_params = fcm.FCMmeteoroid(me['velocity'], me['angle'],
                                        me['density'], me['radius'],
                                        me['strength'], me['cloud_mass_frac'],
                                        groups_list)
    # parameters
    param = param_frame.loc[highest]
    params = fcm.FCMparameters(9.81, 6371, atmosphere,
                                ablation_coeff=param['ablation_coeff'],
                                cloud_disp_coeff=1,
                                strengh_scaling_disp=0,
                                lift_coeff=0, drag_coeff=1,
                                fragment_mass_disp=0, precision=1e-2)
    # simulate
    simudata = fcm.simulate_impact(params, meteroid_params, 100,
                                   craters=False, dedz=True, final_states=True)
    
    print(me)
    print(param)
    print(param_frame.loc[highest, 'ablation_coeff'])
    
    observation.set_index("altitude [km]", inplace=True)
    t.plot_simulation(simudata.energy_deposition, observation)
