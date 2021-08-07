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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import tools as t
import random

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
    # todo: cloud_mass_frac = 0
    meteoroids_frame = pd.DataFrame(columns=['velocity', 'angle',
                                             'density', 'radius',
                                             'strength', 'cloud_mass_frac'])
    # create a dataframe to store paramters
    param_frame = pd.DataFrame(columns=['ablation_coeff', 'cloud_disp_coeff',
                                        'strengh_scaling_disp',
                                        'fragment_mass_disp',
                                        'fitness_value'])

    # ############# define the characteristics of meteoroids ################
    

    # ######### the observed tets #########
    observation = fit.read_event(fit.Event.benesov)
    # make observation round to 2 decimal places
    observation = observation.round({'altitude [km]': 2})
    
    total_energy = t._total_energy(observation)
    # generate the events
    for i in range(event_count):
        # log uniform distribution
        density = ch.RA_logdis_float(1500, 5000)
        strength = ch.RA_logdis_float(1, 10000)

        # generate structural groups
        ch.groups_generater(groups_frame, density, strength, group_count)
        ch.meteroid_generater(meteoroids_frame, 21.3, 81, density,
                              strength, cloud_mass_frac=0,
                              total_energy=total_energy,
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
                                   cloud_disp_coeff=1,
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


def accumulate_probability(pool):
    """[get the accumulative probability of each event]

    Parameters
    ----------
    pool : [Dataframe]
        [store the parameters and fitness]
    """
    pool_count = len(pool)

    # get the highest fitness in pool
    parents_index = []
    parents_index.append(pool['fitness_value'].idxmax())
    
    # get the accumulative probability
    for i in range(pool_count-1):
        pool.loc[i+1, 'fitness_value'] += pool.loc[i, 'fitness_value']
    

def selection(parent_count, pool):
    """[select some events from event pool according to the fitness value]

    Parameters
    ----------
    parent_count : [int]
        [should be lower than event_count]
    pool : [DataFrame]
        [the last column of it is the information of fitness]

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


def update_fitness(group_dataframe, meteoroids_frame, param_frame, index):
    """[update the fitness after cross-over or mutation]

    Parameters
    ----------
    group_dataframe : [type]
        [description]
    meteoroids_frame : [type]
        [description]
    param_frame : [type]
        [description]
    index : [type]
        [description]
    """
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
    
    


def cross_over(g_df, m_df, p_df,
               parents_index, g_count):
    """[according to the probability to cross over parents]

    Parameters
    ----------
    group_dataframe : [type]
        [description]
    meteoroids_frame : [type]
        [description]
    param_frame : [type]
        [description]
    parents_index : [type]
        [description]
    """
    prob = 1.0
    parents_count = len(parents_index)

    for i in range(int(parents_count / 2)):
        # generate a random float number between 0 to 1
        temp_prob = ch.RA_uniform_float(1.0, 1, 0, 1)[0]

        # according to the probability
        if temp_prob < prob:
            # randomly choosed two parents cross-over
            slice = random.sample(parents_index, 2)
            p1 = slice[0]
            p2 = slice[1]

            # in terms of groups, cross-over strength_scaler
            for j in range(g_count):
                index_1 = p1 * g_count + j
                index_2 = p2 * g_count + j
                
                # exchange
                g_df.at[index_1, 'strength_scaler'], g_df.at[index_2, 'strength_scaler'] = g_df.at[index_2, 'strength_scaler'], g_df.at[index_1, 'strength_scaler']
                g_df.at[index_1, 'strength'], g_df.at[index_2, 'strength'] = g_df.at[index_2, 'strength'], g_df.at[index_1, 'strength']
                # g_df.at[index_1, 'pieces'], g_df.at[index_2, 'pieces'] = g_df.at[index_2, 'pieces'], g_df.at[index_1, 'pieces']
                g_df.at[index_1, 'density'], g_df.at[index_2, 'density'] = g_df.at[index_2, 'density'], g_df.at[index_1, 'density']
                g_df.at[index_1, 'cloud_mass_frac'], g_df.at[index_2, 'cloud_mass_frac'] = g_df.at[index_2, 'cloud_mass_frac'], g_df.at[index_1, 'cloud_mass_frac']
                # g_df.at[index_1, 'fragment_mass_fractions'], g_df.at[index_2, 'fragment_mass_fractions'] = g_df.at[index_2, 'fragment_mass_fractions'], g_df.at[index_1, 'fragment_mass_fractions']


            # # in terms of meteoroid, cross-over radius and strength
            m_df.at[p1, 'strength'], m_df.at[p2, 'strength'] = m_df.at[p2, 'strength'], m_df.at[p1, 'strength']
            m_df.at[p1, 'density'], m_df.at[p2, 'density'] = m_df.at[p2, 'density'], m_df.at[p1, 'density']

            p_df.at[p1, 'ablation_coeff'], p_df.at[p2, 'ablation_coeff'] = p_df.at[p2, 'ablation_coeff'], p_df.at[p1, 'ablation_coeff']

    # update the fitness value


def choose_parent(p_df):
    """[choose parents]

    Parameters
    ----------
    p_df : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # traverse the list to find the parent
    prob1 = ch.RA_uniform_float(1.0, 1, 0, 1)
    prob2 = ch.RA_uniform_float(1.0, 1, 0, 1)

    parent_1 = 0
    parent_2 = 0
    for i in range(len(p_df) - 1):
        if prob1 >= p_df.loc[i, 'fitness_value'] and prob1 < p_df.loc[i+1, 'fitness_value']:
            parent_1 = i
        if prob2 >= p_df.loc[i, 'fitness_value'] and prob2 < p_df.loc[i+1, 'fitness_value']:
            parent_2 = i

    return parent_1, parent_2
        


def cross_over_test(g_df, m_df, p_df, offspring_count, group_count):
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
            for j in range(group_count):
                index_1 = p1 * group_count + j
                index_2 = p2 * group_count + j
                
                # exchange
                g_df.at[index_1, 'strength_scaler'], g_df.at[index_2, 'strength_scaler'] = g_df.at[index_2, 'strength_scaler'], g_df.at[index_1, 'strength_scaler']
                g_df.at[index_1, 'strength'], g_df.at[index_2, 'strength'] = g_df.at[index_2, 'strength'], g_df.at[index_1, 'strength']
                g_df.at[index_1, 'density'], g_df.at[index_2, 'density'] = g_df.at[index_2, 'density'], g_df.at[index_1, 'density']
                g_df.at[index_1, 'cloud_mass_frac'], g_df.at[index_2, 'cloud_mass_frac'] = g_df.at[index_2, 'cloud_mass_frac'], g_df.at[index_1, 'cloud_mass_frac']

                groups_frame.loc[index_1] = g_df.loc[index_1]
                groups_frame.loc[index_2] = g_df.loc[index_2]

            m_df.at[p1, 'strength'], m_df.at[p2, 'strength'] = m_df.at[p2, 'strength'], m_df.at[p1, 'strength']
            m_df.at[p1, 'density'], m_df.at[p2, 'density'] = m_df.at[p2, 'density'], m_df.at[p1, 'density']
            meteoroids_frame.loc[p1] = m_df.loc[p1]
            meteoroids_frame.loc[p2] = m_df.loc[p2]

            p_df.at[p1, 'ablation_coeff'], p_df.at[p2, 'ablation_coeff'] = p_df.at[p2, 'ablation_coeff'], p_df.at[p1, 'ablation_coeff']
            param_frame.loc[p1] = p_df.loc[p1]
            param_frame.loc[p2] = p_df.loc[p2]
        else:
            for j in range(group_count):
                index_1 = p1 * group_count + j
                index_2 = p2 * group_count + j
                groups_frame.loc[index_1] = g_df.loc[index_1]
                groups_frame.loc[index_2] = g_df.loc[index_2]

            meteoroids_frame.loc[p1] = m_df.loc[p1]
            meteoroids_frame.loc[p2] = m_df.loc[p2]
            param_frame.loc[p1] = p_df.loc[p1]
            param_frame.loc[p2] = p_df.loc[p2]

    groups_frame.reset_index(drop=True)
    meteoroids_frame.reset_index(drop=True)
    param_frame.reset_index(drop=True)
    return groups_frame, meteoroids_frame, param_frame


def mutation(g_df, m_df, p_df, group_count):
    # the probability of mutation is 3%
    mu_prob = 0.03
    for i in range(len(p_df)):
        temp_prob = ch.RA_uniform_float(1.0, 1, 0, 1)[0]

        if temp_prob > mu_prob:
            continue
        else:
            # just mutate the ablation_coeff
            p_df.at[i, 'ablation_coeff'] = ch.RA_uniform_float(1.0, 1, 1e-9, 9e-8,
                                                               round='.9f')[0]

if __name__ == "__main__":
    group_count = 2
    event_count = 10
    offspring_count = 4
    group_dataframe, meteoroids_frame, param_frame = mate_pool_generater(group_count, event_count=event_count)
    iteration = 1

    # print(group_dataframe)
    # print(meteoroids_frame)
    # print(param_frame)
    # get the accumulate probability of events
    accumulate_probability(param_frame)

    for i in range(iteration):
        # cross-over
        # cross_over(group_dataframe, meteoroids_frame, param_frame, parents_index, group_count)
        off_gd, off_m, off_p = cross_over_test(group_dataframe, meteoroids_frame, param_frame, offspring_count, group_count)

        # mutation
        mutation(off_gd, off_m, off_p, group_count)

        # update the fitness
        
    
    # mutation

    # parents_index = selection(2, param_frame)
    
    # the parents chromosomes
    # p_meteoroid_df = meteoroids_frame.loc[parents_index]
    # p_param_df = param_frame.loc[parents_index]
    # p_group_df = pd.DataFrame(columns=['mass_fraction', 'density',
    #                                    'strength', 'pieces',
    #                                    'cloud_mass_frac', 'strength_scaler',
    #                                    'fragment_mass_fractions'])
    
    # for i in parents_index:
    #     for j in range(group_count):
    #         p_group_df.loc[i * group_count + j] = group_dataframe.loc[i * group_count + j]

    # print(p_group_df)
    # print(p_meteoroid_df)
    # print(p_param_df)

    # # cross-over
    # cross_over(p_group_df, p_meteoroid_df, p_param_df, parents_index,
    #            group_count)
