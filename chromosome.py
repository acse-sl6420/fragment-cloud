# This is the file create the chromosome
from math import radians
import numpy as np
import fcm
from numpy.random import uniform
import pandas as pd
import random
from enum import Enum
import fcm.atmosphere as atm
import matplotlib.pyplot as plt
import tools as t


def random_fraction(count, summation, is_even=False):
    """[Random generate several float numbers which sum up to a given number]

    Parameters
    ----------
    count : [int]
        [The size of List]
    summation : [float]
        [The summation of all elements in List]
    is_even : bool, optional
        [False: the fraction is not equal;
         True: the fraction is equal], by default False

    Returns
    -------
    [List]
        [Get a list, the summary of the element is summation]
    """
    if not is_even:
        # random generate several numbers
        random_num = np.random.rand(count)
        # get the ratio of each number
        ratio = summation / sum(random_num)
        result = random_num * ratio
    else:
        result = [summation/count for i in range(count)]

    return result


def RA_logdis_float(low, high):
    """[Using log uniform distribution to generate a random number]

    Parameters
    ----------
    low : [float]
        [the lower boundry of the result]
    high : [float]
        [the higher boudry of the result]

    Returns
    -------
    [float]
        [One random numbers]
    """
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


def RA_uniform_float(amplifier, count, lower_bound=0.1, high_bound=1.0,
                     round='.2f'):
    """[Randomly generate float numbers]

    Parameters
    ----------
    amplifier : [float]
        [The result is amplifier * result]
    count : [int]
        [The count of result]
    lower_bound : float, optional
        [The lower boundry of the random number], by default 0.1
    high_bound : float, optional
        [The higher boundry of the random number], by default 1.0
    round : str, optional
        [Float number round to a format], by default '.2f'

    Returns
    -------
    [type]
        [Return a list, each element in it is between lower_bound and
         high_bound]
    """
    # format function returns string, so need float() to transform it to float.
    result = [float(format(amplifier * np.random.uniform(
              lower_bound, high_bound), round)) for i in range(count)]
    return result


def RA_int(count, amplifier=1, lower_bound=2, high_bound=16):
    """[Randomly generate int numbers]

    Parameters
    ----------
    count : [int]
        [The size of List]
    amplifier : int, optional
        [The element in the list = amplifier * random float], by default 1
    lower_bound : int, optional
        [The lower bound of element in List], by default 2
    high_bound : int, optional
        [The higher bound of element in List], by default 16

    Returns
    -------
    [List]
        [A list including several random int number]
    """
    result = [amplifier * random.randint(lower_bound, high_bound)
              for i in range(count)]

    return result


def even_fragment(count):
    """[generate evenly fragments fraction]

    Parameters
    ----------
    count : [int]
        [The count of the List]

    Returns
    -------
    [type]
        [A list, which element is 1/count]
    """
    result = [1/count for i in range(count)]
    return result


def groups_generater(groups, density, strength, group_count,
                     strength_higher_bound=10000):
    """[Generate the structural groups]

    Parameters
    ----------
    groups : [Dataframe]
        [The dataframe store the structural groups]
    density : [float]
        [The density of the meteoroid]
    strength : [float]
        [The strength of the meteoroid]
    group_count : [int]
        [the count of structural groups]
    strength_higher_bound : int, optional
        [The higher bound of the strength], by default 10000
    """
    param_count = 6
    temp = np.zeros((group_count, param_count))

    # random generate the numbers
    temp[:, 0] = random_fraction(group_count, 1)
    if (group_count == 2):
        temp[0:0] = RA_uniform_float(1.0, 1, 0.1, 0.9, round='.8f')
        temp[1:0] = 1.0 - temp[0:0]

    # density
    temp[:, 1] = density

    # strength
    if group_count == 1:
        temp[:, 2] = strength
    elif group_count == 2:
        # This parameters is just suitable for benesov
        temp[0, 2] = strength
        temp[1, 2] = RA_logdis_float(strength + 1000, strength_higher_bound)
    else:
        temp[:, 2] = [RA_logdis_float(strength, strength_higher_bound) for i in range(group_count)]

    # pieces
    temp[:, 3] = RA_int(count=group_count, high_bound=5)

    # temp[:, 4] = RA_uniform_float(1.0, 1, 0.1, 0.9)[0]
    # cloud mass fraction
    temp[:, 4] = [RA_uniform_float(1.0, 1, 0.1, 0.9)[0] for i in range(group_count)]

    # strength scaler
    temp[:, 5] = [RA_uniform_float(1.0, group_count, 0.1, 1.0, round=".8f")[0] for i in range(group_count)]

    fragment_mass_fractions = [even_fragment(int(temp[i][3]))
                               for i in range(group_count)]

    # transform numpy array to list
    temp = temp.tolist()

    for i in range(group_count):
        index = len(groups)
        groups.loc[index, ['mass_fraction', 'density',
                           'strength', 'pieces',
                           'cloud_mass_frac', 'strength_scaler']] = temp[i]
        groups.loc[index, 'fragment_mass_fractions'] = fragment_mass_fractions[i]


def meteroid_generater(meteroids, velocity, angle, density,
                       strength, cloud_mass_frac, total_energy,
                       ra_velocity=False,
                       ra_angle=False, ra_density=False, ra_radius=False,
                       ra_strength=False, ra_cloud_mass_frac=False,
                       ):
    """[Generate the parameters of meteroids]

    Parameters
    ----------
    meteroids : [DataFrame]
        [The DataFrame storing meteroids]
    velocity : [float]
        [The velocity of meteoroids]
    angle : [float]
        [The angle of meteoroids]
    density : [float]
        [The density of meteoroids]
    strength : [float]
        [The strength of meteoroids]
    cloud_mass_frac : [float]
        [The cloud_mass_frac of meteoroids]
    total_energy : [float]
        [The total_energy of meteoroids read from the observation]
    ra_velocity : bool, optional
        [Randomly generate velocity or not], by default False
    ra_angle : bool, optional
        [Randomly generate angle or not], by default False
    ra_density : bool, optional
        [Randomly generate density or not], by default False
    ra_radius : bool, optional
        [Randomly generate radius or not], by default False
    ra_strength : bool, optional
        [Randomly generate strength or not], by default False
    ra_cloud_mass_frac : bool, optional
        [Randomly generate cloud_mass_frac or not], by default False
    """
    count = 1

    # randomly generated velocity
    if ra_velocity:
        velocity = RA_uniform_float(velocity, count, 0.9, 1.1)[0]
    if ra_angle:
        angle = RA_uniform_float(angle, count, 0.9, 1.1)[0]
    if ra_density:
        density = RA_uniform_float(density, count, 0.9, 1.1)[0]   
    if ra_strength:
        strength = RA_uniform_float(strength, count, 0.9, 1.1)[0]
    if ra_cloud_mass_frac:
        cloud_mass_frac = RA_uniform_float(cloud_mass_frac, count, 0.9, 1.1)[0]

    # calculate radius
    radius = t._radius(total_energy, density, velocity)
    if ra_radius:
        # total_energy = RA_uniform_float(total_energy, count, 0.5, 1.1)[0]
        # radius = t._radius(total_energy, density, velocity)
        radius = RA_uniform_float(radius, count, 0.6, 1.1)[0]

    parameters = [velocity, angle, density, radius, strength, cloud_mass_frac]
    meteroids.loc[len(meteroids)] = parameters


def FCMparameters_generater(parameters, cloud_disp_coeff,
                            strengh_scaling_disp, fragment_mass_disp,
                            ablation_coeff, RA_ablation=False,
                            RA_cloud_disp_coeff=False,
                            RA_strengh_scaling_disp=False,
                            RA_fragment_mass_disp=False):
    """[Generate the FCMparameters]

    Parameters
    ----------
    parameters : [DataFrame]
        [The DataFrame storing the parameters of meteoroids]
    cloud_disp_coeff : [float]
        [The ablation coefficient of the FCM]
    strengh_scaling_disp : [float]
        [The strength scaling dispersion the FCM]
    fragment_mass_disp : [float]
        [The fragment mass dispersion of the FCM]
    ablation_coeff : [float]
        [The ablation coefficient of the FCM]
    RA_ablation : bool, optional
        [Randomly generate the ablation_coeff or not], by default False
    RA_cloud_disp_coeff : bool, optional
        [Randomly generate the cloud_disp_coeff or not], by default False
    RA_strengh_scaling_disp : bool, optional
        [Randomly generate the strengh_scaling_disp or not], by default False
    RA_fragment_mass_disp : bool, optional
        [Randomly generate the fragment_mass_disp or not], by default False

    Returns
    -------
    [DataFrame]
        [The DataFrame storing the information of Parameters chromosome]
    """

    count = 1
    if RA_ablation:
        ablation_coeff = RA_uniform_float(1.0, count, 1e-9, 9e-8,
                                          round='.9f')[0]
    if RA_cloud_disp_coeff:
        cloud_disp_coeff = RA_uniform_float(cloud_disp_coeff,
                                            count, 0.9, 1.1)[0]
    if RA_strengh_scaling_disp:
        strengh_scaling_disp = RA_uniform_float(strengh_scaling_disp,
                                                count, 0.9, 1.1)[0]
    if RA_fragment_mass_disp:
        fragment_mass_disp = RA_uniform_float(RA_fragment_mass_disp,
                                              count, 0.9, 1.1)[0]
    
    # attention: the intial fitness value is 0
    temp = [ablation_coeff, cloud_disp_coeff, strengh_scaling_disp,
            fragment_mass_disp, 0]

    parameters.loc[len(parameters)] = temp

    return parameters


def compact_groups(groups_dataframe, event_index, group_count):
    """[Compact the Structural_group chromosome to the list
        StructuralGroup in FCM library]

    Parameters
    ----------
    groups_dataframe : [DataFrame]
        [The DataFrame stores the information of the structural groups]
    event_index : [int]
        [The index of this event]
    group_count : [int]
        [The count of structural group]

    Returns
    -------
    [List]
        [Return a list of StructuralGroup]
    """
    groups_list = []
    for i in range(group_count):
        temp = groups_dataframe.loc[i + event_index * group_count]
        groups_list.append(fcm.StructuralGroup(mass_fraction=temp['mass_fraction'], 
                           density=temp['density'],
                           strength=temp['strength'],
                           pieces=int(temp['pieces']),
                           cloud_mass_frac=temp['cloud_mass_frac'],
                           strength_scaler=temp['strength_scaler'],
                           fragment_mass_fractions=temp['fragment_mass_fractions']))

    return groups_list


if __name__ == "__main__":
    # create a dataframe to store paramters
    param_frame = pd.DataFrame(columns=['ablation_coeff', 'cloud_disp_coeff',
                                        'strengh_scaling_disp',
                                        'fragment_mass_disp',
                                        'fitness_value'])

    FCMparameters = FCMparameters_generater(param_frame, ablation_coeff=1e-8,
                                            cloud_disp_coeff=2/3.5,
                                            strengh_scaling_disp=0,
                                            fragment_mass_disp=0,
                                            RA_ablation=True)
    print(FCMparameters)
