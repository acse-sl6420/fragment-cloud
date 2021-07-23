# This python file is to find how to design the data structures of a chromosome
import fcm
import fcm.atmosphere as atm
import numpy as np
import matplotlib.pyplot as plt
from numpy import random


overall_seed = 32


def random_fraction(count, summation, is_even=False):
    """[Random generate several float numbers which sum up to a given number]

    Parameters
    ----------
    count : [type]
        [description]
    summation : [type]
        [description]
    is_even : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    # np.random.seed(overall_seed)
    if not is_even:
        # random generate several numbers
        random_num = np.random.rand(count)
        # get the ratio of each number
        ratio = summation / sum(random_num)
        result = random_num * ratio
    else:
        result = [summation/count for i in range(count)]
        
    return result


def RA_float(amplifier, count, lower_bound=0.1, high_bound=1.0,
             round='.2f', seed=overall_seed):
    """[Randomly generate float numbers]

    Parameters
    ----------
    amplifier : [type]
        [description]
    lower_bound : float, optional
        [description], by default 0.1
    high_bound : float, optional
        [description], by default 1.0
    count : [type], optional
        [description], by default group_count

    Returns
    -------
    [type]
        [description]
    """
    # format function returns string, so need float() to transform it to float.
    # random.seed(seed)
    result = [float(format(amplifier * np.random.uniform(
              lower_bound, high_bound), round)) for i in range(count)]
    return result


def RA_int(count, amplifier=1, lower_bound=2, high_bound=16, seed=overall_seed):
    """[Randomly generate int numbers]

    Parameters
    ----------
    count : [type]
        [description]
    amplifier : int, optional
        [description], by default 1
    lower_bound : int, optional
        [description], by default 2
    high_bound : int, optional
        [description], by default 16
    seed : [type], optional
        [description], by default overall_seed

    Returns
    -------
    [type]
        [description]
    """
    # random.seed(overall_seed)
    result = [int(amplifier * random.randint(lower_bound, high_bound))
              for i in range(count)]

    return result


def RA_tuple(count, high_bound=4, is_even=True):
    """[Randomly generate tuples which summation of fragments is 1.0

    Parameters
    ----------
    count : [type]
        [description]
    high_bound : int, optional
        [description], by default 4
    is_even : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    # randomly generate the count of this tuple
    # here, the count means just generate one random int number
    count = RA_int(count=1, high_bound=high_bound)[0]
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


def create_structural_group(density, strength, group_count):
    """[Generate the structural groups]

    Parameters
    ----------
    density : [type]
        [description]
    min_strength : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    param_count = 6
    groups = np.zeros((group_count, param_count))
    fragment_mass_fractions = []

    # random generate the numbers
    groups[:, params_sg.mass_fraction] = random_fraction(group_count, 1)

    # the density could be a random number
    groups[:, params_sg.density] = density

    groups[:, params_sg.strength] = RA_float(strength, group_count, 1.1, 2.0)

    groups[:, params_sg.pieces] = RA_int(group_count)

    groups[:, params_sg.cloud_mass_frac] = RA_float(1, group_count, 0.1, 1)

    groups[:, params_sg.strength_scaler] = RA_float(1, group_count, 0.1, 1)

    fragment_mass_fractions = [RA_tuple(1) for i in range(group_count)]

    structural_groups = []
    for i in range(group_count):
        structural_groups.append(fcm.StructuralGroup(
                                groups[i, params_sg.mass_fraction],
                                groups[i, params_sg.density],
                                groups[i, params_sg.strength],
                                int(groups[i, params_sg.pieces]),
                                groups[i, params_sg.cloud_mass_frac],
                                groups[i, params_sg.strength_scaler],
                                fragment_mass_fractions[i]))
    return structural_groups


def create_FCMmeteoroid(velocity, angle, density, radius, strength,
                        cloud_mass_frac, groups, ra_velocity=False,
                        ra_angle=False, ra_density=False, ra_radius=False,
                        ra_strength=False, ra_cloud_mass_frac=False,
                        ):
    count = 1
    # randomly generated velocity
    if ra_velocity:
        velocity = RA_float(1.0, count, 0.9, 1.1)[0] * velocity
    if ra_angle:
        angle = RA_float(1.0, count, 0.9, 1.1)[0] * angle
    if ra_density:
        density = RA_float(1.0, count, 0.9, 1.1)[0] * density
    if ra_radius:
        radius = RA_float(1.0, count, 0.9, 1.1)[0] * radius
    if ra_strength:
        strength = RA_float(1.0, count, 0.9, 1.1)[0] * strength
    if ra_cloud_mass_frac:
        cloud_mass_frac = RA_float(1.0, count, 0.9, 1.1)[0] * cloud_mass_frac

    return fcm.FCMmeteoroid(velocity, angle, density,
                            radius, strength,
                            cloud_mass_frac,
                            structural_groups=groups)

def create_FCMparameters(atmosphere, precision, ablation_coeff,
                         cloud_disp_coeff, strengh_scaling_disp,
                         fragment_mass_disp,
                         RA_ablation=False,
                         RA_cloud_disp_coeff=False,
                         RA_strengh_scaling_disp=False,
                         RA_fragment_mass_disp=False):
    """[Generate the FCMparameters]

    Parameters
    ----------
    atmosphere : [type]
        [description]
    precision : [type]
        [description]
    ablation_coeff : [type]
        [description]
    cloud_disp_coeff : [type]
        [description]
    """
    count = 1
    if RA_ablation:
        ablation_coeff = RA_float(1.0, count, 0.9, 1.1)[0] * ablation_coeff
    if RA_cloud_disp_coeff:
        cloud_disp_coeff = RA_float(1.0, count, 0.9, 1.1)[0] * cloud_disp_coeff
    if RA_strengh_scaling_disp:
        strengh_scaling_disp = RA_float(1.0, count, 0.9,
                                        1.1)[0] * strengh_scaling_disp
    if RA_fragment_mass_disp:
        RA_fragment_mass_disp = RA_float(1.0, count, 0.9,
                                         1.1)[0] * RA_fragment_mass_disp

    parameters = fcm.FCMparameters(9.81, 6371, atmosphere,
                                   ablation_coeff=ablation_coeff,
                                   cloud_disp_coeff=cloud_disp_coeff,
                                   strengh_scaling_disp=strengh_scaling_disp,
                                   fragment_mass_disp=fragment_mass_disp,
                                   precision=precision)
    return parameters


def plot_simulation(dEdz):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.plot(dEdz.to_numpy(), dEdz.index.to_numpy(), label='fcm')
    
    plt.xlabel("dEdz [kt TNT / km]")
    plt.ylabel("altitude [km]")
    plt.xscale('log')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig(filename)

    return fig


########## DEBUG PART ###############
def print_structural_groups(groups, group_count):
    for i in range(group_count):
        print("############# " + str(i) + " ################")
        print("mass_fraction: ", groups[i].mass_fraction)
        print("density: ", groups[i].density)
        print("strength: ", groups[i].strength)
        print("pieces: ", groups[i].pieces)
        print("cloud_mass_frac: ", groups[i].cloud_mass_frac)
        print("strength_scaler: ", groups[i].strength_scaler)
        print("fragment_mass_fractions: ", groups[i].fragment_mass_fractions)


def print_meteoroid(meteoroid):
    print("############ meteoroid's information is: ##############")
    print("meteoroid.velocity: ", meteoroid.velocity)
    print("meteoroid.angle: ", meteoroid.angle)
    print("meteoroid.density:", meteoroid.density)
    print("meteoroid.radius: ", meteoroid.radius)
    print("meteoroid.strength: ", meteoroid.strength)
    print("meteoroid.cloud_mass_frac: ", meteoroid.cloud_mass_frac)


def print_FCMparameters(parameters):
    print("precision: ", parameters.precision)
    print(parameters.precision)


if __name__ == "__main__":
    group_count = 1
    atmosphere = atm.US_standard_atmosphere()
    # # randomly generated the structural groups
    # structural_groups = create_structural_group(2.5e3, 0.5, group_count)
    # print_structural_groups(structural_groups, group_count)

    # # set a constant generate meteoroid
    # meteoroid = create_FCMmeteoroid(15.8, 17.8, 1.64e3, 4.5/2, 0.5, 0,
    #                                 structural_groups)
    # print_meteoroid(meteoroid)

    # # create FCMparamters, just randomly generate ablation_coeff
    # paramters = create_FCMparameters(atmosphere, precision=1e-4,
    #                                  ablation_coeff=1e-8,
    #                                  cloud_disp_coeff=2/3.5,
    #                                  strengh_scaling_disp=0,
    #                                  fragment_mass_disp=0)

    # # simulate
    # result = fcm.simulate_impact(paramters, meteoroid, 100,
    #                              craters=False, dedz=True, final_states=True)


    # plot_simulation(result.energy_deposition)

    event_list = [[]]
    for i in range(1):
        structural_groups = create_structural_group(2.5e3, 0.5, group_count)
        print_structural_groups(structural_groups, group_count)
        meteoroid = create_FCMmeteoroid(15.8, 17.8, 1.64e3, 4.5/2, 0.5, 0,
                                        structural_groups)
        parameters = create_FCMparameters(atmosphere, precision=1e-4,
                                          ablation_coeff=1e-8,
                                          cloud_disp_coeff=2/3.5,
                                          strengh_scaling_disp=0,
                                          fragment_mass_disp=0)
        event_list.append = [[structural_groups]]

        result = fcm.simulate_impact(parameters, meteoroid, 100,
                                     craters=False, dedz=True, final_states=True)
        # plot_simulation(result.energy_deposition)
