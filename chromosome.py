# This is the file create the chromosome
from math import radians
import numpy as np
import fcm
import pandas as pd
import random
from enum import Enum
import fcm.atmosphere as atm
import matplotlib.pyplot as plt


class fcm_param_loader:
    def __init__(self, bulk_density_range, strength_range,
                 diameter_range, cloud_frac_range,
                 number_of_frac_range, strengh_scale_range):
        """[]

        Parameters
        ----------
        bulk_density_range : [type]
            [description]
        strength_range : [type]
            [description]
        diameter_range : [type]
            [description]
        cloud_frac_range : [type]
            [description]
        number_of_frac_range : [type]
            [description]
        strengh_scale_range : [type]
            [description]
        """
        # get the standard deviation of the input range
        self.density_std = np.std(bulk_density_range)
        self.strength_std = np.std(strength_range)
        self.diameter_std = np.std(diameter_range)
        self.cloud_frac_std = np.std(cloud_frac_range)
        self.number_of_frac_std = np.std(number_of_frac_range)
        self.strength_scale_std = np.std(strengh_scale_range)

        # get the mean of the input range
        self.density_mean = np.mean(bulk_density_range)
        self.strength_mean = np.mean(strength_range)
        self.diameter_mean = np.mean(diameter_range)
        self.cloud_frac_mean = np.mean(cloud_frac_range)
        self.number_of_frac_mean = np.mean(number_of_frac_range)
        self.strength_scale_mean = np.mean(strengh_scale_range)


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


def RA_logdis_float(low, high, higher_bound):
    """[using log uniform distribution to generate a random number]

    Parameters
    ----------
    mean : [type]
        [description]
    higher_bound : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # result = np.random.lognormal(np.log(mean), np.log(std))
    result = 0

    # generate one random number which less than higher bound
    while (True):
        result = np.random.lognormal(np.log(mean), np.log(std))

        if (result < higher_bound):
            return result

def RA_uniform_float(amplifier, count, lower_bound=0.1, high_bound=1.0,
                     round='.2f'):
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


# the index of parameters of structural group
class sgp(Enum):
    mass_fraction = 'mass_fraction'
    density = 'density'
    strength = 'strength'
    pieces = 'pieces'
    cloud_mass_frac = 'cloud_mass_frac'
    strength_scaler = 'strength_scaler'
    fragment_mass_fractions = 'fragment_mass_fraction'


def RA_int(count, amplifier=1, lower_bound=2, high_bound=16):
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
    result = [amplifier * random.randint(lower_bound, high_bound)
              for i in range(count)]

    return result


def even_fragment(count):
    """[generate evenly fragments fraction]

    Parameters
    ----------
    count : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    result = [1/count for i in range(count)]
    return result


def groups_generater(groups, density, strength, group_count, cloud_frac,
                     fcm_param_loader):
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

    temp = np.zeros((group_count, param_count))

    # random generate the numbers
    temp[:, 0] = random_fraction(group_count, 1)

    # the density could be a random number
    temp[:, 1] = density

    # log distribution
    temp[:, 2] = strength

    # firstly generate just one pieces
    temp[:, 3] = RA_int(count=group_count)[0]

    # assumes that there is just one group, thus this parameter
    # is same as the cloud fraction in FCMmeteoroid
    # here I will keep the random generating interface
    # temp[:, 4] = RA_uniform_float(1.0, 1, 0.1, 0.9)[0]
    temp[:, 4] = [cloud_frac for i in range(group_count)]

    temp[:, 5] = RA_uniform_float(1, group_count, 0.1, 1)

    fragment_mass_fractions = [even_fragment(int(temp[i][3]))
                               for i in range(group_count)]

    # transform numpy array to list
    temp = temp.tolist()
    for i in range(group_count):
        groups.loc[len(groups)] = temp[i]
    # one structural group has group_count sub-fracture
    groups.insert(groups.shape[1], 'fragment_mass_fractions',
                  fragment_mass_fractions)


def meteroid_generater(meteroids, mean_std, velocity, angle, density,
                       radius, strength, cloud_mass_frac, ra_velocity=False,
                       ra_angle=False, ra_density=False, ra_radius=False,
                       ra_strength=False, ra_cloud_mass_frac=False,
                       ):
    count = 1

    parameters = [velocity, angle, density, radius, strength, cloud_mass_frac]

    # randomly generated velocity
    if ra_velocity:
        velocity = RA_uniform_float(velocity, count, 0.9, 1.1)[0]
    if ra_angle:
        angle = RA_uniform_float(angle, count, 0.9, 1.1)[0]
    if ra_density:
        density = RA_uniform_float(density, count, 0.9, 1.1)[0]
    # random
    if ra_radius:
        radius = RA_logdis_float(mean_std.radius_mean, mean_std.radius_std, 50)
    if ra_strength:
        strength = RA_uniform_float(strength, count, 0.9, 1.1)[0]
    if ra_cloud_mass_frac:
        cloud_mass_frac = RA_uniform_float(cloud_mass_frac, count, 0.9, 1.1)[0]

    meteroids.loc[len(meteroids)] = parameters

# ####### TODO: FCMmeteoroid #############
def FCMparameters_generater(parameters, cloud_disp_coeff,
                            strengh_scaling_disp, fragment_mass_disp,
                            ablation_coeff, RA_ablation=False,
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
    temp = []
    if RA_ablation:
        ablation_coeff = RA_uniform_float(1.0, count, 0.9, 1.1)[0] * ablation_coeff
    if RA_cloud_disp_coeff:
        cloud_disp_coeff = RA_uniform_float(1.0, count, 0.9, 1.1)[0] * cloud_disp_coeff
    if RA_strengh_scaling_disp:
        strengh_scaling_disp = RA_uniform_float(1.0, count, 0.9,
                                        1.1)[0] * strengh_scaling_disp
    if RA_fragment_mass_disp:
        RA_fragment_mass_disp = RA_uniform_float(1.0, count, 0.9,
                                         1.1)[0] * RA_fragment_mass_disp
    
    temp = [ablation_coeff, cloud_disp_coeff, strengh_scaling_disp, RA_fragment_mass_disp]

    parameters.loc[len(parameters)] = temp

    return parameters


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

if __name__ == "__main__":
    # define the range of some parameters which maybe
    # log distribution
    bulk_density_range = np.arange(1.5, 5, 0.1)
    strength_range = np.arange(1, 10000, 1)
    diameter_range = np.arange(0.1, 100, 0.1)
    cloud_frac_range = np.arange(0.1, 0.9, 0.1)
    number_of_frac_range = np.arange(2, 16, 1)
    strengh_scale_range = np.arange(0.1, 0.9, 0.1)

    # create dataframe to store structural groups
    groups = pd.DataFrame(columns=['mass_fraction', 'density',
                                   'strength', 'pieces',
                                   'cloud_mass_frac', 'strength_scaler'])
    

    # get the mean and std of parameters
    mean_std = fcm_param_loader(bulk_density_range, strength_range,
                                diameter_range, cloud_frac_range,
                                number_of_frac_range,
                                strengh_scale_range)

    # ############## generate structural groups ####################
    # uniform distribution
    density = RA_logdis_float(mean_std.density_mean, mean_std.density_std, bulk_density_range[-1])

    # log uniform distribution
    strength = RA_logdis_float(mean_std.strength_mean, mean_std.strength_std, strength_range[-1])
    cloud_frac = RA_uniform_float(1.0, 1, 0.1, 0.9)[0]

    # genarate structural groups
    structural_group_count = 2
    groups_generater(groups, density, strength, structural_group_count,
                     cloud_frac, mean_std)

    # ################### generate meteroid ########################
    # radius is log distribution
    meteoroids = pd.DataFrame(columns=['velocity', 'angle',
                                       'density', 'radius',
                                       'strength', 'cloud_mass_frac'])

    
    radius = RA_logdis_float(mean_std.diameter_mean, mean_std.diameter_std,
                             diameter_range[-1]) / 2

    # the cloud_frac is same as structural groups
    meteroid_generater(meteoroids, mean_std, 21.3, 81, density, radius,
                       strength, cloud_frac)
    
    # ################### FCMparameters ###########
    parameters = pd.DataFrame(columns=['ablation_coeff', 'cloud_disp_coeff',
                                       'strengh_scaling_disp', 'fragment_mass_disp'])
    
    FCMparameters = FCMparameters_generater(parameters, ablation_coeff=1e-8,
                                          cloud_disp_coeff=2/3.5,
                                          strengh_scaling_disp=0,
                                          fragment_mass_disp=0,
                                          RA_ablation=True)

    # simulate
    # create the structural groups
    groups_list = []
    for i in range(structural_group_count):
        temp = groups.loc[i]
        groups_list.append(fcm.StructuralGroup(mass_fraction=temp['mass_fraction'], 
                                        density=temp['density'],
                                        strength=temp['strength'],
                                        pieces=int(temp['pieces']),
                                        cloud_mass_frac=temp['cloud_mass_frac'],
                                        strength_scaler=temp['strength_scaler'],
                                        fragment_mass_fractions=temp['fragment_mass_fractions']))
    
    test = meteoroids.loc[0]
    meteroid_params = fcm.FCMmeteoroid(test['velocity'], test['angle'],
                                       test['density'], test['radius'],
                                       test['strength'], test['cloud_mass_frac'],
                                       groups_list)
    
    param = parameters.loc[0]
    atmosphere = atm.US_standard_atmosphere()
    params = fcm.FCMparameters(9.81, 6371, atmosphere, ablation_coeff=2.72e-8,
                               cloud_disp_coeff=1, strengh_scaling_disp=0,
                               fragment_mass_disp=0, precision=1e-2)

    # simulate
    simudata = fcm.simulate_impact(params, meteroid_params, 100,
                                   craters=False, dedz=True, final_states=True)
    
    plot_simulation(simudata.energy_deposition)