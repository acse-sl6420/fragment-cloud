# This is the file create the chromosome
import numpy as np
import fcm
import pandas as pd
import random
from enum import Enum


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


def RA_logdis_float(mean, std):
    """[using log distribution to generate a random number]

    Parameters
    ----------
    mean : [type]
        [description]
    std : [type]
        [description]
    """
    result = np.random.lognormal(np.log(mean), np.log(std))
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
    # random.seed(overall_seed)
    result = [amplifier * random.randint(lower_bound, high_bound)
              for i in range(count)]

    return result


def even_fragment(count):
    result = [1/count for i in range(count)]
    return result


def groups_generater(density, strength, group_count, cloud_frac,
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

    # create the dataframe
    groups = pd.DataFrame(columns=['mass_fraction', 'density',
                                   'strength', 'pieces',
                                   'cloud_mass_frac', 'strength_scaler'])
                                #    sgp.fragment_mass_fractions])
    

    temp = np.zeros((group_count, param_count))

    # random generate the numbers
    temp[:, 0] = random_fraction(group_count, 1)

    # the density could be a random number
    temp[:, 1] = density

    # log distribution
    temp[:, 2] = strength

    # firstly generate just one pieces
    temp[:, 3] = RA_int(count=group_count)

    # assumes that there is just one group, thus this parameter
    # is same as the cloud fraction in FCMmeteoroid
    # here I will keep the random generating interface
    # temp[:, 4] = RA_uniform_float(1.0, 1, 0.1, 0.9)[0]
    temp[:, 4] = [cloud_frac for i in range(group_count)]

    temp[:, 5] = RA_uniform_float(1, group_count, 0.1, 1)

    fragment_mass_fractions = [even_fragment(int(temp[i][3]))
                               for i in range(group_count)]

    temp = temp.tolist()
    for i in range(group_count):
        groups.loc[len(groups)] = temp[i]
    # one structural group has group_count sub-fracture
    groups.insert(groups.shape[1], 'fragment_mass_fraction',
                  fragment_mass_fractions)
    


if __name__ == "__main__":
    bulk_density_range = np.arange(1.5, 5, 0.1)
    strength_range = np.arange(1, 10000, 10)
    diameter_range = np.arange(0.1, 100, 0.1)
    cloud_frac_range = np.arange(0.1, 0.9, 0.1)
    number_of_frac_range = np.arange(2, 16, 1)
    strengh_scale_range = np.arange(0.1, 0.9, 0.1)

    # get the mean and std of parameters
    mean_std = fcm_param_loader(bulk_density_range, strength_range,
                                diameter_range, cloud_frac_range,
                                number_of_frac_range,
                                strengh_scale_range)

    # generate structural groups
    density = 3.44e3
    strength = RA_logdis_float(mean_std.density_mean, mean_std.density_std)
    cloud_frac = RA_uniform_float(1.0, 1, 0.1, 0.9)[0]
    structural_group_count = 2
    groups_generater(density, strength, structural_group_count,
                     cloud_frac, mean_std)

