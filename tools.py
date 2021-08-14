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

def _total_energy(observation):
    total_energy = abs(sum(observation['altitude [km]'].diff().to_numpy()[1:] *
                           observation['dEdz [kt TNT / km]'].to_numpy()[1:])) * 4.184e12
    
    return total_energy


def _radius(energy, density, velocity):
    """[calculate radius through velocity and total energy depostion.
        energy = 1/2 * m * v ^ 2
        m = (2/3)πr³ * density]

    Parameters
    ----------
    energy : [float]
        [J]
    density : [float]
        [kg/m3]
    velocity : [float]
        [km/s]
    """
    # change km/s to m/s
    velocity *= 1000
    mass = (energy * 2) / np.square(velocity)
    radius = ((mass * 3) / (2 * np.pi * density)) ** (1/3)
    return radius

def plot_simulation(dEdz, observation):
    mask = np.logical_and(dEdz.index.to_numpy() >= observation.index.min(),
                          dEdz.index.to_numpy() <= observation.index.max())
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.plot(dEdz.to_numpy()[mask], dEdz.index.to_numpy()[mask], label='fcm')

    plt.plot(observation['dEdz [kt TNT / km]'].to_numpy(), observation.index.to_numpy(),
            "--", label='observation')
    
    plt.xlabel("dEdz [kt TNT / km]")
    plt.ylabel("altitude [km]")
    plt.xscale('log')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig(filename)
    
    return fig