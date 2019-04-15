"""
demo.py

Demonstrates how 

"""

import tap_annealer as tap

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

# read csv, ignores first line.
raw = pd.read_csv("points.csv")
HouseHolds = np.array(raw)


'''
Use this function to set the defaults normally entered at the command line
'''
tap.set_defaults(cool_time=2500, tap_capacity=1000,
                 max_dist=-1, clean_steps=750)

'''
Optimise the taps for the houses contained in HH, each row is a house with 
first and second column coordinates and the third column the weighting.
'''
taps = tap.optimise(HH)

plt.show()

print('hi')
