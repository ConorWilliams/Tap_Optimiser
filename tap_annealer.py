'''
tap_annealer.py

Copyright (c) 2019 C. J. Williams

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
print('')
print("Tap placement annealing-optimiser.")
print('Copyright 2019 C. J. Williams (CHURCHILL COLLEGE)')
print('This is free software with ABSOLUTELY NO WARRANTY.')

print('''
Attempts to place the correct number of taps minimising the: distance to each
house; house water requirements; deviation in tap water use and taking account
of how much water a tap can produce. If no solution exists with the maximum 
tap-house distance below the required value the process is re-run with more taps. 

Each run saves a figure containing the optimum solution for a given number of 
taps as well as writing the positions to a .csv file.

Place the house data .csv file in same directory as source code. The first line 
of the file is ignored but must still have three coma-separated words

Program will let you know the maximum walking distance at end, so run with -1 
first to get an idea of what to set that to.

Press Enter to use default.
    ''')

'''
This program can be used either as a standalone or imported and run using the
optimise() and set_defaults() functions. See demo.py for example.
'''

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

DICT = {'yes': True, 'ye': True, 'y': True, 'n': False, 'no': False, 1: 'yes',
        0: 'no', 't': True, 'true': True, 'f': False, 'false': False}


def strtobool(str):
    return DICT[str]


UNDERVOLT = 0.1  # you can twiddle this to change some subtle behaviour :)
STEP = 1  # can reduce for finer stepping at adjustment stage
T0 = 0.1
CLEAN_STEPS = 500

COOL_TIME = 2500
MAX_DIST = -1
INTERMEDIATES = False
TAP_CAPACITY = 2000  # how much H2O each tap dishes out
FILE = 'example'

# initialise global variables
array = 0
taps = 0
dists = 0
number_of_taps = 0
number_of_houses = 0

if __name__ == "__main__":
    read = input('File name please (.csv auto added) with junk first line: ')
    if read == '':
        print('Using default:', FILE)
    else:
        FILE = str(read)
        print('Using choice:', FILE)

    read = input("Tap output please, default 2000: ")
    if read == '':
        print('Using default:', TAP_CAPACITY)
    else:
        TAP_CAPACITY = float(read)
        print('Using choice:', TAP_CAPACITY)

    read = input(
        'Max allowable walking distance to tap, use -1 (default) for infinity: ')
    if read == '':
        print('Using default:', MAX_DIST)
    else:
        MAX_DIST = float(read)
        print('Using choice:', MAX_DIST)

    read = input(
        'Set number of steps before stopping, higher = better results,  default 2500: ')
    if read == '':
        print('Using default:', COOL_TIME)
    else:
        COOL_TIME = int(read)
        print('Using choice:', COOL_TIME)

    read = input('Show extra info while running (y/n), default n: ')
    if read == '':
        print('Using default:', strtobool(INTERMEDIATES))
    else:
        INTERMEDIATES = strtobool(read)
        print('Using choice:', strtobool(INTERMEDIATES))

    # read csv, ignores first line
    raw = pd.read_csv(FILE + ".csv")
    HH = np.array(raw)


def set_defaults(cool_time=2500, max_dist=-1, verbose=False, tap_capacity=2000,
                 undervolt=0.1, clean_steps=500, step=1):
    '''
    Use this to set the defaults that would normally be set by typing into 
    the terminal. Can also set under-volt, controls max tap overuse
    '''
    global COOL_TIME, MAX_DIST, INTERMEDIATES, TAP_CAPACITY, UNDERVOLT
    global CLEAN_STEPS, STEP

    COOL_TIME = cool_time
    MAX_DIST = max_dist
    INTERMEDIATES = verbose
    TAP_CAPACITY = tap_capacity
    UNDERVOLT = undervolt
    CLEAN_STEPS = clean_steps
    STEP = step

    return 0


def update(i):
    '''
    update the distances for the i'th tap
    '''
    global array, taps, number_of_houses

    # add the dist to each tap to each house to the array
    d = array[::, 0:2] - taps[i, ::]
    d = d * d
    array[::, 3 + i] = np.sum(d, axis=1)


def score(want_dist=0):
    '''
    the scoring function
    '''
    global array, taps, dists, number_of_taps, number_of_houses, MAX_DIST

    capacity = np.ones(number_of_taps) * TAP_CAPACITY
    tap = np.arange(number_of_taps)

    # put the last col as the smallest distance
    array[::, -1] = np.amin(array[::, 3:-2], axis=1)

    # sort the array in place by the last col ^
    array = array[np.argsort(array[::, -1]), ::]

    # calculate the score
    score = 0
    for c, row in enumerate(array[::, 2:-1:1]):
        order = np.argsort(row[1:-1])  # sort the distances for each tap

        # detect if it cant fit in any tap
        maxi = np.max(np.subtract(capacity, row[0]))
        if maxi < -UNDERVOLT * TAP_CAPACITY:
            _k = 2
            print("A house is too big!")
        else:
            _k = 1

        # assign the closest tap with available water to the house
        for index in order:
            if capacity[index] - row[0] * _k > -UNDERVOLT * TAP_CAPACITY * _k:
                capacity[index] -= row[0]
                row[-1] = tap[index]

                mult = 1
                if capacity[index] < 0:
                    mult *= 2  # penalise overusing
                if row[index + 1] > MAX_DIST**2 and MAX_DIST > 0:
                    mult *= 50  # penalise too far
                if want_dist:
                    dists[c] = np.sqrt(row[index + 1])

                score += row[index + 1] * row[0] * _k * mult
                break

    # penalise unbalanced taps
    if number_of_taps > 1:
        score *= np.log(capacity.std() / TAP_CAPACITY + 1.01)

    return score, capacity


def get_cmap(n, name='hsv'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n)


def optimise(HH):
    '''
    Optimise the taps for the HouseHolds contained in HH, each row is a house 
    with first and second column coordinates and the third column the weighting.

    Returns an numpy array of taps, each each row a tap with first and second 
    column coordinates and the third column the percentage utilisation.

    If maximum distance is unfulfillable then more taps will be added until a 
    solution is found.
    '''
    global array, taps, dists, number_of_taps, number_of_houses

    # find the grid and centre
    maxLat = (np.amax(HH[:, 0]))
    minLat = (np.amin(HH[:, 0]))
    maxLong = (np.amax(HH[:, 1]))
    minLong = (np.amin(HH[:, 1]))
    magLat = maxLat - minLat
    magLong = maxLong - minLong

    if magLat > magLong:
        length = magLat
    else:
        length = magLong

    centre = np.array([minLat + 0.5 * length, minLong + 0.5 * length])

    # work out how many taps needed
    temp = np.sum(HH[::, 2])
    number_of_taps = math.ceil(temp / TAP_CAPACITY)
    number_of_houses = np.size(HH[:, 0])

    print('This area has', number_of_houses,
          'house-holds and needs', number_of_taps, 'tap(s).')

    worst = MAX_DIST + 1
    while worst > MAX_DIST:
        print('Trying with', number_of_taps, 'taps')

        # big array to hold all the data
        array = np.ones((number_of_houses, 3 + number_of_taps + 2))
        array[:, 0:3] = HH

        taps = np.ones((number_of_taps, 2))
        taps[::, 0] = centre[0]
        taps[::, 1] = centre[1]

        dists = np.zeros(number_of_houses)

        # to make score a dimensionless parameter
        kB = number_of_houses / number_of_taps * length**2

        #/*-------------------------------------------------------------------*/

        # main Monte-Carlo with annealing
        for i in range(number_of_taps):
            update(i)

        s0 = score()[0]

        s = s0
        T = T0
        for j in range(COOL_TIME):
            for i in range(number_of_taps):
                rand = 2 * np.random.rand(2) - 1
                rand *= length * 0.25 * T / T0 * np.random.random()
                taps[i, ::] += rand

                if np.abs(taps[i, ::] - centre)[0] > length / 2 or np.abs(taps[i, ::] - centre)[1] > length / 2:
                    taps[i, ::] -= rand
                    continue

                update(i)
                sp = score()[0]

                if sp <= s:
                    s = sp
                elif np.random.random() <= np.exp(-(sp - s) / (T * kB)):
                    s = sp
                else:
                    taps[i, ::] -= rand
                    update(i)

                if INTERMEDIATES:
                    print(str(i).zfill(4), COOL_TIME - j, s / s0 * 100)

            print(COOL_TIME - j, s / s0 * 100)
            T -= T0 / COOL_TIME

        #/*-------------------------------------------------------------------*/

        # clean up randomness with 500 pure steps
        s = score()[0]
        for j in range(CLEAN_STEPS):
            for i in range(number_of_taps):
                rand = 2 * np.random.rand(2) - 1
                rand *= length * np.random.random() * STEP
                taps[i, ::] += rand

                update(i)
                sp = score()[0]

                if sp <= s:
                    s = sp
                else:
                    taps[i, ::] -= rand
                    update(i)

            print('Adjusting', j, '/', CLEAN_STEPS)

        #/*-------------------------------------------------------------------*/

        for i in range(number_of_taps):
            update(i)

        capacity = score(1)[1]
        capacity = (TAP_CAPACITY - capacity) / TAP_CAPACITY * 100
        capacity = np.round(capacity)

        print('Tap usage percent:', capacity)
        for c, tap in enumerate(taps):
            print(tap, str(capacity[c]).zfill(5))

        worst = dists.max()

        print('Biggest walk is:', worst)

        #/*-------------------------------------------------------------------*/

        cmap = get_cmap(len(taps), 'nipy_spectral')

        plt.scatter(array[::, 1], array[::, 0],
                    c=array[::, -2], cmap=cmap, label='Houses', s=16)
        plt.plot(taps[:, 1], taps[:, 0], '+',
                 color='k', markersize=8, label='Taps')

        plt.title("Optimised for " + str(number_of_taps) + ' taps')
        plt.gca().set_aspect('equal')

        plt.axis(xmin=minLong, ymin=minLat, xmax=minLong +
                 length, ymax=minLat + length)

        plt.xlabel('Latitude')
        plt.ylabel('Longitude')

        plt.legend()
        plt.savefig(FILE + '_' + str(number_of_taps) + '_taps' + '.pdf')

        #/*-------------------------------------------------------------------*/

        with open(FILE + '_' + str(number_of_taps) + '_taps' + '.csv', 'w') as file:
            file.write("x, y, percent utilisation\n")
            for c, tap in enumerate(taps):
                file.write(str(tap[0]) + "," + str(tap[1]) +
                           "," + str(capacity[c]) + "\n")

        if worst < MAX_DIST:
            break
        elif MAX_DIST < 0:
            break
        else:
            plt.clf()

        number_of_taps += 1

    if __name__ == "__main__":
        plt.show()
    else:
        out = np.ones((number_of_taps, 3))
        out[::, 0:2:1] = taps
        out[::, 2] = capacity
        return out


if __name__ == "__main__":
    optimise(HH)
