# make_pattern5.py

import math
from penrose import PenroseP3, BtileL, psi
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# A star with five-fold symmetry

# The Golden ratio
phi = 1 / psi
scale = 46.9789153789/2.6180*100/1.000017*500/500.00013720 # for ngen = 6
# scale = scale / 36.47450844 * 250 / 374.99999997 * 375 # for ngen = 10

theta = 2*math.pi / 5
rot = math.cos(theta) + 1j*math.sin(theta)


# for calculating distance between two tiles
def calc_size(sc, sp):
    p_on_y = 1j * phi * sc
    p_rot = p_on_y / rot
    dis = abs(p_on_y - p_rot)/2
    sz = dis/(dis+sp)
    exp = (dis+sp)/dis
    return sz, exp


def make_penrose_limit(sc, sf, sz, limit):
    """
    Make a penrose pattern based on its scale, shift, and size.
    :param sc: scale
    :param sf: shift
    :param sz: size, i.e. percentage of original size, for example, 90%, 50%
    :param limit: radius limit of final drawing.
    :return:
    """
    B11 = sc
    p = B11 * rot
    q = p*rot

    C55 = -sc * phi
    r = C55 / rot
    s = r / rot
    A = [0]*5
    B = [sc, p, p, q, q]
    C = [s, s, r, r, C55]

    # you can tune the figure parameters here:
    config = {'tile-opacity': 0.9, 'stroke-colour': '#800',
              'Stile-colour': '#63ace5', 'Ltile-colour': '#fdf498',
              'rotate': math.pi / 2,
              'size': sz}
    tiling = PenroseP3(sc * 2, ngen=6, config=config)
    tiling.set_initial_tiles([BtileL(*v) for v in zip(A, B, C)])
    tiling.make_tiling()
    ele = []
    dis = []
    for e in tiling.elements:
        if isinstance(e, BtileL):
            D = e.A + e.C - e.B
            if abs(e.A) <= limit and abs(e.B) <= limit and abs(e.C) <= limit and abs(D) <= limit:
                cen = e.centre()
                dis.append(abs(cen) + 1 * math.atan2(cen.imag, cen.real) / 2 / math.pi)
                ele.append(shift_ele(e, sf))
    new_ele = [ele[x] for x in np.argsort(dis)]

    # draw the elements in svg
    tiling.elements = deepcopy(new_ele)
    return tiling, new_ele


def complex_to_vec(x):
    return x.real, x.imag, 0


def rotate_ele(e, theta2):
    """ Rotate the figure clockwise by theta radians."""
    rot2 = math.cos(theta2) + 1j * math.sin(theta2)
    e2 = deepcopy(e)
    e2.A = e.A/rot2
    e2.B = e.B/rot2
    e2.C = e.C/rot2
    return e2


def shift_ele(e, sf2):
    """ Shift the figure clockwise by sf2."""
    e2 = deepcopy(e)
    e2.A = e.A - sf2
    e2.B = e.B - sf2
    e2.C = e.C - sf2
    return e2


def one_pattern_limit(scale_size, gap_size, radius_limit):
    # control the percentage size to get ideal gap size
    new_scale = scale_size / 100 * scale
    size, expand = calc_size(scale_size, gap_size)
    scale_in_use = new_scale * expand
    tiling, ele = make_penrose_limit(scale_in_use, 0, size, radius_limit)
    return tiling, ele


def main(tile_size, gap_size, select_num):
    r_limit = 20000 # unit nm, default is the scale size, here we choose a 2 um length circle.
    tiling, elements = one_pattern_limit(tile_size, gap_size, r_limit)
    #print(np.shape(elements))
    # we can choose any ones you like to draw
    # tiling.elements = [elements[0]]  # choose the first one
    select_element = [elements[i] for i in select_num]
    tiling.elements = select_element  # choose the selected tiles
    tiling.config['margin'] = r_limit / scale * 1.05  # change the margin size
    tiling.config['Ltile-colour'] = '#63ace5'
    tiling.write_svg('pattern5.svg')
    # elements: each content in the list is a penrose object
    # if you need the center as coordinates:
    coord = []
    A = []
    B =[]
    C = []
    for e in select_element:
        coord.append(e.centre())
        A.append(e.A)
        B.append(e.B)
        C.append(e.C)
    return coord, A, B, C



# Example:
# for a penrose with tile size 250 nm, gap size 45 nm
# we can get the coordinates of centers as:

tiles_fullclosed1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
tiles_fullclosed2 = [348, 378, 303, 398, 258, 434, 208, 399, 198, 379, 199, 349, 209, 304, 259,264, 283, 284, 329, 328]
tiles_halfclosed = [42,62,63, 74, 75, 90, 91,105,106,120,121,134,135, 194,195, 204, 205, 145, 154, 155]
one = [0,1]

coordinates, A, B, C = main(250, 45, one)
x = np.array(np.real(coordinates)).T
y = np.array(np.imag(coordinates)).T
coords_cent = np.array([x,y]).T
np.savetxt('coords_dimer.txt', coords_cent)

x = np.array(np.real(B)).T
y = np.array(np.imag(B)).T
coords_B = np.array([x,y]).T

x = np.array(np.real(C)).T
y = np.array(np.imag(C)).T
coords_C = np.array([x,y]).T

corners_long = coords_C - coords_cent
corners_short = coords_B - coords_cent
coords_corners = np.concatenate((corners_long, corners_short),axis=0)
np.savetxt('basevecs_dimer.txt', coords_corners)




