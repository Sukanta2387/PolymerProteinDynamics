#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from pactools.tools.droplet_centerofmass import get_com



def getR_gyration(particles_cord, no_of_particles):

    com=get_com(particles_cord, no_of_particles, 1.0)

    Rg=0.0

    for p in range(no_of_particles):

        Rg=Rg+((particles_cord[p][0]-com[0])**2+(particles_cord[p][1]-com[1])**2+(particles_cord[p][2]-com[2])**2)

    return Rg/no_of_particles

