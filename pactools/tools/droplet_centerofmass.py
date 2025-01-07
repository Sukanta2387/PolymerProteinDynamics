#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np




# This function gives the center of mass coordinate of the droplet 





def get_com(particles_cord, no_of_particles, mass_of_particles):
    com_cord=list()
    Xi=0.0; Yi=0.0; Zi=0.0


    #for com of droplet with reference of heterochromatin

    for p in range(no_of_particles):
        
        Xi+=(mass_of_particles*particles_cord[p][0])
        Yi+=(mass_of_particles*particles_cord[p][1])
        Zi+=(mass_of_particles*particles_cord[p][2])

    total_mass=mass_of_particles*no_of_particles

        

    com_cord.extend([Xi/total_mass, Yi/total_mass, Zi/total_mass])

    return com_cord




 
