#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def local_hpconcentration_check(monomer_cord, HP_molecules, cut_off_radius):
    volume=(4/3)*np.pi*((cut_off_radius)**3)
    hp_inside=0
    for hp in range(np.shape(HP_molecules)[0]):
        dist=np.linalg.norm(monomer_cord- HP_molecules[hp])
        if dist<=cut_off_radius:
            hp_inside+=1

    return (hp_inside/volume)
if __name__ == "__main__":
        config = loadfile('finalconfig.out')
        HP= SL(config[0])
        C=[12.5445454, -4.8964, 6.44478]
        ct=3.0
        rho=local_concentration_check(C, HP, ct)
        print (rho)
