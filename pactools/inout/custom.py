#!/bin/env python3

import os,sys
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

CUSTOM_NPY_EXT = '_custom.npy'

def load_custom(filename: str,savenpy=True,loadnpy=True,sortbyid=True, splitargs=False) -> dict:
    if not os.path.isfile(filename):
        raise FileNotFoundError( f"No such file or directory: '{filename}'" )
    fnpy = os.path.splitext(filename)[0]+CUSTOM_NPY_EXT
    if os.path.isfile(fnpy) and loadnpy and os.path.getmtime(fnpy) >= os.path.getmtime(filename):
        specs = read_specs(filename)
        print(f"loading data from '{fnpy}'")
        specs['data'] = np.load(fnpy)
    else:
        specs = read_custom(filename,sortbyid=sortbyid, splitargs=False)
        if savenpy:
            _save_custom_binary(fnpy,specs['data'])
    if splitargs:
        specs = customdata_splitargs(specs,remove_data=True)
    return specs

def read_specs(filename: str) -> dict:
    specs = dict()
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 3:
                specs['num_atoms'] = int(line.strip())
            if i == 4:
                specs['boundary'] = line.strip().split(' ')[3:]
            if i == 5:
                xlim = [float(entry) for entry in line.strip().split(' ')]
            if i == 6:
                ylim = [float(entry) for entry in line.strip().split(' ')]
            if i == 7:
                zlim = [float(entry) for entry in line.strip().split(' ')]
                simbox = np.array([xlim,ylim,zlim])
                specs['simulation_box'] = simbox
            if i == 8:
                specs['args'] = line.strip().split(' ')[2:]
            if i == 9:
                break
    return specs

def nskip(filehandler,n):
    line = filehandler.readline()
    if line == '':
        return False
    [filehandler.readline() for i in range(n-1)]
    return True

def customdata_splitargs(specs: dict,remove_data=True) -> dict:
    args = specs['args']
    data = specs['data']

    cpargs = [str(arg) for arg in args]
    if ('x' in cpargs and 'y' in cpargs and 'z' in cpargs):
        cpargs.append('position')
        cpargs.remove('x')
        cpargs.remove('y')
        cpargs.remove('z')
    if ('vx' in args and 'vy' in args and 'vz' in args):
        cpargs.append('velocity')
        cpargs.remove('vx')
        cpargs.remove('vy')
        cpargs.remove('vz')

    for arg in cpargs:
        if arg == 'position':
            specs['position'] = data[:,:,[args.index(d) for d in ['x','y','z']]]
            continue
        if arg == 'velocity':
            specs['velocity'] = data[:,:,[args.index(d) for d in ['vx','vy','vz']]]
            continue
        specs[arg] = data[:,:,args.index(arg)]
    if remove_data:
        del specs['data']

    return specs

def read_custom(filename: str, sortbyid=True, splitargs=False) -> dict:
    # print(f"reading '{filename}'")

    specs = read_specs(filename)
    N     = specs['num_atoms']
    args  = specs['args']
    nargs = len(args)

    data = list()
    with open(filename) as f:
        while nskip(f,9):
            snap = np.empty((N,nargs))
            for i in range(N):
                line = f.readline()
                snap[i] = [float(elem) for elem in line.strip().split(' ')]
            if sortbyid and 'id' in args:
                snap = snap[np.argsort(snap[:,args.index('id')])]
            data.append(snap)
    specs['data'] = np.array(data)
    if splitargs:
        specs = customdata_splitargs(specs,remove_data=True)
    return specs

def _save_custom_binary(outname: str,data: np.ndarray) -> None:
    if os.path.splitext(outname)[-1] == '.npy':
        outname += '.npy'
    np.save(outname,data)    


# if __name__ == "__main__":
    
#     if len(sys.argv) < 3:
#         print("usage: python %s fin fout"%sys.argv[0])
#         sys.exit(0)
#     fin  = sys.argv[1]
#     fout = sys.argv[2]
#     xyz  = load_xyz(fin)
#     # ~ fnpy = fin[:-4]+XYZ_NPY_EXT+'.npy'
#     # ~ save_xyz_binary(fnpy,xyz)
    
#     types = read_xyz_atomtypes(fin)
#     print(f'number of atoms = {len(types)}')
    
#     # ~ print(xyz.keys())
#     # ~ write_xyz(fout,xyz)
    
