#!/bin/env python3

import os,sys
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

"""
########################################################
    
    
    specs = load_xyz(filename)
    specs = read_xyz(filename)
    
        readxyz always reads the xyz file while loadxyz accesses the 
        binary if it exists and creates it if it doesn't such that access
        will be accelerated next time.
            
########################################################
"""

XYZ_NPY_EXT = '_xyz.npy'

def load_xyz(filename: str,savenpy=True,loadnpy=True) -> dict:
    if not os.path.isfile(filename):
        raise FileNotFoundError( f"No such file or directory: '{filename}'" )
    fnpy = os.path.splitext(filename)[0]+XYZ_NPY_EXT
    if os.path.isfile(fnpy) and loadnpy and os.path.getmtime(fnpy) >= os.path.getmtime(filename):
            xyz          = dict()
            print(f"loading positions from '{fnpy}'")
            xyz['pos']   = np.load(fnpy)
            xyz['types'] = read_xyz_atomtypes(filename)
            return xyz
    xyz = read_xyz(filename)
    if savenpy:
        _save_xyz_binary(fnpy,xyz['pos'])
    return xyz

def load_pos_of_type(filename: str, selected_types: List[str], savenpy=True,loadnpy=True) -> np.ndarray:
    xyz = load_xyz(filename,savenpy=savenpy,loadnpy=savenpy)
    ids = [id for id in range(len(xyz['types'])) if xyz['types'][id] in selected_types]
    data = np.array([snap[ids] for snap in xyz['pos']])
    return data

def _linelist(line):
    return [elem for elem in line.strip().split(' ') if elem != '']

def read_xyz(filename: str) -> np.ndarray:
    print(f"reading '{filename}'")
    data = list()
    with open(filename) as f:
        line = f.readline()
        while line!='':
            ll = _linelist(line)
            if len(ll)>=4 and ll[0]!='Atoms.':
                snapshot = list()
                while len(ll)>=4:
                    snapshot.append( [float(ft) for ft in ll[1:4]])
                    line = f.readline()
                    ll   = _linelist(line)
                data.append(snapshot)
            line = f.readline()
    data = np.array(data)
    xyz = dict()
    xyz['pos']   = data
    xyz['types'] = read_xyz_atomtypes(filename)
    return xyz
    
def read_xyz_atomtypes(filename: str) -> List:
    data = list()
    with open(filename) as f:
        line = f.readline()
        num = 0
        types = list()
        while line!='':
            ll = _linelist(line)
            if len(ll)>=4 and ll[0]!='Atoms.':
                num += 1
                if num > 1:
                    break
                while len(ll)>=4:
                    types.append(ll[0])
                    line = f.readline()
                    ll   = _linelist(line)
            line = f.readline()
    return types
    
def write_xyz(outfn: str,data: dict,add_extension=True,append=False) -> None:
    """
    Writes configuration to xyz file
    
    Parameters
    ----------
    outfn : string 
        name of xyz file
    
    """
    if '.xyz' not in outfn.lower() and add_extension:
        outfn += '.xyz'
    
    pos   = data['pos']
    types = data['types']
    
    if 'timesteps' in data.keys() and len(data['timesteps']) == len(data['pos']):
        timesteps_provided = True
    else:
        timesteps_provided = False

    nbp   = len(pos[0])
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(outfn,mode) as f:
        for s,snap in enumerate(pos):
            f.write('%d\n'%nbp)
            if timesteps_provided:
                f.write('Atoms. Timestep: %d\n'%(data['timesteps'][s]))
            else:
                f.write('Atoms. Timestep: %d\n'%(s))
            for i in range(nbp):
                f.write('%s %.4f %.4f %.4f\n'%(types[i],snap[i,0],snap[i,1],snap[i,2]))
    
def _save_xyz_binary(outname: str,data: np.ndarray) -> None:
    if os.path.splitext(outname)[-1] == '.npy':
        outn = outname
    else:
        outn = outname + '.npy'
    np.save(outn,data)    


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python %s fin fout"%sys.argv[0])
        sys.exit(0)
    fin  = sys.argv[1]
    fout = sys.argv[2]
    xyz  = load_xyz(fin)
    types = read_xyz_atomtypes(fin)
    print(f'number of atoms = {len(types)}')
