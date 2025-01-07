#!/bin/env python3

import sys, glob, os
from ast import literal_eval

try:
    import numpy as np
except ModuleNotFoundError:
    print('numpy not installed. Please install numpy')
    sys.exit(1)

from pactools.tools import conditional_numba as condnumba


########################################################################
########################################################################
########################################################################

def simplest_type(string,convert_p=True):
    try:
        if convert_p:
            string = string.replace('p','.')
        return literal_eval(string)
    except:
        return string

def find_files(path: str, ext: str, params: dict, separator='_', recursive=False,excludes=[]):
    fns    = [fn for fn in glob.glob(os.path.join(path,'*.'+ext),recursive=recursive) if all(excl not in fn for excl in excludes)]
    fndata = list()
    for fn in fns:
        fndict = dict()
        fndict['fn'] = fn
        basefn = os.path.splitext(os.path.split(fn)[-1])[0]
        entries = [entry for entry in basefn.split(separator) if entry != '']
        for key in params.keys():
            identifier = params[key]
            for entry in entries:
                if entry[:len(identifier)] == identifier:
                    value = simplest_type(entry[len(identifier):])
                    fndict[key] = value
                    break
            if key not in fndict.keys():
                raise ValueError(f"file '{fn}' does not contain identifier '{identifier}'.")
        fndata.append(fndict)
    return fndata




