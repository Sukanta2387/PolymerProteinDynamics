import numpy as np
import sys,os

def split_xyz(xyzfn,outdir):
    
    with open(xyzfn) as f:
        lines = f.readlines()
        snaps = list()
        snap = list()
        for line in lines:
            if len(line.strip().split(' ')) == 1:
                if len(snap) > 0:
                    snaps.append(snap)
                snap = list()
            line = line.replace('A ','C ')
            snap.append(line)
        snaps.append(snap)
    
    print(len(snaps))
    
    for i,snap in enumerate(snaps):
        fn = outdir + '/' + f'{i+1}'.zfill(5) + '.xyz'
        with open(fn,'w') as f:
            for line in snap:
                f.write(line)
    
            
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("usage: python %s fin outdir"%sys.argv[0])
        sys.exit(0)

    xyzfn  = sys.argv[1]
    outdir = sys.argv[2]

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    split_xyz(xyzfn,outdir)
    
