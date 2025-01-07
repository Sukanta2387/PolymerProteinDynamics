#!/bin/env python3

import os
import pickle
import subprocess
import sys
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

from pactools.runlmp.dump import Dump
from pactools.runlmp.elements import Elements
from pactools.runlmp.types import TypeSetup

########################################################################
########################################################################
########################################################################

"""
TODO:
    - write state dump via pickle
    - write log file containing series of simulations and basenames of output files


"""


class LMPSim:

    # elements: Elements
    # types: TypeSetup
    # simulation_box: np.ndarray
    # boundary: str
    # timestep: float
    # temperature: float
    # gamma: float
    # units: str
    # dimension: int
    # atom_style: str
    # init_velocity: bool
    # thermo: int
    # integrator: str

    # additional_lines: List[str]

    # dumps: List[Dump]
    # run_type: str
    # exec: str

    # num_fixes = 0

    def __init__(
        self,
        exec,
        run_type,
        simulation_box: np.ndarray,
        timestep: float,
        boundary="periodic",
        temperature=1,
        gamma=1,
        units="lj",
        atom_style="molecular",
        init_velocity=True,
        thermo=1000,
        integrator="langevin",
        elements=None,
        types=None,
        max_radius=0,
  
    ):

        self.exec = exec
        self.run_type=run_type

        self.simulation_box = simulation_box
        self.dimension = len(simulation_box)
        self.boundary = boundary
        
        self.num_fixes = 0

        #print('types:', types is None)
        #print('elements:', elements is None)

        if elements is None:
            self.elements = Elements(
                simulation_box, boundary, types=types, max_radius=max_radius
            )
        else:
            self.elements = elements
        self.types = self.elements.types

        self.timestep = timestep
        self.temperature = temperature
        self.gamma = gamma
        self.units = units

        self.atom_style = atom_style
        self.init_velocity = init_velocity
        self.thermo = thermo
        self.integrator = integrator

        self.additional_lines = list()

        self.dumps = list()

    def save_state(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def add_dump(
        self,
        name: str,
        groupname: str,
        type: str,
        every: int,
        fileext=None,
        args=None,
        overwrite=False,
    ):
        # find and remove existing dump if overwrite is True
        existing_dump = self._find_dump(name)
        if existing_dump is not None and overwrite:
            self.dumps.remove(existing_dump)
        # create and append new dump
        new_dump = Dump(name, groupname, type, every, fileext=fileext, args=args)
        self.dumps.append(new_dump)
        return new_dump

    def purge_dumps(self) -> None:
        self.dumps = list()

    def remove_dump(self, name: str) -> bool:
        dump = self._find_dump(name)
        if dump is None:
            return False
        self.dumps.remove(dump)
        return True

    def _find_dump(self, name: str) -> Dump:
        for dump in self.dumps:
            if dump.name == name:
                return dump
        return None

    def _find_dump_id(self, name: str) -> int:
        for i, dump in enumerate(self.dumps):
            if dump.name == name:
                return i
        return None

    def set_integrator(self, integrator: str) -> None:
        self.integator = integrator

    def add_additional_line(self, line: str) -> bool:
        if line[-1] != "\n":
            line += "\n"
        if line not in self.additional_lines:
            self.additional_lines.append(line)
            return True
        return False

    def add_additional_lines(self, lines: List[str]) -> None:
        for line in lines:
            self.add_additional_line(line)

    def run_sim(self, basefn: str, steps: int, processors =None, run_soft=False, seed=-1, save_state=True, additional_args = [],show_lmp_screen=False,output=True):
        """
        TODO: Add a default dump of the entire simulation state. Perhaps via pickle.
        """
        if steps <= 0:
            return

        if output:
            print(f"\n###################################################")
            print(f"# Running LAMMPS for {steps} timesteps")
            print(f"\n###################################################")

        # additional arguments passed to the commandline lmp execute
        additional_args = list(additional_args)

        # conf file
        conffn = basefn + ".conf"
        self.elements.genconf(conffn)

        # state xyz file
        # self.remove_dump('statexyz')
        # statedump = self.add_dump('statexyz', 'all', 'xyz', steps, fileext='_state.xyz')
        # statexyzfn = statedump.get_outfn(basefn)

        self.remove_dump("state")
        # statedump = self.add_dump(
        #     "state",
        #     "all",
        #     "custom",
        #     steps,
        #     fileext=".lmpstate",
        #     args="id x y z vx vy vz",
        # )

        statedump = self.add_dump(
            "state",
            "all",
            "custom",
            steps,
            fileext=".lmpstate",
            args="id x y z vx vy vz",
        )
        statedumpfn = statedump.get_outfn(basefn)

        # gen input file
        input_fn = basefn + ".in"
        self.gen_input_file(input_fn, conffn, basefn, steps, processors, run_soft, seed=seed)

        # specify print to logfile
        logfn = basefn + '.log'
        additional_args += ['-log',logfn]

        # surpress print to screen
        if not show_lmp_screen:
            additional_args += ['-screen','none']
        
        # run simulation
        self.execute_sim(input_fn,additional_args=additional_args)

        # # read positions from state file
        # if not os.path.isfile(statexyzfn):
        #     raise Exception("Cannot find statexyz file. The simulation presumably didn't finish")
        # self.elements.xyz2pos(statexyzfn)

        # read positions from state file
        if not os.path.isfile(statedumpfn):
            raise Exception(
                f"Cannot find statedump file '{statedumpfn}'. The simulation presumably didn't finish"
            )
        self.elements.custom2pos(statedumpfn)

        if save_state:
            self.save_state(basefn + ".state")

    def execute_sim(self, infile,additional_args=None):
        command = [arg for arg in self.exec.strip().split(" ") if arg != ""] + [
            "-in",
            infile,
        ]
        if additional_args is not None:
            if not isinstance(additional_args,list):
                additional_args = [additional_args]
            for aa in additional_args:
                command += ([arg for arg in aa.strip().split(" ") if arg != ""]) 
        # print(commasnd)
        cmd = subprocess.Popen(command)
        cmd.communicate()

    def gen_input_file(
        self,
        filename: str,
        conf_filename: str,
        baseout_filename: str,
        steps: int,
        processors: list,
        run_soft: bool,
        seed=-1,
        force_random_velocity=False,
    ):

        self.num_fixes = 0
        if seed == -1:
            seed = self._gen_seed()

        with open(filename, "w") as f:

            if (self.run_type=='parallel'):
                f.write('######################################\n')
                f.write('# Processors\n')
                f.write('\n')
                f.write(f' processors {processors[0]}  {processors[1]}  {processors[2]}\n')

            f.write("######################################\n")
            f.write("# Variables\n")
            f.write("\n")
            # variables
            f.write(f"variable T equal {self.temperature}\n")
            f.write(f"variable gamma equal {self.gamma}\n")
            f.write(f"variable seed  equal {seed}\n")

            f.write("\n")
            f.write("######################################\n")
            f.write("# Setup\n")
            f.write("\n")

            # general setups
            f.write(f"units {self.units}\n")
            f.write(f"dimension {self.dimension}\n")
            f.write(f"atom_style {self.atom_style}\n")

            bline = "boundary"
            if self.elements.boundary == "periodic":
                badd = " p"
            elif self.elements.boundary == "reflecting":
                badd = " f"
            else:
                raise Exception("Unknown boundary option '{self.elements.boundary}")
            for d in range(self.dimension):
                bline += badd
            f.write(bline + "\n")

            f.write("\n")
            f.write("######################################\n")
            f.write("# Read Conf\n")
            f.write("\n")

            # conf file
            f.write(f"read_data {conf_filename}\n")

            f.write("\n")
            for atomtype in self.types.atomtypes_list:
                f.write(f"mass {atomtype.id} {atomtype.mass}\n")

            f.write("\n")
            f.write("######################################\n")
            f.write("# Interactions\n")
            f.write("\n")

            # interactions
            for line in self.elements.types.gen_input_lines(run_soft):
                f.write(line)

            f.write("\n")
            f.write("######################################\n")
            f.write("# Additional Lines\n")
            f.write("\n")

            # interactions
            for line in self.additional_lines:
                f.write(line)

            f.write("\n")
            f.write("######################################\n")
            f.write("# Integrator\n")
            f.write("\n")
            # integrator:
            for line in self._integrator_lines():
                f.write(line)

            if self.boundary == "reflecting":
                f.write("\n")
                f.write("######################################\n")
                f.write("# Walls\n")
                f.write("\n")
                for line in self._repulsive_wall_lines():
                    f.write(line)
                f.write("\n")

            f.write("\n")
            f.write("######################################\n")
            f.write("# Dumps\n")
            f.write("\n")
            # dumplines = self._gen_dump_lines(baseout_filename)
            # for dumpline in dumplines:
            #     f.write(dumpline)
            for dump in self.dumps:
                dls = dump.dump_lines(baseout_filename)
                for dl in dls:
                    f.write(dl)

            f.write("\n")
            f.write("######################################\n")
            f.write("# Thermo\n")
            f.write("\n")
            f.write("thermo_modify    norm no\n")
            f.write(f"thermo           {self.thermo}\n")

            f.write("\n")
            f.write("######################################\n")
            f.write("# Time step\n")
            f.write("\n")
            f.write(f"timestep        {self.timestep}\n")

            f.write("\n")
            f.write("######################################\n")
            f.write("# Run\n")
            f.write("\n")
            f.write(f"run             {steps}\n")

            if run_soft:
                f.write("\n")
                f.write("######################################\n")
                f.write("# Unfix\n")
                f.write(f'unfix S')


    def _integrator_lines(self) -> List[str]:
        lines = list()
        if self.integrator == "langevin":
            self.num_fixes += 1
            lines.append("fix %d all nve\n" % (self.num_fixes))
            self.num_fixes += 1
            lines.append(
                "fix %d all langevin ${T} ${T} ${gamma} ${seed}\n" % (self.num_fixes)
            )
            return lines
        else:
            raise ValueError(
                f"Integrator specification '{self.integrator}' not implemented"
            )

    def _gen_seed(self, upper=10000000, lower=0) -> int:
        return np.random.randint(lower, upper)

    # def _gen_dump_lines(self,baseout_filename: str) -> List[str]:
    #     dumplines = list()
    #     for dump in self.dumps:
    #         fn = baseout_filename + dump.fileext
    #         dl = f'dump {dump.name} {dump.group} {dump.type} {dump.every} {fn}\n'
    #         dumplines.append(dl)
    #     return dumplines

    def _repulsive_wall_lines(self):
        lines = list()
        self.num_fixes += 1
        lines.append("fix %d all wall/lj93 xlo EDGE 1 1 0.7147\n" % (self.num_fixes))
        self.num_fixes += 1
        lines.append("fix %d all wall/lj93 xhi EDGE 1 1 0.7147\n" % (self.num_fixes))

        self.num_fixes += 1
        lines.append("fix %d all wall/lj93 ylo EDGE 1 1 0.7147\n" % (self.num_fixes))
        self.num_fixes += 1
        lines.append("fix %d all wall/lj93 yhi EDGE 1 1 0.7147\n" % (self.num_fixes))

        self.num_fixes += 1
        lines.append("fix %d all wall/lj93 zlo EDGE 1 1 0.7147\n" % (self.num_fixes))
        self.num_fixes += 1
        lines.append("fix %d all wall/lj93 zhi EDGE 1 1 0.7147\n" % (self.num_fixes))
        return lines


########################################################################
########################################################################
########################################################################
