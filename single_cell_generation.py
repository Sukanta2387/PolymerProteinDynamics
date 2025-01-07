
#!/bin/env python3


#################################################################################################################
# This script simulate single cell generation and perform post translation modifications in Monte Carlo cycles
#################################################################################################################


import os
import sys
import numpy as np
#from mp4py import MPI
import shutil

from pactools.runlmp.sim         import *
from pactools.runlmp.polymer     import *
from pactools.evals.concentration_check import local_hpconcentration_check as lconc
from pactools.evals.analysis_with_periodicbox import *
from pactools.evals.polymer_centerofmass import get_com 
#from pactools.evals.methylation_count import readsequence

###########################################################################
###########################################################################
# Gives no of HP1 particles inside spefified box given the concentration
###########################################################################
###########################################################################

def cal_N(c: float, L: float) -> int:
    return int(np.ceil(c*L**3))

###########################################################################
###########################################################################
# A check whether the cutoff volume of radius 1.5 crosses the box 
###########################################################################
###########################################################################

def _check_monomer_inbox(monomer_pos, box, cutoff_radius):
     flag=1
     for d in range(len(monomer_pos)):
          if (monomer_pos[d]>=(box[d][0]+cutoff_radius) and monomer_pos[d]<=(box[d][1]-cutoff_radius)):
               flag=1
          else:
               flag=0
               break
     if (flag==1): return True
     else: return False

###########################################################################
###########################################################################
# Gives radius of gyration of the polymer
###########################################################################
###########################################################################

def _getR_gyration(particles_cord):
    no_of_particles=np.shape(particles_cord)[0]

    com=get_com(particles_cord,  1.0)

    Rg=0.0

    for p in range(no_of_particles):

        Rg=Rg+((particles_cord[p][0]-com[0])**2+(particles_cord[p][1]-com[1])**2+(particles_cord[p][2]-com[2])**2)

    return Rg/no_of_particles

###########################################################################
###########################################################################
# Gives euchromatin and heterochromatin monomer positions separately
###########################################################################
###########################################################################

def _get_monomer_cords_blockwise(monomer_cord, polymer_block_length, block_no):
    total_no_neu=sum(block_no)*polymer_block_length
    Euch_cord= list(); Hetch_cord=list()
    block_sections=[np.arange(0, block_no[0], 1), np.arange(0, block_no[1], 1)]
    #print (np.shape(monomer_cord))

    for i in block_sections[0]:
        #print (i)
        Euch_cord.append(monomer_cord[2*i*polymer_block_length:2*i*polymer_block_length+polymer_block_length, 0:3])
    for i in block_sections[1]:
        Hetch_cord.append(monomer_cord[(2*i+1)*polymer_block_length:(2*i+1)*polymer_block_length+polymer_block_length, 0:3])
    #print (Euch_cord)

    Euch_cord = np.array(Euch_cord); Hetch_cord = np.array(Hetch_cord)
    #Euch_cord=np.reshape(Euch_cord, (polymer_block_length*block_no[0],3)); Hetch_cord=np.reshape(Hetch_cord, (polymer_block_length*block_no[1],3))

    return Euch_cord,Hetch_cord

###########################################################################
###########################################################################
# Generate the initial sequence given the distribution bias
###########################################################################
###########################################################################
def get_Heterochromatin_sequence(block_length: int, distribution_bias: float)-> str:

       H_seq=''

       if (np.random.uniform(0,1)<0.5):

           H_seq+='M'
           bias_flag=0

       else:
           H_seq+='N'
           bias_flag=1

       for i in range(block_length-1):

           if (bias_flag==0):

              if (np.random.uniform(0,1)<distribution_bias):
                 H_seq+='M'
                 bias_flag=0

              else:
                 H_seq+='N'
                 bias_flag=1

           else:

              if (np.random.uniform(0,1)<(1-distribution_bias)):

                 H_seq+='M'
                 bias_flag=0

              else:
                 H_seq+='N'
                 bias_flag=1

       return H_seq

###########################################################################
###########################################################################
# get current sequence of monomers
###########################################################################
###########################################################################

def getseq(atoms):
    seq = ''
    for atom in atoms:
        seq += atom.type.name
    return seq  

if __name__ == "__main__":

    
    run=int(sys.argv[1])  # optional run variable to avoid overlap

    #--------------------------------------------------
    # System parameters
    block_length = 50
    distribution_bias=0.3
    block_sections=[4,3]
    #--------------------------------------------------
    # MD parameters
    L       =50    # simulation box dimension
    conc    = 0.048 # HP1 concentration
    eps_hp1 = 1.1   # interaction strength between HP1- HP1
    eps_nuc = 2.0   # interaction strength between HP1- methylated nucleosome

    soft_run_steps=5000
    poly_eq_steps    = 20000
    total_eq_steps    = 5000000
    every   = 100000
    #--------------------------------------------------
    # MC parameters
    # methylation parameters
    eps_me=-1.4  
    prf_me=0.00001

    # demethylation parameters
    eps_de=.2634
    prf_de=0.0
    
    k = 75    # no of methylation attempts per monte carlo cycle
    R_local = 1.5  
    volume=(4/3)*np.pi*(R_local**3)
    steps   = 50000   # no of MD equilibration runs after each monte carlo cycle
    cycles  = 1200    # Total no of monte carlo cycle

    #--------------------------------------------------
    # LAMMPS parameters
    box         = np.array([[0, L], [0, L], [0, L]])
    boundary    = 'periodic'
    timestep = 0.0025
    temperature = 1
    gamma = 1
    units = 'lj'
    dimension = 3
    atom_style='molecular'
    init_velocity=True
    thermo=10000
    integrator='langevin'
    max_radius = 1
    disc_len    = 1
    lb          = 0
    run_soft=True
    lmp_path='/${lmp_path}'
    mpi_path='/${mpi_path}'
    sys.path.append(f'{mpi_path}')

   
    #--------------------------------------------------
    # Generating specific sequence of polymer
    # example sequence has 3 heterochromatin and 4 euchromatin blocks
    
    seq = 'N'*50
    seq += get_Heterochromatin_sequence(block_length, distribution_bias)
    seq += 'N'*50
    seq += get_Heterochromatin_sequence(block_length, distribution_bias)
    seq += 'N'*50
    seq += get_Heterochromatin_sequence(block_length, distribution_bias)
    seq += 'N'*50


    print(seq.count('M'))
 
    
    #--------------------------------------------------
    # run type and command line for MD run
    run_type='parallel'

    if run_type=='serial':
        processors=None
        exec = f'/{lmp_path}/lmp_serial'
    else:
        processors=[1,4,4]
        exec = f'mpirun -np 16 --mca btl self,vader,tcp /{lmp_path}/lmp_mpi'    # flags are subject to change depending on the system codes are running in
 

    #--------------------------------------------------   

    # create working directory

    new_dir=('with_methyEps_%.5f_demethyEps_%.5f_cycle_%.1f_group_%.1f_bias_%.2f_run_%.1f'%(eps_me, eps_de, cycles, k,distribution_bias, run)).replace('.','p')

    output_dir='~/output_dir'


    path= os.path.join(output_dir, new_dir)

    os.mkdir(path) 
    equilibration_achieved=False
    while equilibration_achieved is False:

        # header for all the files name (File type will be decided by choosing specific file extension)

        main_outname = ('~/output_dir/with_methyEps_%.5f_demethyEps_%.5f_cycle_%.1f_group_%.1f_bias_%.2f_run_%.1f/L%.1f_c%.6f_he%.3f_me%.3f'%(eps_me, eps_de, cycles, k, distribution_bias, run, L,conc,eps_hp1,eps_nuc)).replace('.','p')
       


    #--------------------------------------------------
        # creating a simulation object

        sim = LMPSim(exec, run_type,box,timestep,boundary=boundary,temperature=temperature,gamma=gamma,units=units,atom_style=atom_style,init_velocity=init_velocity,thermo=thermo,integrator=integrator,max_radius=max_radius)


        #--------------------------------------------------
        # setting up MD for polymer equilibration
        
        nuc  = sim.types.add_atomtype(  "N"  , 0.5 , 1 )
        mnuc = sim.types.add_atomtype(  "M"  , 0.5 , 1 )


        # generating polymer
        print('Generating polymer')
        num_monomer = len(seq)
        sequence    = [c for c in seq]
        add_polymer(    sim.elements, 
                        num_monomer, 
                        disc_len, 
                        sequence,
                        bend_modulus=lb,  
                        ev=True
                        )

        # soft run with only polymer

        soft1=main_outname+ '_polysoft'
        sim.run_sim(soft1, soft_run_steps, processors, run_soft)

        # run with default pair style

        polyequifn = main_outname + '_polyequi'
        sim.run_sim(polyequifn,poly_eq_steps, processors)

        #--------------------------------------------------
        # setting up MD for polymer+ protein equilibration
        hp1  = sim.types.add_atomtype( "hp1" , 0.5 , 1 )
        Nhp1  = cal_N(conc, L)
        print('Generating hp1')
        for i in range(Nhp1):
            if i%1000==0:
                print(f'{i}/{Nhp1} atoms added')
            sim.elements.add_atom("hp1",ev=True)

        soft2=main_outname+ '_totalsoft'
        sim.run_sim(soft2, soft_run_steps, processors, run_soft)
        sim.save_state(main_outname + '_init.state')

        #--------------------------------------------------
        # polymer_protein, protein-protein interaction on

        if eps_nuc != 0:
            sim.types.add_pair("M","hp1","lj/cut",2.5,[eps_nuc, 1.0])
        if eps_hp1 != 0:
            sim.types.add_pair("hp1","hp1","lj/cut",2.5, [eps_hp1, 1.0])
        #--------------------------------------------------   
        #  equilibration run
        #dump = sim.add_dump('custom', 'all', 'xyz', every, fileext='.xyz')
        equifn = main_outname + '_equilibration'
        sim.run_sim(equifn,total_eq_steps, processors)
        sim.save_state(equifn + '.state')


        droplet_formed=False
        target_Rg=6.3
        extra_MD_run=1000000
        extra_run=1
        max_extra_run=20

        # check whether the polymer has collapsed into a protein-polymer droplet

        while droplet_formed is False:

            het=list()

            polymer_config=sim.elements.get_atompos_of_types(['N', 'M'])[:,:3]
            updated_polymer_config=unwrap_polymer(polymer_config, L)

            Euch_cord, Hetch_cord=_get_monomer_cords_blockwise(updated_polymer_config, block_length, block_sections)
            for i in range(np.shape(Hetch_cord)[0]):
                het.extend(Hetch_cord[i])
            Rg=np.sqrt(_getR_gyration(het))

            if Rg<target_Rg:
                droplet_formed=True
                equilibration_achieved=True
                print (f'droplet has formed with Rg={Rg}')
            else:
                print (Rg)
                equifn = main_outname + f'_equilibration_{extra_run}'
                sim.run_sim(equifn,extra_MD_run, processors)
                sim.save_state(equifn + '.state')
                extra_run+=1

                if (extra_run>max_extra_run):
                    print ('droplet did not form in this run. setting up another run')
                    shutil.rmtree(path)
                    new_dir=('with_methyEps_%.5f_demethyEps_%.5f_cycle_%.1f_group_%.1f_bias_%.2f_run_%.1f'%(eps_me, eps_de, cycles, k,distribution_bias, run)).replace('.','p')
                    parent_dir='~/output_dir'
                    dir_path= os.path.join(output_dir, new_dir)
                    os.mkdir(dir_path)
                    break
    #--------------------------------------------------

    # write the sequence to file
    seqfn = main_outname + '.seq'
    with open(seqfn,'w') as f:
        f.write(getseq(sim.elements.get_atoms_of_types(['N','M']))+'\n')

 
    ###########################################################################
    ########################## Monte Carlo loops ##############################
    ###########################################################################
  

    # performing only the methylation reactions. No demethylation have been considered.

    for cc in range(cycles):


        cycle_outname = main_outname + f'_cycle{cc+1}'

        sim.run_sim(cycle_outname,steps, processors)
        # get all atoms of type N and M

        monomer_objects = sim.elements.get_atoms_of_types(['N','M'])
        polymer_config=sim.elements.get_atompos_of_types(['N', 'M'])[:,:3]

        #print (np.shape(polymer_config))
        # print([atom.type.name for atom in nucs])
        #-------------------------------
        #unwrap polymer get center of mass
        updated_polymer_config=unwrap_polymer(polymer_config, L)
        het=list()
        Euch_cord, Hetch_cord=_get_monomer_cords_blockwise(updated_polymer_config, block_length, block_sections)
        for i in range(np.shape(Hetch_cord)[0]):
               het.extend(Hetch_cord[i])
        polymer_com=get_com(het, 1.0)

        # generate new box and get the hps inside
        shifted_box=get_newbox(polymer_com, L)
        hps = sim.elements.get_atompos_of_type('hp1')[:,:3]
        hps_innewbox=get_HP_in_newbox(hps, shifted_box)
        #-------------------------------
        # select k atoms at random
        selected_ids = np.random.choice(len(monomer_objects),size=k,replace=False)
        #selected_nuc = [monomer_objects[id] for id in select]

        # loop over selected atoms and check whether its type should be changed
        for atom_id in selected_ids:
            atom=monomer_objects[atom_id]
            atom_pos=updated_polymer_config[atom_id]
            #methylation event
            if (atom.type.name != 'N'):
                continue
            else:
                  # here should be the concentration condition
                  #hps = sim.elements.get_atompos_of_type('hp1')[:,:3]
                  if _check_monomer_inbox(atom_pos, shifted_box, R_local):
                      local_hpconc = lconc(atom_pos,hps_innewbox,R_local)
                      no_hp=(local_hpconc*volume)
                      prb_methy=min(1,prf_me*(np.exp(-(eps_me*(no_hp))))) 
                      rn=np.random.uniform(0,1)
                      if (rn<prb_methy):
                        sim.elements.change_atomtype(atom.id, 'M')
                  else:
                      print ('alert')
                      new_box=get_newbox(atom_pos, L)
                      new_hps=get_HP_in_newbox(hps_innewbox, new_box)
                      local_hpconc = lconc(atom_pos,new_hps,R_local)
                      no_hp=(local_hpconc*volume)
                      prb_methy=min(1,prf_me*(np.exp(-(eps_me*(no_hp)))))
                      rn=np.random.uniform(0,1)
                      if (rn<prb_methy):
                        sim.elements.change_atomtype(atom.id, 'M')

        # write the sequence to file
        with open(seqfn,'a') as f:
            f.write(getseq(sim.elements.get_atoms_of_types(['N','M']))+'\n')

 
