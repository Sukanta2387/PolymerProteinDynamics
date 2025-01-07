#!/bin/env python3


#################################################################################################################
# This script simulate multiple cell generations and perform post translation modifications in Monte Carlo cycles
#################################################################################################################



import os
import sys
import numpy as np
import subprocess
#import shlex
import shutil

from pactools.runlmp.sim         import *
from pactools.runlmp.polymer     import *
from pactools.evals.analysis_with_periodicbox import *
from pactools.evals.concentration_check import local_hpconcentration_check as lconc
from pactools.evals.polymer_centerofmass import get_com 
from pactools.tools.radius_of_gyration import getR_gyration


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
# Distributes the parental nucleosomes to two daughters
###########################################################################
###########################################################################

def old_nucleosome_distribution(parental_seq:list, distribution_bias:float)->list:

    daughter1=list()
    daughter2=list()
    #first nuclesome with equal probability
    if (np.random.uniform(0,1)<0.5):
           daughter1.append(parental_seq[0])
           daughter2.append('B')
           bias_flag=0
    else:
           daughter1.append('B')
           daughter2.append(parental_seq[0])
           bias_flag=1
    for i in range(1, len(parental_seq)):
           if (bias_flag==0):
              if (np.random.uniform(0,1)<distribution_bias):
                 daughter1.append(parental_seq[i])
                 daughter2.append('B')
                 bias_flag=0
              else:
                 daughter1.append('B')
                 daughter2.append(parental_seq[i])
                 bias_flag=1
           else:
              if (np.random.uniform(0,1)<(1-distribution_bias)):
                 daughter1.append(parental_seq[i])
                 daughter2.append('B')
                 bias_flag=0
              else:
                 daughter1.append('B')
                 daughter2.append(parental_seq[i])
                 bias_flag=1

    return daughter1 #, daughter2

###########################################################################
###########################################################################
# Deposites new neoclesomes on newly formed daughter strand
###########################################################################
###########################################################################

def new_nucleosome_deposition(daughter:list):

    daughter_seq=''
    for i in range(len(daughter)):
        if (daughter[i]=='B'):
            daughter_seq+='N'
        else:
            daughter_seq+=daughter[i]

    return daughter_seq

###########################################################################
###########################################################################
# Gives the fully methylated sequence for 1st cell generation
###########################################################################
###########################################################################

def get_seed_seq(block_length:int, total_blocks:int ):

    seed_seq=''
    flag=0
    for i in range(total_blocks):
        if (flag==0):
         seed_seq+='N'*block_length
         flag=1
        else:
            seed_seq+='M'*block_length
            flag=0

    return seed_seq


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
# get current sequence of monomers
###########################################################################
###########################################################################

def getseq(atoms):
    seq = ''
    for atom in atoms:
        seq += atom.type.name
    return seq


if __name__ == "__main__":

    #parameters
    #-----------------------------------------------------------#

    box_length=50
    no_of_cellgen=50  # no of cell generations to simulate
    copolymer_length=350  
    block_length=50
    no_of_blocks=7
    data_dump=10000

    distribution_bias=0.1

    # MD param
    e1=1.1        # interaction strength between HP1- HP1
    e2=2.0       # interaction strength between HP1- methylated nucleosome
    rho=0.048    # HP1 concentration
    soft_run_steps=5000
    MD_eq=5000000

    # MC param
    # methylation parameters
    eps_me=-1.4
    eps_de=.2634

    # demethylation parameters
    prf_de=0.0
    prf_me=0.00001

    R_cut=1.5
    attempts_perMC=75  # no of methylation attempts per monte carlo cycle
    MC_eq=50000     # no of MD equilibration runs after each monte carlo cycle
    total_MC=450    # Total no of monte carlo cycle

    sys_params={'L': box_length, 'every':data_dump, 'block_length':block_length}
    MD_params={'eps_hp1':e1, 'eps_nuc':e2, 'conc':rho, 'soft_run_steps':soft_run_steps, 'equi':MD_eq }
    MC_params={'eps_me':eps_me, 'eps_de':eps_de, 'prf_me':prf_me, 'prf_de':prf_de, 'R_local':R_cut, 'group':attempts_perMC, 'steps':MC_eq, 'cycles':total_MC }

    run=int(sys.argv[1])   # optional run variable to avoid overlap

    ###########################################################################
    ###########################################################################
    # Perform the equilibration run and methylation reactions in single cell generation
    ###########################################################################
    ###########################################################################
    def one_cell_generation( Gen: int, initial_seq: str, sys_params, MD_params, MC_params, dir_path, run):

        cell_evolution=list()
        #run=int(sys.argv[1])
        #processors=[2,2,4]
        L       = sys_params.get('L')
        polymer_block_length=sys_params.get('block_length')
        block_sections=[4,3]
        conc    = MD_params.get('conc')
        eps_hp1 = MD_params.get('eps_hp1')
        eps_nuc = MD_params.get('eps_nuc')
        soft_run_steps=MD_params.get('soft_run_steps')
        equi    = MD_params.get('equi')
        every   = sys_params.get('every')

        eps_me= MC_params.get('eps_me')
        prf_me= MC_params.get('prf_me')
        eps_de= MC_params.get('eps_de')
        prf_de= MC_params.get('prf_de')
        R_local = MC_params.get('R_local')
        volume=(4/3)*np.pi*(R_local**3)
        group = MC_params.get('group')
        steps= MC_params.get('steps')
        cycles= MC_params.get('cycles')
        run_soft=True

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
        lmp_path='/${lmp_path}'
        mpi_path='/${mpi_path}'
        sys.path.append(f'{mpi_path}')

        #--------------------------------------------------
        # Generating specific sequence of polymer

        seq=initial_seq

        #--------------------------------------------------
        # run type and command line for MD run
        run_type='parallel'

        if run_type=='serial':
            processors=None
            exec = f'/{mpi_path}/lmp_serial'
        else:
            processors=[2, 2, 4]
            exec = f'mpirun -np 16 --mca btl self,vader,tcp /{lmp_path}/lmp_mpi'     # flags are subject to change depending on the system codes are running in


        ################################################################
        equilibration_achieved=False
        while equilibration_achieved is False:
            
            main_outname = ('~/output_dir/cell_cycle_%d_of_total_%d_with_methyEps_%.5f_demethyEps_%.5f_cycle_%.1f_meth-attempts_%d_run_%d/L%.1f_c%.6f_he%.3f_me%.3f'%(Gen,no_of_cellgen, eps_me, eps_de, cycles, group,run, L,conc,eps_hp1,eps_nuc)).replace('.','p')
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
            
            #dump = sim.add_dump('custom', 'all', 'xyz', every, fileext='.xyz')

            # soft run with only polymer

            soft1=main_outname+ '_polysoft'
            sim.run_sim(soft1, soft_run_steps, processors, run_soft)

            # run with default pair style

            polyequifn = main_outname + '_polyequi'
            sim.run_sim(polyequifn,20000, processors)


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
            sim.run_sim(equifn,equi, processors)
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

                Euch_cord, Hetch_cord=_get_monomer_cords_blockwise(updated_polymer_config, polymer_block_length, block_sections)
                for i in range(np.shape(Hetch_cord)[0]):
                    het.extend(Hetch_cord[i])
                Rg=np.sqrt(_getR_gyration(het))

                if Rg<target_Rg:
                    droplet_formed=True
                    equilibration_achieved=True
                    print (f'droplet has formed with Rg={Rg} after {extra_run+5} million steps')
                else:
                    print (Rg)
                    equifn = main_outname + f'_equilibration_{extra_run}'
                    sim.run_sim(equifn,extra_MD_run, processors)
                    sim.save_state(equifn + '.state')
                    extra_run+=1

                    if (extra_run>max_extra_run):
                        print ('droplet did not form in this run. setting up another run')
                        shutil.rmtree(dir_path)
                        new_dir=('cell_cycle_%d_of_total_%d_with_methyEps_%.5f_demethyEps_%.5f_cycle_%.1f_meth-attempts_%d_run_%d'%(Gen,no_of_cellgen, eps_me, eps_de, cycles, group, run)).replace('.','p')
                        output_dir='~/output_dir'
                        dir_path= os.path.join(output_dir, new_dir)
                        os.mkdir(dir_path)
                        break
            

        # write the sequence to file

        seqfn = main_outname + '.seq'
        with open(seqfn,'w') as f:
            f.write(getseq(sim.elements.get_atoms_of_types(['N','M']))+'\n')

        cell_evolution.append(getseq(sim.elements.get_atoms_of_types(['N','M'])))


            
        ###########################################################################
        ########################## Monte Carlo loops ##############################
        ###########################################################################

        # for periodic boundary condition
        # performing only the methylation reactions. No demethylation have been considered.

        for cc in range(cycles):
            print (cc)


            cycle_outname = main_outname + f'_cycle{cc+1}'

            sim.run_sim(cycle_outname,steps, processors)


            # get all atoms of type N and M
            monomer_objects = sim.elements.get_atoms_of_types(['N','M'])
            polymer_config=sim.elements.get_atompos_of_types(['N', 'M'])[:,:3]
        
            #-------------------------------
            #unwrap polymer get center of mass
            updated_polymer_config=unwrap_polymer(polymer_config, L)
            het=list()
            Euch_cord, Hetch_cord=_get_monomer_cords_blockwise(updated_polymer_config, polymer_block_length, block_sections)
            for i in range(np.shape(Hetch_cord)[0]):
                 het.extend(Hetch_cord[i])
            polymer_com=get_com(het, 1.0)
            # generate new box and get the hps inside
            shifted_box=get_newbox(polymer_com, L)
            hps = sim.elements.get_atompos_of_type('hp1')[:,:3]
            hps_innewbox=get_HP_in_newbox(hps, shifted_box)
            #-------------------------------
            # select k atoms at random
            selected_ids = np.random.choice(len(monomer_objects),size=group,replace=False)
      
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

            cell_evolution.append(getseq(sim.elements.get_atoms_of_types(['N','M'])))
            # write the sequence to file
            with open(seqfn,'a') as f:
                f.write(getseq(sim.elements.get_atoms_of_types(['N','M']))+'\n')



        # for reflective boundary condition

        # for cc in range(cycles):
        #     print ('cycles are running')
        #     cycle_outname = main_outname + f'_cycle{cc+1}'

        #     sim.run_sim(cycle_outname,steps, processors)

        #     sim.elements.dump2xyz(main_outname+'_totalevolution',append=True,typedict=typedict,variable_atoms=variable_atoms, timestep=cc)

        #     # get all atoms of type N and M
        #     nucs = sim.elements.get_atoms_of_types(['N','M'])
        #     # print([atom.type.name for atom in nucs])

        #     # select k atoms at random
        #     select = np.random.choice(len(nucs),size=group,replace=True)
        #     # replace=true for multiple time selection and replace=False for unique selection
        #     selected_nuc = [nucs[id] for id in select]
        #     print ('still running')
        #     # loop over selected atoms and check whether its type should be changed
        #     for atom in selected_nuc:
        #         rn1=np.random.uniform(0,1)
        #         #if (rn1<0.5):
        #         #demethylation event
        #         #  if (atom.type.name == 'M'):
        #         #     hps = sim.elements.get_atompos_of_type('hp1')[:,:3]
        #         #     localconc = lconc(atom.position,hps,R_local)
        #         #     no_hp=(localconc*volume)
        #         #     prb_demethy=min(1,prf_de*(np.exp(-(eps_de*(no_hp)))))
        #         #     rn2=np.random.uniform(0,1)
        #         #     if (rn2<prb_demethy):
        #         #        sim.elements.change_atomtype(atom.id, 'N')


        #         if (rn1>0.0):
        #         #methylation event
        #         if (atom.type.name != 'N'):
        #             continue
        #         else:
        #             # here should be the concentration condition
        #             hps = sim.elements.get_atompos_of_type('hp1')[:,:3]
        #             localconc = lconc(atom.position,hps,R_local)
        #             no_hp=(localconc*volume)
        #             prb_methy=min(1,prf_me*(np.exp(-(eps_me*(no_hp)))))
        #             rn3=np.random.uniform(0,1)
        #             if (rn3<prb_methy):
        #                 sim.elements.change_atomtype(atom.id, 'M')
        #                 print ('changed type')
        #     print ('it ran')
        #     cell_evolution.append(getseq(sim.elements.get_atoms_of_types(['N','M'])))
        #     with open(seqfn,'a') as f:
        #         f.write(getseq(sim.elements.get_atoms_of_types(['N','M']))+'\n')

        #     print('##############################################')

        return cell_evolution

    
    all_cell_generations=list()

    for Gen in range(1, no_of_cellgen+1):
        print (Gen)

        # create directory to dump data for single generation

        
        new_dir=('cell_cycle_%d_of_total_%d_with_methyEps_%.5f_demethyEps_%.5f_cycle_%.1f_meth-attempts_%d_run_%d'%(Gen,no_of_cellgen, eps_me, eps_de, total_MC, attempts_perMC, run)).replace('.','p')
        output_dir='~/output_dir'

        path= os.path.join(output_dir, new_dir)
        os.mkdir(path)

        if (Gen==1):

            seed=[*get_seed_seq(block_length, no_of_blocks)]
            daughter_chromosome=new_nucleosome_deposition(old_nucleosome_distribution(seed, distribution_bias)) # get newly formed half methylated sequence

            daughter_after_PTM=one_cell_generation(Gen, daughter_chromosome, sys_params, MD_params, MC_params, path, run) # run single cell generation

            all_cell_generations.append(daughter_after_PTM)

        else:

            seed=[*all_cell_generations[Gen-2][-1]]
            print (seed)
            daughter_chromosome=new_nucleosome_deposition(old_nucleosome_distribution(seed, distribution_bias))  # get newly formed half methylated sequence

            daughter_after_PTM=one_cell_generation(Gen, daughter_chromosome, sys_params, MD_params, MC_params, path, run)  # run single cell generation

            all_cell_generations.append(daughter_after_PTM)

        shutil.rmtree(path)

    fn=open(('~/output_dir/all_cell_evolution_for_%d_blocks_%d_generations_with_bias_%.2f_mcattempts_%d_cyclepergen_%d_run_%d'%(no_of_blocks, no_of_cellgen, distribution_bias, attempts_perMC, total_MC, run)).replace('.','p'), 'a')


    for i in range(no_of_cellgen):

        total_evolution=all_cell_generations[i]

        for sq in total_evolution:
            fn.write(str(sq)+'\n')

        fn.write(str('--------------------------------------------------------------------')+'\n')

    fn.close()



