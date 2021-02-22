import parmed
import mdtraj as md
from mdtraj.reporters import HDF5Reporter
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
# from openforcefield.typing.engines.smirnoff.forcefield import PME
from simtk import unit
from simtk.openmm import LangevinIntegrator, CustomBondForce
from simtk.openmm.app import Simulation, HBonds, NoCutoff, AllBonds

import mlddec
from rdkit.Geometry import Point3D
from rdkit import Chem

import copy
import tqdm
import tempfile
import collections
import numpy as np
from .trajectory import Trajectory


class Simulator:
    _MLDDEC_MODEL = collections.namedtuple("MLDDEC_MODEL", ["epsilon", "model"], defaults = [None, None])
    MLDDEC_MODEL = _MLDDEC_MODEL()

    #================================================================

    @classmethod
    def load_mlddec(cls, epsilon): #XXX reload only when epsilon differ
        model  = mlddec.load_models(epsilon = epsilon)
        cls.MLDDEC_MODEL = cls._MLDDEC_MODEL(epsilon = epsilon, model = model)

    @classmethod
    def unload_mlddec(cls):
        cls.MLDDEC_MODEL = cls._MLDDEC_MODEL()

    @classmethod
    def parameterise_solute(cls, mol, which_conf = 0,
        force_field_path = "openff_unconstrained-1.3.0.offxml"):

        forcefield = ForceField(force_field_path, allow_cosmetic_attributes=True)


        molecule = Molecule.from_rdkit(mol, allow_undefined_stereo = True)
        topology = Topology.from_molecules(molecule)

        # model  = mlddec.load_models(epsilon = 4) #FIXME
        try:
            charges = mlddec.get_charges(mol, cls.MLDDEC_MODEL.model) #XXX change
        except TypeError:
            raise TypeError("No charge model detected. Load charge model first.")
        molecule.partial_charges = unit.Quantity(np.array(charges), unit.elementary_charge)

        openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules= [molecule])

        structure = parmed.openmm.topsystem.load_topology(topology.to_openmm(), openmm_system)

        conf = mol.GetConformer(which_conf)
        structure.coordinates = unit.Quantity(
            np.array([np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]), unit.angstroms)
        return structure

    @classmethod
    def create_noe_force(cls, mol, spring_constant = 2000 * unit.kilojoule_per_mole/unit.nanometers**2):
        """
        sprint constant either a value or a list of value same in size as number of restraints
        """
        flat_bottom_harmonic_force = CustomBondForce('step(r-r0) * (k/2) * (r-r0)^2')
        flat_bottom_harmonic_force.addPerBondParameter('r0')
        flat_bottom_harmonic_force.addPerBondParameter('k')
        for _, row in mol.distance_upper_bounds.iterrows(): #XXX generalise
            flat_bottom_harmonic_force.addBond( int(row.idx1),
                                                int(row.idx2),
                                                [row.distance * unit.angstrom,  #XXX make this already a Quantity before reaching here
                                                spring_constant]) #XXX how to support iteration?
        
        return flat_bottom_harmonic_force

    @classmethod
    def simulate_tar(cls, mol, *, which_conf = 0,
        force_field_path = "openff_unconstrained-1.3.0.offxml",
        num_step = 5000, 
        avg_power = 3, 
        update_every = 1, 
        write_out_every = 2500, #5 picosecond
        **kwargs):

        solute_pmd = cls.parameterise_solute(mol, which_conf)
        solute_pmd.title = Chem.MolToSmiles(mol)

        try:
            noe_force = cls.create_noe_force(mol)
        except:
            raise ValueError("No Restraints to apply.")



        integrator = LangevinIntegrator(273*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        system = solute_pmd.createSystem(nonbondedMethod=NoCutoff, nonbondedCutoff=1*unit.nanometer, constraints=AllBonds)
        system.addForce(noe_force)
        simulation = Simulation(solute_pmd.topology, system, integrator)
        simulation.context.setPositions(solute_pmd.positions)


        # tmp_dir = tempfile.mkdtemp()
        # h5_path = "{}/tmp.h5".format(tmp_dir)
        # h5_path = "tmp.h5"


        coordinates = []
        
        ##################++++
        exp_time_ratio = np.exp(2 / (50*10e3))#XXX 2fs / 50 ps  should be tunable
        one_minus_exp_time_ratio = 1 - exp_time_ratio

        idx1 = [noe_force.getBondParameters(i)[0] for i in range(noe_force.getNumPerBondParameters())]
        idx2 = [noe_force.getBondParameters(i)[1] for i in range(noe_force.getNumPerBondParameters())]
        force_consts = [noe_force.getBondParameters(i)[2][1] for i in range(noe_force.getNumPerBondParameters())] #only record the force constants

        tmp = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        current_dist = np.linalg.norm(tmp[idx1] - tmp[idx2], axis = 1)
        ##################----   
        # h5_reporter = HDF5Reporter(h5_path, write_out_every)
        # simulation.reporters.append(h5_reporter)

        ##################++++

        simulation.minimizeEnergy()
        for iteration in tqdm.tqdm(range(num_step // update_every)):
            simulation.step(update_every)
            tmp = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

            if (update_every * iteration) % write_out_every == 0:
                coordinates.append(tmp)

            previous_dist = np.copy(current_dist)
            current_dist =  np.linalg.norm(tmp[idx1] - tmp[idx2], axis = 1)
            if avg_power == 1:
                r0 = previous_dist * exp_time_ratio + current_dist * one_minus_exp_time_ratio 
            else:
                r0 = (previous_dist ** (-avg_power) * exp_time_ratio + current_dist ** (-avg_power) * one_minus_exp_time_ratio) ** (-1/avg_power)

            for i in range(noe_force.getNumPerBondParameters()):
                noe_force.setBondParameters(i, idx1[i], idx2[i], [r0[i], force_consts[i]])

            noe_force.updateParametersInContext(simulation.context)
        ##################----


        traj = Trajectory(solute_pmd)
        traj.add_frames(coordinates)
        return traj
