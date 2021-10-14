import parmed
import mdtraj as md
from mdtraj.reporters import HDF5Reporter
import multiprocessing
from itertools import repeat

try:
    from openff.toolkit.topology import Molecule, Topology
    from openff.toolkit.typing.engines.smirnoff import ForceField
    from openff.toolkit.utils import AmberToolsToolkitWrapper
except ModuleNotFoundError:
    print("OpenFF not avaialble, trying OpenForceField")
    # from openforcefield.utils.toolkits import RDKitToolkitWrapper, ToolkitRegistry
    from openforcefield.topology import Molecule, Topology
    from openforcefield.typing.engines.smirnoff import ForceField
    from openforcefield.utils.toolkits import AmberToolsToolkitWrapper
    # from openforcefield.typing.engines.smirnoff.forcefield import ME
from simtk import unit
from simtk.openmm import LangevinIntegrator, CustomBondForce, AndersenThermostat, MonteCarloBarostat, VerletIntegrator, Platform
from simtk.openmm.app import Simulation, HBonds, NoCutoff, AllBonds, PME

import mlddec
from rdkit.Geometry import Point3D
from rdkit import Chem
from rdkit.Chem import AllChem

import copy
import tqdm
import tempfile
import collections
import numpy as np
from .trajectory import Trajectory

import logging
logger = logging.getLogger(__name__)


def _enemin_func(system, topology, coord):
    """returns optimised coord of a input conformer"""
    integrator = LangevinIntegrator(273 * unit.kelvin, 1/unit.picosecond, 0.002 * unit.picosecond)
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(coord)
    simulation.minimizeEnergy()
    new_coord = simulation.context.getState(getPositions = True).getPositions(asNumpy = True).value_in_unit(unit.angstrom)
    return new_coord

def _ene_func(system, topology, coord):
    """returns energy of a conformer"""
    integrator = LangevinIntegrator(273 * unit.kelvin, 1/unit.picosecond, 0.002 * unit.picosecond)
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(coord)
    energy = simulation.context.getState(getEnergy = True).getPotentialEnergy()
    return energy

class Simulator: #XXX put some variable to the class, e.g. the write out frequency, force field used
    # _MLDDEC_MODEL = collections.namedtuple("MLDDEC_MODEL", ["epsilon", "model"], defaults = [None, None])
    # MLDDEC_MODEL = _MLDDEC_MODEL()
    temperature = 273.15 * unit.kelvin
    time_step = 2 * unit.femtoseconds
    pressure = 1.013 * unit.bar
    

    solvent_lookup = {
        "WATER" : tuple(), #water has default parameters
        "CHLOROFORM" : ("C(Cl)(Cl)Cl", 1.5 * (unit.gram/(unit.centimeters**3)), 300),
        "DMSO" : ("CS(=O)C", 1.2 * (unit.gram/(unit.centimeters**3)), 300),
    }
        

    #================================================================

    @classmethod
    def add_solvent(name, smiles, density, num_solvent):
        """Add a new solvent to the solvent database as possible solvent for simulation. 

        Parameters
        ----------
        name : str
            Solvent name
        smiles : str
            SMILES of solvent
        density : Quantity
            Solvent density.
        num_solvent : int
            Number of solvent to add.
        """
        cls.solvent_lookup[name.upper()] = (smiles, density, num_solvent)

    @classmethod
    def load_mlddec(cls, epsilon): #XXX reload only when epsilon differ
        """Load machine learning partial charge charger.

        Parameters
        ----------
        epsilon : int
            Dielectric of the charger, either 4 or 80.
        """
        cls.model  = mlddec.load_models(epsilon = epsilon)
        cls.epsilon = epsilon
        # cls.MLDDEC_MODEL = cls._MLDDEC_MODEL(epsilon = epsilon, model = model)

    @classmethod
    def unload_mlddec(cls):
        """Unload machine learning partial charge charger, to save memory.
        """
        # cls.MLDDEC_MODEL = cls._MLDDEC_MODEL()
        del cls.model, cls.epsilon

    @classmethod
    def parameterise_system(cls, mol, which_conf = 0,
        force_field_path = "openff_unconstrained-1.3.0.offxml", solvent = None):
        """Parameterise the system for MD simulation.

        Parameters
        ----------
        mol : RestrainedMolecule
            Mol for simulation.
        which_conf : int, optional
            Which conformer in the mol to begin simulation, by default 0
        force_field_path : str, optional
            Force field file, by default "openff_unconstrained-1.3.0.offxml"
        solvent : str, optional
            The name of solvent for solvation, by default None

        Returns
        -------
        ParMed Structure
            Parameterised system.

        Raises
        ------
        ValueError
            Solvent not recognisable.
        """

        # model  = mlddec.load_models(epsilon = 4) #FIXME
        try:
            charges = mlddec.get_charges(mol, cls.model) #XXX change
        except TypeError:
            raise TypeError("No charge model detected. Load charge model first.")

        molecule = Molecule.from_rdkit(mol, allow_undefined_stereo = True) #XXX lost residue info and original atom names
        # molecule.to_file("/home/shuwang/sandbox/tmp.pdb", "pdb")

        topology = Topology.from_molecules(molecule)

        molecule.partial_charges = unit.Quantity(np.array(charges), unit.elementary_charge)
        molecule._conformers = [molecule._conformers[which_conf]]

        if solvent is None:
            forcefield = ForceField(force_field_path, allow_cosmetic_attributes=True)
            openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules= [molecule])

            structure = parmed.openmm.topsystem.load_topology(topology.to_openmm(), openmm_system)

            structure.title = Chem.MolToSmiles(mol)
            conf = mol.GetConformer(which_conf)
            structure.coordinates = unit.Quantity(
                np.array([np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]), unit.angstroms)
            
        elif solvent.upper() in cls.solvent_lookup:
            from mdfptools.Parameteriser import SolutionParameteriser

            if solvent.upper() == "WATER":
                structure = SolutionParameteriser.run(smiles = Chem.MolToSmiles(mol), default_padding = 1.25*unit.nanometer, solute_molecule = molecule, backend = "off_molecule", ff_path = force_field_path)

            else:
                
                solvent_smiles, density, num_solvent = cls.solvent_lookup[solvent.upper()]
                tmp = Chem.AddHs(Chem.MolFromSmiles(solvent_smiles))
                AllChem.EmbedMolecule(tmp)
                solvent_molecule = Molecule.from_rdkit(tmp)
                solvent_molecule.compute_partial_charges_am1bcc(toolkit_registry = AmberToolsToolkitWrapper())
                
                structure = SolutionParameteriser.run(smiles = Chem.MolToSmiles(mol), density = density, num_solvent = num_solvent, solvent_smiles = solvent_smiles, solvent_molecule = solvent_molecule, solute_molecule = molecule, backend = "off_molecule", ff_path = force_field_path)
        else:
            raise ValueError("{} is not a recognised solvent. Use `add_solvent` first to incorporate into the class.".format(solvent))

        cls.structure = structure
        return structure

    @classmethod
    def create_noe_force(cls, mol, spring_constant, tolerance = 0.0 * unit.angstrom, **kwargs):
        """Create time-averaged distance restraints to hydrogen pairs.

        Parameters
        ----------
        mol : RestrainedMolecule
            Mol contains the hydrogen atom pairs that require time-averaged distance restraining.
        spring_constant : float or list of floats
            The force constant(s) being applied to each hydrogen atom pair.
        tolerance : Quantity, optional
            Deviation to the reference value that can be tolerated, by default 0.0*unit.angstrom

        Returns
        -------
        OpenMM.CustomForce
            A flat bottom harmonic potential.
        """
        if type(spring_constant) is unit.Quantity:
            spring_constant = [spring_constant]

        if len(spring_constant) == 1: #all the same spring constant
            spring_constant *= len(mol.distance_upper_bounds)
        
        assert len(spring_constant) == len(mol.distance_upper_bounds), "Name of distance restraints do not match the number of spring constants"

        flat_bottom_harmonic_force = CustomBondForce('step(r-r0) * (k/2) * (r-r0)^2')
        flat_bottom_harmonic_force.addPerBondParameter('r0')
        flat_bottom_harmonic_force.addPerBondParameter('k')
        for idx, row in mol.distance_upper_bounds.iterrows(): #XXX generalise
            flat_bottom_harmonic_force.addBond( int(row.idx1),
                                                int(row.idx2),
                                                [row.distance * unit.angstrom + tolerance,  #XXX make this already a Quantity before reaching here
                                                spring_constant[idx]]) #XXX how to support iteration?
        
        return flat_bottom_harmonic_force

    @classmethod
    def simulate_tar(cls, mol, *, solvent = None, which_conf = 0,
        force_field_path = "openff_unconstrained-1.3.0.offxml",
        num_step = 5000, 
        avg_power = 3, 
        update_every = 1, 
        spring_constant = 1000 * unit.kilojoule_per_mole/(unit.nanometers**2),
        write_out_every = 2500, #5 picosecond
        platform = "CPU", #FIXME add platform
        use_cache = False,
        **kwargs):
        """Run time-averaged restraint simulation on a conformer of the molecule.

        Parameters
        ----------
        mol : RestrainedMolecule
            Mol containing the atom pair to be restrained info and the conformer to start simulation.
        solvent : str, optional
            Whether to include explicit solvent, None means no solvent, by default None
        which_conf : int, optional
            Conforer index for starting simulation, by default 0
        force_field_path : str, optional
            Path to forcefield file, by default "openff_unconstrained-1.3.0.offxml"
        num_step : int, optional
            Number of integration steps for MD, by default 5000
        avg_power : int, optional
            Exponential power to be used for averaging the forces, by default 3
        update_every : int, optional
            Update the simulation system with new reference distance value, by default 1
        spring_constant : Quantity or list of Quantities, optional
            Strength of force constant to applied, by default 1000*unit.kilojoule_per_mole/(unit.nanometers**2)
        write_out_every : int, optional
            Trajectory framing out frequency, by default 2500

        Returns
        -------
        Trajectory  
            Trajectory object containing the simulated frames.

        Raises
        ------
        ValueError
            If no restraining pairs can be found from the molecule object.
        """        

        platform = Platform.getPlatformByName(platform)

        if use_cache is True and cls.structure.title == Chem.MolToSmiles(mol):
            system_pmd = cls.structure
        else:
            system_pmd = cls.parameterise_system(mol, which_conf, force_field_path, solvent)

        if solvent is None:
            system = system_pmd.createSystem(nonbondedMethod=NoCutoff, nonbondedCutoff=1*unit.nanometer, constraints=HBonds)
            integrator = LangevinIntegrator(cls.temperature, 1/unit.picosecond, cls.time_step)

        else: #TODO does vacumm also need thermostat?
            system = system_pmd.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*unit.nanometer, constraints=HBonds)
            thermostat = AndersenThermostat(cls.temperature, 1/unit.picosecond)
            barostat = MonteCarloBarostat(cls.pressure , cls.temperature)
            system.addForce(thermostat)
            system.addForce(barostat)
            integrator = VerletIntegrator(cls.time_step)

        try:
            noe_force = cls.create_noe_force(mol, spring_constant, **kwargs)
            system.addForce(noe_force)
        except Exception as e:
            raise ValueError("No restraints can be applied to the simulation: \n {}".format(e))

        simulation = Simulation(system_pmd.topology, system, integrator, platform)
        simulation.context.setPositions(system_pmd.positions)

        simulation.minimizeEnergy()


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
        logger.info(""" Simulation at {} and {},
        spanning {} ps with frame writeout every {} ps.
        """.format(
            cls.temperature, cls.pressure, 
            (cls.time_step * num_step).value_in_unit(unit.picosecond), (cls.time_step * write_out_every).value_in_unit(unit.picosecond)
        ))
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


        traj = Trajectory()
        # tmp_df = system_pmd.to_dataframe()
        
        traj.from_parmed(system_pmd)
        traj.add_frames(coordinates)
        traj.copy_names_from_mol(mol, resnum = [0]) #XXX at the moment can only do this after `add_frames`
        return traj

    
    @classmethod
    def simulate_free(cls, mol, *, solvent = None, which_conf = 0,
        force_field_path = "openff_unconstrained-1.3.0.offxml",
        num_step = 5000, 
        avg_power = 3, 
        write_out_every = 2500, #5 picosecond
        platform = "CPU",
        use_cache = False,
        **kwargs):
        """Simulation of the system without any distance restraining forces.

        Parameters
        ----------
        mol : RestrainedMolecule
            Mol containing the atom pair to be restrained info and the conformer to start simulation.
        solvent : str, optional
            Whether to include explicit solvent, None means no solvent, by default None
        which_conf : int, optional
            Conforer index for starting simulation, by default 0
        force_field_path : str, optional
            Path to forcefield file, by default "openff_unconstrained-1.3.0.offxml"
        num_step : int, optional
            Number of integration steps for MD, by default 5000
        write_out_every : int, optional
            Trajectory framing out frequency, by default 2500

        Returns
        -------
        Trajectory  
            Trajectory object containing the simulated frames.
        """        

        if use_cache is True and cls.structure.title == Chem.MolToSmiles(mol):
            system_pmd = cls.structure
        else:
            system_pmd = cls.parameterise_system(mol, which_conf, force_field_path, solvent)

        if solvent is None:
            system = system_pmd.createSystem(nonbondedMethod=NoCutoff, nonbondedCutoff=1*unit.nanometer, constraints=HBonds)
            integrator = LangevinIntegrator(cls.temperature, 1/unit.picosecond, cls.time_step)

        else: #TODO does vacumm also need thermostat?
            system = system_pmd.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*unit.nanometer, constraints=HBonds)
            thermostat = AndersenThermostat(cls.temperature, 1/unit.picosecond)
            barostat = MonteCarloBarostat(cls.pressure , cls.temperature)
            system.addForce(thermostat)
            system.addForce(barostat)
            integrator = VerletIntegrator(cls.time_step)


        platform = Platform.getPlatformByName(platform)
        simulation = Simulation(system_pmd.topology, system, integrator, platform)
        simulation.context.setPositions(system_pmd.positions)

        simulation.minimizeEnergy()


        coordinates = []
        
        simulation.minimizeEnergy()
        logger.info(""" Simulation at {} and {},
        spanning {} ps with frame writeout every {} ps.
        """.format(
            cls.temperature, cls.pressure, 
            (cls.time_step * num_step).value_in_unit(unit.picosecond), (cls.time_step * write_out_every).value_in_unit(unit.picosecond)
        ))

        for iteration in tqdm.tqdm(range(num_step // write_out_every)):
            simulation.step(write_out_every)
            coordinates.append(simulation.context.getState(getPositions=True).getPositions(asNumpy=True))


        traj = Trajectory()
        # tmp_df = system_pmd.to_dataframe()
        
        traj.from_parmed(system_pmd)
        traj.add_frames(coordinates)
        traj.copy_names_from_mol(mol, resnum = [0]) #XXX at the moment can only do this after `add_frames`
        return traj


    @classmethod
    def minimise_energy_all_confs(cls, mol, n_jobs = -1, force_field_path = "openff_unconstrained-1.3.0.offxml",spring_constant = 1000 * unit.kilojoule_per_mole/(unit.nanometers**2), **kwargs): #XXX have a in_place option?
        """Forcefield based minimisation of all conformers in molecule object.

        Parameters
        ----------
        mol : RestrainedMolecule
            Mol containing all the conformers to minimise.
        n_jobs : int, optional
            Number of threads to run, by default -1 meaning all usable threads.
        force_field_path : str, optional
            Path to forcefield, by default "openff_unconstrained-1.3.0.offxml"
        spring_constant : Quantity or list of Quantities, optional
            Force constants for the instantaneous restraining, by default 1000*unit.kilojoule_per_mole/(unit.nanometers**2)

        Returns
        -------
        Trajectory  
            Trajectory object containing the simulated frames.
        """        

        system_pmd = cls.parameterise_system(mol, 0, force_field_path, None)

        #the list of task to distribute
        coord_list = []
        for i in range(mol.GetNumConformers()):
            conf = mol.GetConformer(i)
            system_pmd.coordinates = unit.Quantity(np.array([np.array(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())]), unit.angstroms)
            coord_list.append(system_pmd.positions)

        system = system_pmd.createSystem(nonbondedMethod=NoCutoff, nonbondedCutoff=1*unit.nanometer, constraints=HBonds)

        try:
            noe_force = cls.create_noe_force(mol, spring_constant, **kwargs)
            system.addForce(noe_force)
            logger.info("Setting up constrained minimisation.")
        except Exception as e:
            logger.info("Setting up unconstrained minimisation.")

        if n_jobs == -1:
            cores = multiprocessing.cpu_count()
        else:
            cores = int(n_jobs)

        with multiprocessing.Pool(processes=cores) as pool:
            out_coord = pool.starmap(_enemin_func, tqdm.tqdm(zip(repeat(system), repeat(system_pmd.topology), coord_list), total = len(coord_list)))


        out_mol = copy.deepcopy(mol)
        for i, val in enumerate(out_coord):
            conf = out_mol.GetConformer(i)
            for j in range(out_mol.GetNumAtoms()):
                conf.SetAtomPosition(j, Point3D(*val[j]))

        return out_mol

    @classmethod
    def calculate_energy_all_confs(cls, mol, n_jobs = -1, force_field_path = "openff_unconstrained-1.3.0.offxml",spring_constant = 1000 * unit.kilojoule_per_mole/(unit.nanometers**2), **kwargs): #XXX have a in_place option?
        """Calculate the energies of all conformers in the molecule object using potential energy surface provided by the forcefield.

        Parameters
        ----------
        mol : RDKit Mol
        n_jobs : int, optional
            Number of threads to run, by default -1 meaning all usable threads.
        force_field_path : str, optional
            Path to forcefield, by default "openff_unconstrained-1.3.0.offxml"

        Returns
        -------
        list of floats
            Energies (kJ/mol) of each conformer.
        """

        system_pmd = cls.parameterise_system(mol, 0, force_field_path, None)

        #the list of task to distribute
        coord_list = []
        for i in range(mol.GetNumConformers()):
            conf = mol.GetConformer(i)
            system_pmd.coordinates = unit.Quantity(np.array([np.array(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())]), unit.angstroms)
            coord_list.append(system_pmd.positions)

        system = system_pmd.createSystem(nonbondedMethod=NoCutoff, nonbondedCutoff=1*unit.nanometer, constraints=HBonds)

        if n_jobs == -1:
            cores = multiprocessing.cpu_count()
        else:
            cores = int(n_jobs)

        with multiprocessing.Pool(processes=cores) as pool:
            out_ene = pool.starmap(_ene_func, tqdm.tqdm(zip(repeat(system), repeat(system_pmd.topology), coord_list), total = len(coord_list)))

        return out_ene