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


def _job_func(system, topology, coord):
    integrator = LangevinIntegrator(273 * unit.kelvin, 1/unit.picosecond, 0.002 * unit.picosecond)
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(coord)
    simulation.minimizeEnergy()
    new_coord = simulation.context.getState(getPositions = True).getPositions(asNumpy = True).value_in_unit(unit.angstrom)
    return new_coord
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
        cls.solvent_lookup[name.upper()] = (smiles, density, num_solvent)

    @classmethod
    def load_mlddec(cls, epsilon): #XXX reload only when epsilon differ
        cls.model  = mlddec.load_models(epsilon = epsilon)
        cls.epsilon = epsilon
        # cls.MLDDEC_MODEL = cls._MLDDEC_MODEL(epsilon = epsilon, model = model)

    @classmethod
    def unload_mlddec(cls):
        # cls.MLDDEC_MODEL = cls._MLDDEC_MODEL()
        del cls.model, cls.epsilon

    @classmethod
    def parameterise_system(cls, mol, which_conf = 0,
        force_field_path = "openff_unconstrained-1.3.0.offxml", solvent = None):

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
        """
        sprint constant either a value or a list of value same in size as number of restraints
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
        spring_constant = 2000 * unit.kilojoule_per_mole/(unit.nanometers**2),
        write_out_every = 2500, #5 picosecond
        platform = "CPU", #FIXME add platform
        use_cache = False,
        **kwargs):

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
    def minimise_energy_all_confs(cls, mol, n_jobs = -1, force_field_path = "openff_unconstrained-1.3.0.offxml", **kwargs): #XXX have a in_place option?

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
            out_coord = pool.starmap(_job_func, tqdm.tqdm(zip(repeat(system), repeat(system_pmd.topology), coord_list), total = len(coord_list)))


        out_mol = copy.deepcopy(mol)
        for i, val in enumerate(out_coord):
            conf = out_mol.GetConformer(i)
            for j in range(out_mol.GetNumAtoms()):
                conf.SetAtomPosition(j, Point3D(*val[j]))

        return out_mol