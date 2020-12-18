from simtk import unit
from simtk.openmm import LangevinIntegrator, CustomBondForce, Platform
from simtk.openmm.app import Simulation, HBonds, NoCutoff, OBC2
from rdkit import Chem
from rdkit.Geometry import Point3D
from mdtraj.reporters import HDF5Reporter
import mlddec
import copy
import tqdm
import parmed
import numpy as np
import joblib
from joblib import Parallel, delayed
import contextlib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument:
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def min_conformer(mol, structure, system, integrator, simsteps, i, hdf5_reporter_path):
    platform = Platform.getPlatformByName('CPU')
    int = copy.deepcopy(integrator)
    simulation = Simulation(structure.topology, system, int, platform)
    conf = mol.GetConformer(i)

    try:  # do not let one error kill the whole simulation
        structure.coordinates = unit.Quantity(
            np.array([np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]), unit.angstroms)
        simulation.context.setPositions(structure.positions)

        if hdf5_reporter_path is not False:
            simulation.reporters.append(HDF5Reporter(hdf5_reporter_path+'/report_conf_'+str(i+1)+'.h5', 1000))

        simulation.minimizeEnergy()
        simulation.step(simsteps)

        coords = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        return coords
    except Exception as e:
        print(e)
        import warnings
        warnings.warn(f'Conformer {i} (starting at 0) could not be simulated. Returning original coords.')
        coords = np.array([np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        return coords


def minimise_energy_confs(mol, noe_restraints=None, implicitSolvent=False, solventDielectric=1, use_mlddec=True,
                              mlddec_models=None, all_confs=True, confs_index=None, tar=False,
                              charges=None, int_stepsize_ps=0.002, simtime_ns=1,
                              allow_undefined_stereo=True, restraints=False, hdf5_reporter_path=False):

    simsteps = int((simtime_ns * unit.nanosecond)/(int_stepsize_ps * unit.picoseconds))

    mol = Chem.AddHs(mol, addCoords=True)

    # do some checks
    if not all_confs:
        assert confs_index is not None, 'Must provide \"confs_index\" for selection if \"all_confs=False\".'
    if restraints:
        assert noe_restraints is not None, 'Requested simulation with NOE restraints, but did not specify file'
    if (charges is None) and use_mlddec:
        print('Generating charges using mlddec. Otherwise must specify \"charges\".')
        if mlddec_models is None:
            if solventDielectric < 30:
                mlddec_models = mlddec.load_models(4)
            else:
                mlddec_models = mlddec.load_models(78)
        charges = mlddec.get_charges(mol, mlddec_models)


    from openforcefield.topology import Molecule, Topology
    from openforcefield.typing.engines.smirnoff import ForceField
    forcefield = ForceField('openff-1.2.0.offxml')

    tmp = copy.deepcopy(mol)
    tmp.RemoveAllConformers()   # Shuzhe: workround for speed beacuse openforcefield records all conformer
                                # informations, which takes a long time. but I think this is a ill-practice

    molecule = Molecule.from_rdkit(tmp, allow_undefined_stereo=allow_undefined_stereo)

    if charges is None:
        print('Generating charges using AM1 from Amber/sqm or OpenEye. This might take a long time. '
              'Otherwise must specify \"charges\" or set \"use_mlddec\" to true.')
        topology = Topology.from_molecules(molecule)
        openmm_system = forcefield.create_openmm_system(topology)
    else:
        molecule.partial_charges = unit.Quantity(np.array(charges), unit.elementary_charge)
        topology = Topology.from_molecules(molecule)
        openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules=[molecule])

    structure = parmed.openmm.topsystem.load_topology(topology.to_openmm(), openmm_system)
    integrator = LangevinIntegrator(273 * unit.kelvin, 1 / unit.picosecond, int_stepsize_ps * unit.picoseconds)
    integrator.setRandomNumberSeed(42)

    # C-H bonds are not assigned a type, so we fake one. We then need to constrain those as they are meaningless!
    bond_type = parmed.BondType(1.0, 1.0, list=structure.bond_types)
    for bond in structure.bonds:
        if bond.type is None:
            bond.type = bond_type

    # setup force for NOE restraints
    spring_const = 2000 * unit.kilojoule_per_mole/unit.nanometers**2  # spring constant K (see energy expression above)
                                                                    # in units compatible with joule/nanometer**2/mole
    flat_bottom_harmonic_force = CustomBondForce('step(r-r0) * (k/2) * (r-r0)^2')
    flat_bottom_harmonic_force.addPerBondParameter('r0')
    flat_bottom_harmonic_force.addPerBondParameter('k')
    for i in range(len(noe_restraints)):
        flat_bottom_harmonic_force.addBond(int(noe_restraints.iloc[i][0]), int(noe_restraints.iloc[i][4]),
                                           [noe_restraints.iloc[i][8] * unit.angstrom, spring_const])
    print(f'Added {flat_bottom_harmonic_force.getNumBonds()} NOE restraints to the simulation.')

    if implicitSolvent:
        system = structure.createSystem(nonbondedMethod=NoCutoff, nonbondedCutoff=1 * unit.nanometer,
                                        constraints=HBonds, implicitSolvent=OBC2, solventDielectric=solventDielectric,
                                        soluteDielectric=4)
    else:
        system = structure.createSystem(nonbondedMethod=NoCutoff, nonbondedCutoff=1 * unit.nanometer,
                                        constraints=HBonds)

    system.addForce(flat_bottom_harmonic_force)

    out_mol = copy.deepcopy(mol)
    if all_confs:
        confs_index = list(range(0, out_mol.GetNumConformers()))

    if tar:
        # parallelize expensive simulations
        with tqdm_joblib(tqdm.tqdm(desc="Simulation of conformers", total=len(confs_index))) as progress_bar:
            coords_set = Parallel(n_jobs=1)(delayed(tar_sim_conformer)(mol, structure, system, integrator, simsteps, i,
                                                                    hdf5_reporter_path) for i in confs_index)
    else:
        # parallelize expensive simulations
        with tqdm_joblib(tqdm.tqdm(desc="Simulation of conformers", total=len(confs_index))) as progress_bar:
            coords_set = Parallel(n_jobs=1)(delayed(min_conformer)(mol, structure, system, integrator, simsteps, i,
                                                                    hdf5_reporter_path) for i in confs_index)

    # assign new coords
    for i in confs_index:
        for j in range(out_mol.GetNumAtoms()):
            coords = coords_set[i-min(confs_index)]
            out_mol.GetConformer(i).SetAtomPosition(j, Point3D(*coords[j]))

    return out_mol

#######################################################################################################################
# TAR from Shuzhe

def tar_sim_conformer(mol, structure, system, integrator, simsteps, i, hdf5_reporter_path, avg_power = 1, update_every = 1):

    int = copy.deepcopy(integrator)
    simulation = Simulation(structure.topology, system, int)

    conf = mol.GetConformer(i)

    structure.coordinates = unit.Quantity(np.array([np.array(conf.GetAtomPosition(i)) for i in
                                                    range(mol.GetNumAtoms())]), unit.angstroms)

    simulation.context.setPositions(structure.positions)

    if hdf5_reporter_path is not False:
        simulation.reporters.append(HDF5Reporter(hdf5_reporter_path+'/report_conf_'+str(i+1)+'.h5', 1000))

    ##################++++

    # exp_time_ratio = np.exp(-2 / (50*10e3))#XXX 2fs / 50 ps should be tunable
    # one_minus_exp_time_ratio = 1 - exp_time_ratio
    #
    # noe_force = [i for i in system.getForces() if "CustomBondForce" in str(i)][-1] #TODO better selection criteria based on force group
    # idx1 = [noe_force.getBondParameters(i)[0] for i in range(noe_force.getNumBonds())]
    # idx2 = [noe_force.getBondParameters(i)[1] for i in range(noe_force.getNumBonds())]
    # force_consts = [noe_force.getBondParameters(i)[2][1] for i in range(noe_force.getNumBonds())] #only record the force constants
    #
    # tmp = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    #
    # current_dist = np.linalg.norm(tmp[idx1] - tmp[idx2], axis = 1)
    # lol = np.copy(current_dist)

    ##################----

    simulation.minimizeEnergy()

    ##################++++

    simulation.step(simsteps)
    # for abcde in range(simsteps // update_every):

        # simulation.step(update_every)
        # tmp = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        # previous_dist = np.copy(current_dist)
        # current_dist = np.linalg.norm(tmp[idx1] - tmp[idx2], axis = 1)
        #
        # if avg_power == 1:
        #     r0 = previous_dist * exp_time_ratio + current_dist * one_minus_exp_time_ratio
        # else:
        #     r0 = (previous_dist ** (-avg_power) * exp_time_ratio + current_dist ** (-avg_power) * one_minus_exp_time_ratio) ** (-1/avg_power)
        # r0 = lol
        # for i in range(noe_force.getNumBonds()):
        #     noe_force.setBondParameters(i, idx1[i], idx2[i], [r0[i], force_consts[i]])
        # print(abcde, r0)
        # noe_force.updateParametersInContext(simulation.context)

    ##################----
    coords = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    return coords