

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from simtk.unit import nanometer, angstrom, dalton
from simtk.unit import nanometer as nm
from simtk.unit import sin, cos, acos
from sys import stdout

pdb = PDBFile('input.pdb')
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('output.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(90000)


forcefield = ForceField('amoeba2009.xml', 'amoeba2009_gk.xml')
system=forcefield.createSystem(nonbondedMethod=NoCutoff, soluteDielectric=2.0,
        solventDielectric=80.0)