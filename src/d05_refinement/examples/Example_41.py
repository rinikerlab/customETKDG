

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from simtk.unit import nanometer, angstrom, dalton
from simtk.unit import nanometer as nm
from simtk.unit import sin, cos, acos
from sys import stdout

pdb = PDBFile('input.pdb')
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

modeller = Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield, pH=5.0)
modeller.addSolvent(forcefield) # add water and ions for neutrality

system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('output41.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(10000)