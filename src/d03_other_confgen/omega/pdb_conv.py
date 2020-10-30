import mdtraj as md
import cpeptools
import tempfile

obj  = md.load("CsE_omegamc_5400.pdb")
print(len(obj))
md.load("CsE_omegamc_5400.pdb").save("CsE_OMEGA_Macrocycle_numconf.pdb")