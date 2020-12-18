import mdtraj as md

for i in range(6):
    inpt = "../../data/1" + str(i+1) + "_DPep" + str(i+1) + "/01_ref.pdb"
    outpt = "../../data/1" + str(i+1) + "_DPep" + str(i+1) + "/01_ref_corr.pdb"
    md.load(inpt).save(outpt)