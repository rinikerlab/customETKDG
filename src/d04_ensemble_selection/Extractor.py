ans = []

#mol = Chem.MolFromPDBFile("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf5400.pdb", removeHs=False)

i = 0

with open("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconf5400.pdb") as rf:
    for line in rf:
        line = line.strip()
        ans.append(line)
        i += 1
        if i == 70000:
            break

with open("/home/kkajo/Workspace/Conformers/CsA/CsA_chcl3_noebounds_numconfXXXX.pdb", 'w') as wf:
    for line in ans:
        wf.write(line)