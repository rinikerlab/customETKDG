import sys
from openeye import oechem
from openeye import oeomega




omegaOpts = oeomega.OEMacrocycleOmegaOptions()
buildOpts = oeomega.OEMacrocycleBuilderOptions()

buildOpts.SetRandomSeed(42)

omegaOpts.SetMaxConfs(10)
omegaOpts.SetMacrocycleBuilderOptions(buildOpts)

# generate
mcomega = oeomega.OEMacrocycleOmega(omegaOpts)