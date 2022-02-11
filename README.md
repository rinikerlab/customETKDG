# customETKDG
This repo contains code associated with our 2022 publication: [Incorporating NOE-Derived Distances in Conformer Generation of Cyclic Peptides with Distance Geometry
](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c01165)

See the `./examples/` directory for demo notebooks showcasing the API calls.

## Installation
Download the `environment.yml` from this repo and navigate to that directory in the terminal:
```
#change ENVNAME to desired environment name
# we recommend first install and use `mamba` over `conda` as it creates the environment much faster
conda env create --name ENVNAME --file environment.yml

conda activate ENVNAME

pip install git+https://github.com/rinikerlab/customETKDG
```

In order to use the molecular dynamics functions, the [mlddec](https://github.com/rinikerlab/mlddec) and [mdfptools](https://github.com/rinikerlab/mdfptools) packages need to be additionally installed for system parameterisation (former for fast partial charge assignment of the ligand and the latter for solvated system setup).

Once installed, one should be able to run code from start to end in the `./examples/` directory.

## Use in Docker 
Alternatively, the repo can also be used inside a docker container:

### Build
```
#if build failed with error code 137, increase the RAM allocated to Docker.
docker build -t custom_etkdg .
```

### Run
```
#as interactive bash session
docker run -it --entrypoint /bin/bash custom_etkdg:latest

#as jupyter session, to run the demo notebook in `examples` folder
docker run -p 13579:13579 custom_etkdg
```

## Citation
```
@article{wang2022incorporating,
  title={Incorporating NOE-Derived Distances in Conformer Generation of Cyclic Peptides with Distance Geometry},
  author={Wang, Shuzhe and Krummenacher, Kajo and Landrum, Gregory A and Sellers, Benjamin D and Di Lello, Paola and Robinson, Sarah J and Martin, Bryan and Holden, Jeffrey K and Tom, Jeffrey YK and Murthy, Anastasia C and Popovych, Nataliya and Riniker, Sereina},
  journal={Journal of Chemical Information and Modeling},
  year={2022},
  publisher={ACS Publications}
  doi={10.1021/acs.jcim.1c01165}
}
```

## Contributors
Shuzhe Wang, Kajo Krummenacher, Greg Landrum