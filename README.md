# Conditional Constrained Graph Variational Autoencoders (CCGVAE) for Molecule Design
This repository contains the code used to generate the results reported in the paper: [Conditional Constrained Graph Variational Autoencoders for Molecule Design](https://arxiv.org/abs/2009.00725).

```
@article{rigoni2020conditional,
  title={Conditional Constrained Graph Variational Autoencoders for Molecule Design},
  author={Rigoni, Davide and Navarin, Nicol{\`o} and Sperduti, Alessandro},
  journal={arXiv preprint arXiv:2009.00725},
  year={2020}
}
```

All the files related to the CCGVAE model will be uploaded soon.

# Dependencies
This project uses the `conda` environment.
In the `root` folder you can find, for each model, the `.yml` file for the configuration of the `conda` environment and also the `.txt` files for the `pip` environment. 
Note that some versions of the dependencies can generate problems in the configuration of the environment. For this reason, although the `setup.bash` file is present for the configuration of each project, it is better to configure them manually.

# Structure
The project is structured as follows: 
* `data`: contains the code to execute to make the dataset;
* `results`: contains the checkpoints and the results;
* `model`: contains the code about the model;
* `utils`: contains all the utility code;
* `histogramAnalysis`: contains all the code necessary to print the images about the histogram distribution.

# Usage
### Data Download
First you need to download the necessary files and configuring the environment by running the following commands:
```bash
sh setup.bash install
conda activate givae
```

### Data Pre-processing
In order to make de datasets type the following commands:
```bash
cd data
python make_dataset.py --dataset [dataset]
```
Where _dataset_ can be:
* qm9
* zinc


### Model Training
In order to train the model use:
```bash
python CCGVAE.py --dataset [dataset] --config '{"generation":0, "log_dir":"./results", "use_mask":false}'
```

### Model Test
In order to generate new molecules:
```bash
python CCGVAE.py --dataset [dataset] --restore results/[checkpoint].pickle --config '{"generation":1, "log_dir":"./results", "use_mask":false}'
```

While, in order to reconstruct the molecules:
```bash
python CCGVAE.py --dataset [dataset] --restore results/[checkpoint].pickle --config '{"generation":2, "log_dir":"./results", "use_mask":true}'
```

In order to analyze the results, we used the following environmet: [ComparisonsDGM](https://github.com/drigoni/ComparisonsDGM).

# Information
For any questions and comments, contact [Davide Rigoni](mailto:davide.rigoni.2@phd.unipd.it).

**NOTE:** Some functions are extracted from the following source [code](https://github.com/microsoft/constrained-graph-variational-autoencoder).

# Licenze
MIT

