This codebase implements the experimental framework described in:

## Self-Replicating Artificial Neural Networks Give Rise to Complex Evolutionary Dynamics
By  
**Boaz Shvartzman & Yoav Ram**

### Abstract
Evolution by natural selection is a universal phenomenon that requires heritable variation to take place.
Most evolutionary models apply exogenous heuristics to produce this heritable variation:
mutations do not occur spontaneously during replication,
but rather they are introduced exogenously by the modeler.
Here, we present a new model for studying evolution: the Self-Replicating Artificial Neural Network (SeRANN).
These neural networks are defined by a Python source code, which is mapped to a bit-string genotype and vice-versa using a novel and robust coding scheme based on variational autoencoders.
SeRANN individuals are trained using gradient descent to self-replicate their own genotype, which introduces endogenous and spontaneous mutations that provide the only source of genetic variation. 
In parallel with the replication task, SeRANN individuals are trained to perform a classification task that determines their fertility -- the number of offspring they will contribute to the next generation.
We evolved a population of 1,000 SeRANN individuals for 6,000 generations and documented a variety of complex evolutionary dynamics: 126,158 unique genotypes emerged during the experiment; the distribution of fitness effects (DFE) was similar to those estimated in viruses and microbes; the population evolved a reduced mutation rate as well as mutational robustness; adaptive evolution was prevalent, exhibited by the increase in frequency of multiple mutant alleles and genotypes; clonal interference between competing genotypes prevented any single genotype from taking over the population and reduced the rate of adaptation; both epistatis and phenotypic variation were common and had a complex effect on adaptation; and a trade-off between fertility and replication fidelity emerged, leading to sub-optimal performance of SeRANNs in both tasks.
Our results suggest that SeRANNs provide a compelling evolutionary model that bridges between experimental evolution with living organisms and theoretical models: it allows us to monitor and evaluate every mutant and every phenotypic trait without explicitly defining the mechanisms of replication and fertility, while giving rise to complex evolutionary dynamics.

## Software
This repository contains all the necessary components for a full evolutionary experiment on SeRANN:  

* Synthetic SeRANN generator: in order to train the ribosomal autoencoder, a dataset of diverse SeRANN source code examples is needed. This script generates as many examples as needed, according to a predefined transitions map.
* Helper scripts: before the ribosomal autoencoder training, the generated SeRANN source codes must be converted to numeric tokens. After the training, the dataset must also be encoded to binary genotypes, so the SeRANN could train on it to replicate. For both of these conversions, helper scripts are provided.
* Ribosomal autoencoder: the ribosomal autoencoder model definition and training script.
* Evolutionary experiment: the main part of this work. This module is used to run the evolutionary experiment and collect the results.
* SeRANN evaluation: some properties of SeRANN, such as fertility and mutation rate, can be measured more accurately using multiple training and evaluation cycles. This script facilitates this repeated evaluation procedure.

The components mentioned above are used for the following tasks, and in this order:
1. Generate a synthetic SeRANN source codes dataset.
2. Convert the dataset source codes to numeric token sequences.
3. Train the ribosomal autoencoder on the numeric token sequences.
4. Use the trained ribosomal autoencoder to encode the numeric token sequences into genotype vectors. 
5. Execute the evolutionary experiment on ancestor(s) chosen from the synthetic dataset.
6. (optional) evaluate specific/all genotypes retrospectively and more accurately using the SeRANN evaluation script. 

Alternatively, you can skip steps 1 and 3 and download our [generated synthetic source codes dataset](https://serann.s3.amazonaws.com/data/srann_datasets/generated_27032020.csv), [tokens vocabulary](https://serann.s3.amazonaws.com/data/vocabularies/generated_27032020.csv), and [pretrained ribosomal autoencoder model](https://serann.s3.amazonaws.com/models/genetic_autoencoder/sloppy-cornflower-dane_b69079.zip).

### Prerequisites
This code was tested with Python 3.8 on Ubuntu 20.04. The required Python packages are listed in *requirements.txt* and can be simply installed using pip:  
`pip install -r requirements.txt`

Some components mentioned above require intensive computations and GPU utilization. For example, to train hundreds of neural networks concurrently, multiple GPUs are needed. 
Because multi-GPU machines are expensive, we developed a Python library for distributed GPU computing. 
Using this library, multiple machines, each with at least one GPU device, located in different places around to world, can be used as workers to perform multiple GPU intensive tasks.
The API is similar to multiprocessing.Pool - the standard Python API for multiprocessing. For more information, refer to the [Github page of the library](https://github.com/boaz85/DistributedComputing).
