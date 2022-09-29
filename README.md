This repository implements the experimental framework described in:

## Self-Replicating Artificial Neural Networks Give Rise to Universal Evolutionary Dynamics
### Boaz Shvartzman and [Yoav Ram](https://www.yoavram.com)

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
This code was tested with Python 3.8 on Ubuntu 20.04. The required Python packages are listed in *requirements.txt* and can be installed using pip:  
`python -m pip install -r requirements.txt`

Some components mentioned above require intensive computations and GPU utilization.
For example, multiple GPUs are needed to train hundreds of neural networks concurrently. 
Because multi-GPU machines are expensive, we developed a Python library for distributed GPU computing. 
Using this library, multiple machines, each with at least one GPU device, located in different places around the world, can be used as workers to perform multiple GPU intensive tasks.
The API is similar to `multiprocessing.Pool`, the standard Python API for multiprocessing.
For more information, refer to the [Github page of the library](https://github.com/boaz85/DistributedComputing).


### Main experiment results  
[https://figshare.com/s/f3b14612224d201cfbf2](https://figshare.com/s/f3b14612224d201cfbf2)
