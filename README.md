This repository implements the experimental framework described in:

## Self-Replicating Artificial Neural Networks Give Rise to Universal Evolutionary Dynamics
### Boaz Shvartzman and [Yoav Ram](https://www.yoavram.com)

## Software
This repository contains all the necessary components for a full evolutionary experiment with SeRANN (Self-Replicating Artificial Neural Networks):  

* Ribosomal autoencoder: the ribosomal autoencoder model definition and training script.
* Synthetic SeRANN generator: to train the ribosomal autoencoder, a dataset of diverse SeRANN source code examples is needed. This script generates as many examples as needed, according to a predefined transitions map.
* Helper scripts: before training the ribosomal autoencoder, the generated SeRANN source codes must be converted to numeric tokens. After training the ribosomal autoencoder, the dataset must be encoded to binary genotypes, so that SeRANN could train on it for their replication task. For both of these conversions, helper scripts are provided.
* Evolutionary experiment: the main part of this work. This module is used to run the evolutionary experiment and collect the results.
* SeRANN evaluation: some properties of SeRANN, such as fertility and mutation rate, can be measured more accurately using multiple training and evaluation cycles. This script facilitates the repeated evaluation procedure.

The components mentioned above are used in the following stages, and in this order:
1. Generate a synthetic SeRANN source code dataset.
2. Convert the dataset source codes to numeric token sequences.
3. Train the ribosomal autoencoder on the numeric token sequences.
4. Use the trained ribosomal autoencoder to encode the numeric token sequences into genotype vectors. 
5. Execute the evolutionary experiment on ancestor(s) chosen from the synthetic dataset.
6. (optional) evaluate specific/all genotypes retrospectively and more accurately using the SeRANN evaluation script. 

Alternatively, you can skip steps 1 to 4 and:  
1. Download our [pre-generated synthetic source codes dataset](https://figshare.com/ndownloader/files/37700322?private_link=f3b14612224d201cfbf2). It is not used directly by the evoluationary experiment, but is useful in order to train a new ribosomal autoencoder from scratch.  
2. Download our [pre-generated tokens vocabulary](https://figshare.com/ndownloader/files/37700310?private_link=f3b14612224d201cfbf2) and place it under data/vocabularies
3. Download our [pre-generated encoded synthetic source codes dataset](https://figshare.com/ndownloader/files/37700325?private_link=f3b14612224d201cfbf2) and place it under data/encodings_datasets  
4. Download [pre-trained ribosomal autoencoder model](https://figshare.com/ndownloader/files/37700277?private_link=f3b14612224d201cfbf2), unzip it place the extracted directory under models/ribosomal_autoencoder.  

### Prerequisites
This code was tested with Python 3.8 on Ubuntu 20.04. The required Python packages are listed in *requirements.txt* and can be installed using pip:  
`python -m pip install -r requirements.txt`  
The libraries [CuDNN](https://developer.nvidia.com/cudnn) and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) are also required. These are not Python libraries and must be downloaded and installed manually. The whole installation process should not take more than 30 minutes on a standard desktop.  

Some components mentioned above require intensive computations and GPU utilization.
For example, multiple GPUs are needed to train hundreds of neural networks concurrently. 
Because multi-GPU machines are expensive, we developed a Python library for distributed GPU computing. 
Using this library, multiple machines, each with at least one GPU device, located in different places around the world, can be used as workers to perform multiple GPU intensive tasks.
The API is similar to `multiprocessing.Pool`, the standard Python API for multiprocessing.
For more information, refer to the [Github repository of the library](https://github.com/boaz85/DistributedComputing).

### Example small-scale evolutionary experiment   
This repository contains an example parameters file under evolutionary_experiment/parameters. With the provided pre-generated and pre-trained files, it can be used to evolve a small population of 50 SeRANNs on a single GPU for 100 generations. Most of the modern GPUs should be able to support populations of up to 100 SeRANNs. We used 8 nVIDIA Tesla V100 GPUs to evolve a population of 1000 SeRANNs.  
To run the evoluationary experiment, navigate to the project's root and run:  

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python evolutionary_experiment/run_experiment --parameters=evolutionary_experiment/parameters/example.json
```  

On nVIDIA Titan X, a single generation execution time takes between 1 to 3 minutes. Thus, a whole experiment of 100 generations should take approximately 3.5 hours. Note that due to the small size of the population, it might collapse during the first generations, before reaching lower mutation rates and stablizing. In this case, no valid SeRANN is left to reproduce and the experiment ends.

### Results

The experiment results are written to an SQLite database file (generated in data/experiment_results) and updated after each generation ends.
The structure of the results DB and the meaning of each column in the tables are explained below.  

**execution_info**: experiment execution parameters and info.  
├── ancestor_genotye: the genotype of the ancestor, as bitstring.  
├── autoencoder: ribosomal autoencoder model friendly name.  
├── classification_image_dimensions: dimensions in pixels of the classification task images.  
├── encodings_dataset: the synthetic examples dataset name.  
├── vocabulary: the source code tokens vocabulary name.  
├── genotype_size: the length of the genotype.  
├── host_name: the server hostname, on which this experiment was executed.  
├── initial_offspring_pool_size: the number of offspring each SeRANN creates before sampling.  
├── max_srann_parameters: maximum number of parameters a SeRANN is allowed to have.  
├── max_srann_tokens: maximum number of source code tokens a SeRANN can have.  
├── num_classification_classes: the number of classes used in the classification task.  
├── num_generations: the number of generations after which the experiment would stop.  
├── num_sranns: the population size.  
├── offspring_pool_size_factor: the offspring pool size is updated to the maximum number of offspring any of the SeRANN had in the previous generation, multiplied by this factor. This number is greater than 1, and is used is a safety margin, to avoid cases where the pool size is smaller than the sample size.  
├── offspring_selection_strategy: how to offspring are selected from the offspring pool:  
├──── random: offspring are sampled randomly  
├──── best: offspring are selected according to their genotype's similarity to their parent's genotype.  
├── random_seed: necessary for reproducibility.  
├── selection_pressure: SeRANN absolute fertility values are raised by this power before normalization. Values greater than 1 will give better classifiers higher reproduction probabilities. Values less than 1 will equalize the probabilities.  
├── start_time: execution start time.  
├── training_batch_size: number of examples in each training step.  
├── training_epochs: The number of training epochs each SeRANN is going through.  
  

**generation**: generations info and cumulative statistics, collected during the experiment.  
├── experiment_id: the unique identifier of the experiment  
├── generation: the generation sequential number.  
├── start_time: time stamp of the generation processing start.  
├── overweight_rate: percentage of SeRANN with number of parameters above the limit (see experiment_info.max_srann_parameters)  
├── invalid_rate: percentage of SeRANN with invalid Python source code.  
├── survival_rate: percentage of valid SeRANN with a valid number of parameters in this generation.  
├── mean_parameters_count: the average number of SeRANN parameters in this generation.  
├── mean_absolute_fertility: the average classification accuracy, or absolute fertility, in this generation.  
├── absolute_fertility_std: the standard deviation of classification accuracy, or absolute fertility, in this generation.  
├── mean_loss_balance: the average loss balance parameter value in this generation.  
├── mean_classification_validation_accuracy: the average classification accuracy on the MNIST validation set, which determines SeRANNs fertility.  
├── mean_classification_training_accuracy: the average classification accuracy on the MNIST training set.  
├── mean_classification_test_accuracy: the average classification accuracy on the MNIST test set.  
├── max_classification_test_accuracy: the maximum classification accuracy on the MNIST test set.  
├── mean_replication_mse: the average SeRANN replication mean squared error on the synthetic SeRANN validation set.  
├── learning_time_seconds: the learning phase time.  
├── replication_time_seconds: the replication phase time.  
├── total_time_seconds: total time for this generation.  
├── mean_classification_layers: average number of classification layers in this generation.  
├── mean_replication_layers: average number of replication layers in this generation.  
├── mean_merged_layers: average number of merged layers in this generation.  
├── genotype_mean_pairwise_euclidean_distance: average euclidean distance between any possible pairs of genotypes in this generation.  
├── genotype_mean_pairwise_hamming_distance: average hamming distance between any possible pairs of genotypes in this generation.  
├── genotype_mean_euclidean_distance_from_parent: average euclidean distance of genotypes from their parents.  
├── genotype_mean_hamming_distance_from_parent: average hamming distance of genotypes from their parents.  
├── genotype_shannon_index: the shannon index (entropy) of genotypes distribution in this generation.  
├── genotype_nucleotide_diversity: the nucleotide diversity of genotypes in this generation.  
├── genotype_species_richness: the number of unique genotypes in this generation.  
├── source_code_median_levenshtein_distance_from_parent: the median levenshtein distance of source codes from their parents.  
├── source_code_shannon_index: the shannon index (entropy) of source codes in this generation.  
├── source_code_species_richness: the number of unique source codes in this generation.  
  
**serann**: meta information on each SeRANN individual in the experiment.  
├── id: unique identifier of this SeRANN in the experiment.  
├── genotype: the genotype of this SeRANN, as bitstring.  
├── source_code: the Python source code of this SeRANN.  
├── genotype_euclidean_distance_from_parent: the euclidean distance of this SeRANN's genotype from its parent's genotype.  
├── genotype_hamming_distance_from_parent: the euclidean hamming of this SeRANN's genotype from its parent's genotype, normalized by the genotype length.  
├── source_code_levenshtein_distance_from_parent: the levenshtein of this SeRANN's source code from its parent's source code.  
├── experiment_id: the unique identifier of the experiment.  
├── generation: the generation number this SeRANN is part of.  
├── num_offspring: the number of offspring this SeRANN has.  
├── parameters_count: the number of parameters in this SeRANN.  
├── loss_balance: the loss balance parameter value of this SeRANN.  
├── is_valid: wheather this SeRANN has a valid Python source code or not.  
├── is_overweight: Wheather the number of parameters of this SeRANN is greater than the upper limit.  
├── classification_validation_accuracy: the classification accuracy of this SeRANN on the MNIST validation set, which determines its fertility.  
├── classification_training_accuracy: the classification accuracy of this SeRANN on the MNIST training set.  
├── classification_test_accuracy: the classification accuracy of this SeRANN on the MNIST test set.  
├── replication_mse: the replication mean squared error of this SeRANN on the synthetic SeRANN validation set.  
├── absolute_fertility: the classification accuracy of this SeRANN, or absolute fertility, in this generation. The same as classification_validation_accuracy.  
├── relative_fertility: the relative fertility of this SeRANN, which is equal to its absolute fertility divided by the sum of all SeRANN absolute fertilities in this generation.  
├── classification_layers: the number of classification layers in this SeRANN.  
├── replication_layers: the number of replication layers in this SeRANN.  
├── merged_layers: the number of merged layers in this SeRANN.  


### Main experiment results  
The results of our main experiment, as described in the paper, are available as an SQLite file here:
[https://figshare.com/s/f3b14612224d201cfbf2](https://figshare.com/s/f3b14612224d201cfbf2).
