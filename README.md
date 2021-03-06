# ML-Project2  [![YourActionName Actions Status](https://github.com/tlpss/ML-Project2/workflows/unittest/badge.svg)](https://github.com/tlpss/ML-project2/actions)
Project 2 of ML course @ EPFL - Ensemble Kernel Methods for Portfolio Valuation

## Project Description 
This repo contains the code for the second project of the ML Course @ EPFL. We build on a [paper](https://arxiv.org/abs/1906.03726) written by Boudabsa and Filipovic, in which they describe a first ML method for dynamic portfolio valuation. We try to improve on their results using ensemble methods, while making sure that the valuation process can still be evaluated in closed-form.  To this end we implement Ensemble methods using the GPR implementation of Sklearn, and create the required scripts to evaluate them in low dimensions on jupyter notebooks and in higher dimensions on a HPC cluster with sbatch interface. 

## Project Structure

Tree of most important folders & files in the project:
```
.
├── Results
├── Test                                   unittests for models and helpers
├── aggregating/              
│   ├── gridsearch.py                     functions to train and evaluate a model and log the resulting error
│   ├── models.py                         the different aggregating models that are used
│   └── utils.py                          utils for creating models, datasets, loggers...
├── boosting/       
│   └── boosting_model_evaluation.py      code for evaluating the boosting models 
├── doc                                   contains the report
├── logs                                  stub folder to create logs files
├── mpi/                                  mpi python files and batch scripts for the different models
│   ├── Test                              tests to see if mpi is configured correctly
│   ├── *.py                              script that can be executed by mpi nodes to evaluate models
│   └── *.run                             sbatch run files that specify the required resources and provide the entry point for the related script
├── notebooks                             notebooks to evaluate the models in lower dimensions
├── requirements/                         pip requirement files
│   ├── requirements.txt        
│   └── mpi-requirements.txt
├── create_scitas_environment.sh          convenience script that sets up an MPI environment using gcc & mvapich2             
├── profile_memory_usage.py               script to get the memory used by a GPR
├── stochastic_models.py                  driver models that generate the datasets
└── visualisations.py                     functions to generate plots of the results 
```
## Setup
### Python 
- create a virtualenv (python 3.7 is guaranteed to work)
- `pip install -r requirements/requirements.txt`

### MPI
#### Local
- install working MPI distribution (e.g. windows -> MSMPI, Linux -> OpenMPI )
- `pip install -r requirements/mpi_requirements.txt`
- check if everything works fine by running `mpiexec -n 5 python mpi/Test/test_config.py`

#### SCITAS
- ssh into the fides cluster
- create a virtualenv using the `create_scitas_env.sh` file, this will build the `mpi4py` library using a specific chain of compiler(gcc), mpi flavour (mvapich) and the system configuration of the nodes 
- clone the project 
- change the username in the `.run`files to your own username 
- activate the environment
- test if the config is working by running `sbatch -p debug mpi_dummmy.run` from the mpi folder.
- monitor if your job executes as expected with `Sjob <jobID>` and the queue with `squeue -u <username>`
- verify  if the `<jobID>.out` file contains no errors

## Developer Guide 
- ReST docstrings for documentation
- unittest for testing
- SCITAS: cf Setup to run a job, make sure to check the SBATCH configuration and don't submit in the debug partition, more about python MPI on scitas --> https://scitas-data.epfl.ch/confluence/display/DOC/MPI4PY



