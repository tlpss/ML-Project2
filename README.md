# ML-Project2  [![YourActionName Actions Status](https://github.com/tlpss/ML-Project2/workflows/unittest/badge.svg)](https://github.com/tlpss/ML-project2/actions)
Project 2 of ML course @ EPFL - Ensemble Kernel Methods for Portfolio Valuation

## Project Description 
TODO
## Project Structure
TODO
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



