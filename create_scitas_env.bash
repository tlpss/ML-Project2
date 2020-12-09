!/bin/bash

mkdir venv/${SYS_TYPE}_gcc_mvapich
ls venv/${SYS_TYPE}_gcc_mvapich
# import modules
module load gcc mvapich2 python
module list

# create venv
echo "creating virtualenv"
virtualenv -p python3 --system-site-packages venv/${SYS_TYPE}_gcc_mvapich
source venv/${SYS_TYPE}_gcc_mvapich/bin/activate

# pip install mpi4py
echo "install mpi4py"
pip install --no-cache-dir mpi4py

echo "environment configured, use source <env>/bin/activate to activate the environment"
