# LAMMPS and SMD Integration Framework

This repository provides a Python-based framework for integrating LAMMPS simulations with Synchronized Molecular Dynamics (SMD) method. 
## Features
- Implements a hybrid MD-CFD method using LAMMPS API.
  
## Requirements
### Software
- Python 3.8+ or higher
- The LAMMPS Python Module and Shared Library

Follow the link to install the LAMMPS python module
https://docs.lammps.org/Python_head.html

### Python Packages
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy
pandas
mpi4py
```

### Required Files
- **LAMMPS Input File**: Defines the molecular dynamics setup.
- **LAMMPS Data File**: Contains the initial atomic configuration.
- **SMD Parameter File**: Specifies parameters such as temperature, density, and stress coefficients.

## Usage
### Running the Script
The script is executed with the following command-line arguments:

```bash
##setup structure and files
mkdir -p RESULT_DIRECTORY
for i in `seq 0 NUMCELLS-1 `
do
  dir=${result_dir}/cell_${i}
  mkdir -p ${dir}
  cp LAMMPS_INPUT_FILE $dir/
  cp LAMMPS_DATA_FILE $dir/
done

## execute smd simulation
python smd_lmp.py --input LAMMPS_INPUT_FILE --output OUTPUT_FILE \
                     --result_dir RESULT_DIRECTORY --ncell NUM_CELLS \
                     --param PARAMETER_FILE [--pre_output PREVIOUS_OUTPUT]
```

#### Arguments
- `--input`: Path to the LAMMPS input file.
- `--output`: Name of the output file to store simulation results.
- `--result_dir`: Directory to save simulation outputs.
- `--ncell`: Number of spatial cells in the SMD model.
- `--param`: Path to the SMD parameter file.
- `--pre_output` (optional): Path to the previous output file for initializing arrays.


## Files and Directories
- `smd_lmp.py`: Main script to run the hybrid MD-SMD simulation.
- `requirements.txt`: List of required Python packages.
- `smd.param`: Parameters for calculation conditions .
- `README.md`: Documentation for the repository.

### Output Structure
Results for each spatial cell are stored in subdirectories under the `result_dir` with the naming convention:
```
result_dir/
  cell_0/
    results.out
  cell_1/
    results.out
  ...
  cell_N/
    results.out
```

## Contributing
Feel free to open issues or submit pull requests to improve this framework.

