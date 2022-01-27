# Classroom_stress
Evaluate stress level through resting EEG

## Build environment
Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) and build a virtual environment by following code  
`conda env create -f environment.yml`

## Input new data
Since I hard-coded different forms to read data according to the folder name, it would be easier to change the folder name to resting_ASR with ch_lib.mat in this folder and put them in `data/`. 

## Run evaluation
- Open classification.ipynb
- `Modify parameters` (choose input_type, frequency range, select models...)
- `Run evaluation`, csv files containing results for each combination of methods will be generated automatically in `results/{folder}/`
- `Find highest performance among multiple csv files`, modify `data_name` to be the above `{folder}`, highest performance for different validation methods will be recorded in `results/record.csv`
