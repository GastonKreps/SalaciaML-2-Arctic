# Data Processing and Quality Control

The Python scripts perform the ML training on UDASH data.

## Data

Download UDASH data at:
- Temperature: https://doi.pangaea.de/10.1594/PANGAEA.973235 !!!!! update
- Salininty: https://doi.pangaea.de/10.1594/PANGAEA.973235 !!!!! update

Name of the data files should be
- UDASH-SML2A-Temperature.csv
- UDASH-SML2A-Salinity.csv


## Testing

Just for technical testing purposes you can use the here provided files:
- UDASH-SML2A-Temperature_test.csv
- UDASH-SML2A-Salinity_test.csv

Note that these are very small files to check if the scripts are
working. They will not produce scientifically meaningful output.


## Environment

We have tested the methods within the following setup:

- Ubuntu 20.04

- Python 3.10
 - apt-get install python3.10
 - apt-get install python3-pip

- pip3 install pandas==2.0.3 scikit-learn==1.3.1 numpy==1.23 tensorflow==2.16.1 keras==3.10.0 glob2==0.7 Seawater==3.3.4 Hampel==0.0.5 joblib==1.4.2 tqdm==4.67.1 glob2==0.7 matplotlib==3.7.5  seaborn==0.13.2


## Training:

- Place the *UDASH-SML2A-Salinity.csv* or *UDASH-SML2A-Temperature.csv* file and the python scripts in the same folder.
- Change the input file names in the Python scripts to fit to the input data.
- Run the script: *python3 SalaciaML-2-Arctic-Salinity.py* or *python3 SalaciaML-2-Arctic-Temperature.py*.
- Output (model files, history, plots) will be saved in the *./model_salinity_output/* and *./model_temperature_output/* directory, respectively.
