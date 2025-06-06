# Data Processing and Quality Control

The Python scripts perform the ML training on UDASH data.

## Data

Download UDASH data at:
- Temperature: https://doi.pangaea.de/10.1594/PANGAEA.973235 !!!!! update
- Salininty: https://doi.pangaea.de/10.1594/PANGAEA.973235 !!!!! update

Name of the data files should be
- UDASH-SML2A-Temperature.csv
- UDASH-SML2A-Salinity.csv


## Environment

We have tested the methods within the following setup:

- Ubuntu 20.04

- Python 3.10
 - apt-get install python3.10
 - apt-get install python3-pip

- pip3 install pandas==2.0.3 scikit-learn==1.3.1 numpy==1.23  tensorflow==2.13.1 keras==2.13.1  glob2==0.7  Seawater==3.3.4 Hampel==0.0.5  joblib==1.5.1 tqdm==4.67.1 matplotlib==3.1.0 scikit-learn==0.21.2 seaborn==0.9.0


## Training:

- Place the *UDASH-SML2A-Salinity.csv* or *UDASH-SML2A-Temperature.csv* file and the python scripts in the same folder.
- Run the script: *python3 SalaciaML-2-Arctic-Salinity.py* or *python3 SalaciaML-2-Arctic-Temperature.py*.
- Output (model files, history, plots) will be saved in the *./model_output/* directory.
