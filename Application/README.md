# Data Processing and Quality Control

The Python scripts process oceanographic data, perform quality control
checks, and predict data quality using classical algorithms and a
machine learning model.

Please have a look into our respective publication for details.

## Data

Prepare your data exactly as in *TEST_DATA.csv*. It is crucial that
you follow the column names.

For temperature data:
Prof_no,year,month,Longitude_[deg],Latitude_[deg],Depth_[m],Temp_[°C],Salinity_[psu]

For salinity data:
Prof_no,year,month,Longitude_[deg],Latitude_[deg],Depth_[m],Salinity_[psu]

The order of columns can be changed and additional columns can be included.

For instance, our *TEST_DATA.csv*, which is actually a small UDASH
subset, contains Quality Flags for temperature and salinity named QF.


## Environment

We have tested the methods within the following setup:

- Ubuntu 20.04

- Python 3.10
 - apt-get install python3.10
 - apt-get install python3-pip

- pip3 install \\
  pandas==2.0.3 \\
  scikit-learn==1.3.1 \\
  numpy==1.23 \\
  tensorflow==2.13.1 \\
  keras==2.13.1 \\
  glob2==0.7 \\
  Seawater==3.3.4 \\
  Hampel==0.0.5 \\
  joblib==1.5.1 \\
  argparse==
  tqdm==4.67.1

## Edits

If you want to use your own data, you have to change the name of the
input file and optional for the output file in the Python scripts:
- App-SalaciaML-2-Arctic-Temperature.py: Line 416 and 417
- App-SalaciaML-2-Arctic-Salinity.py: Line 315 and 316


## Application

python3 ./App-SalaciaML-2-Arctic-Temperature.py
python3 ./App-SalaciaML-2-Arctic-Salinity.py


## Output

Will be the same file with 2 columns added, *Trad_QF_Salinity*,
representing the classical flags and *ML_QF_Salinity*, which are the final QC flags.
A *0* means "good" data, a *2* depicts a "subspect gradient" and a *4* represents a potential "spike".

