# Data Processing and Quality Control

This Python script processes oceanographic data, performs quality control checks, and predicts data quality using a machine learning model. The script takes input data in CSV format, processes it, checks for various data quality issues, and provides predictions for problematic data points.

All in 1 scripts are for temperature and salinity and they perform the traditional check and then the Machine alearning process. 

## Prerequisites

Before running the script, ensure you have the following installed:

Ubuntu 20.04

Python 3.10.12

pip3 install pandas==2.0.3 scikit-learn==1.3.1 numpy==1.23
tensorflow==2.13.1 keras==2.13.1 glob2==0.7 Seawater==3.3.4 Hampel==0.0.5

## Input

Files should be in .csv format and same structure as the example in this folder

## Ouput

Will be the same file with 2 columns added, Trad_QF_Salinity,ML_QF_Salinity
## License

This project is licensed under the project M-VRE: The MOSAiC - Virtual Research Environment
