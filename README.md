# SalaciaML-2-Arctic
# Oceanographic Data Processing and Quality Control
We have extended a classical algorithm by a deep learning neural network to support the quality
control (QC) of Arctic Ocean profile temperature and salinity data

## Train The Model - Salinity and Temperature

This files contains the code to train a machine learning model for predicting data quality. Data can be download from https://www.pangaea.de/ 

## Requirements:
* Python 3.x
* Pandas
* Matplotlib
* NumPy
* TensorFlow & Keras
* Scikit-learn
* Seaborn
* A data file named `UDASH-SML2A-Salinity.csv` or `UDASH-SML2A-Temperature.csv` in the same directory as the script, or the `DATA_FILE_PATH` variable updated accordingly.

## How to Run:
1.  Ensure all required libraries are installed (`pip install pandas matplotlib numpy tensorflow scikit-learn seaborn`).
2.  Place the `UDASH-SML2A-Salinity.csv` or `UDASH-SML2A-Temperature.csv` file in the correct location.
3.  Run the script: `python SalaciaML_Arctic_Salinity.py` or `python SalaciaML_Arctic_Temperature.py` 
4.  Output (model files, history, plots) will be saved in the `./model_output/` directory.
