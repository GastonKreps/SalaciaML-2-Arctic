# SalaciaML-2-Arctic
# Oceanographic Data Processing and Quality Control
We have extended a classical algorithm by a deep learning neural network to support the quality
control (QC) of Arctic Ocean profile temperature and salinity data.

Details are explained in our paper submitted to *Frontiers of Marine Science*, named **SalaciaML-2-Arctic -- A deep learning quality control algorithm for arctic ocean temperature and salinity data**.


## Limitations

*SalaciaML-2-Arctic* is restricted to work correctly only for data
similar to UDASH, i.e. temperature and salinity profiles, measured by
CTD, bottle samples, mechanical thermographs and expendable
thermographs, located north of 65 °N.


## Training / Reproduce

The folder Training contains all information how to reproduce our analysis. Further the code can be used to extend or improve our algorithms.

## Application

This folder contains information how to apply our method to own data
using an own Python environment. In addition, to provide fast and easy
access to our method, we have implemented SalaciaML-2-Arctic as a web
service at https://mvre.autoqc.cloud.awi.de, where users simply upload
their data for the QC without programming or any extra software installation.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<!-- This files contains the code to train a machine learning model for predicting data quality. Data can be download from Pangaea (https://www.pangaea.de/)   -->

<!-- ## Requirements: -->
<!-- * Python 3.x -->
<!-- * Pandas -->
<!-- * Matplotlib -->
<!-- * NumPy -->
<!-- * TensorFlow & Keras -->
<!-- * Scikit-learn -->
<!-- * Seaborn -->
<!-- * A data file named `UDASH-SML2A-Salinity.csv` or `UDASH-SML2A-Temperature.csv` in the same directory as the script, or the `DATA_FILE_PATH` variable updated accordingly. -->

<!-- ## How to Run: -->
<!-- 1.  Ensure all required libraries are installed (`pip install pandas matplotlib numpy tensorflow scikit-learn seaborn`). -->
<!-- 2.  Place the `UDASH-SML2A-Salinity.csv` or `UDASH-SML2A-Temperature.csv` file in the correct location. -->
<!-- 3.  Run the script: `python SalaciaML_Arctic_Salinity.py` or `python SalaciaML_Arctic_Temperature.py`  -->
<!-- 4.  Output (model files, history, plots) will be saved in the `./model_output/` directory. -->
