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

## Docker

A convenient way to use our algorithms is within an encapsulated
Docker environment. Therefore we provide the needed files to create a
Docker image and run a Docker container.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


