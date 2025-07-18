FROM ubuntu:22.04

MAINTAINER autoqc <autoqc@autoqc.org>

RUN apt-get update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive \
    apt-get -y install \
    sudo \
    emacs \
    git \
    software-properties-common \
    python3 \	
    python3-pip

RUN pip3 install pandas==2.0.3 scikit-learn==1.3.1 numpy==1.26.4 glob2==0.7 Seawater==3.3.4 Hampel==0.0.5 joblib==1.4.2 tqdm==4.67.1 glob2==0.7 matplotlib==3.7.5 seaborn==0.13.2 keras==3.10.0 tensorflow==2.16.1

CMD sleep infinity
