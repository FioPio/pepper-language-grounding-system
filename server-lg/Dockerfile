FROM ubuntu:18.04

RUN apt update && apt install -y  libboost-all-dev\
                                  git\
				  software-properties-common\
				  python3.6

# CUDA
RUN add-apt-repository ppa:graphics-drivers -y

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 

RUN bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

RUN bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y cuda-10-0

RUN apt install -y libcudnn7 

ADD cudapath.txt .

RUN cat cudapath.txt>> ~/.profile

RUN apt install -y python3-pip libjpeg-dev zlib1g-dev

#############################
# Installing conda
#############################

# Install base utilities
RUN apt-get install -y build-essential  && \
    apt-get install -y wget &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


ADD environment.yml .

RUN conda env create -f environment.yml

########################################
# Activating conda
#######################################

# Initialize conda in bash config fiiles:
RUN conda init bash

# Activate the environment
RUN echo "conda activate zsgnet" > ~/.bashrc

#  use the new environment:
SHELL ["conda", "run", "-n", "zsgnet", "/bin/bash", "-c"]

RUN python -m spacy download en_core_web_md

RUN git clone https://github.com/TheShadow29/zsgnet-pytorch.git

WORKDIR zsgnet-pytorch

ADD demo.jpg .

RUN mkdir -p /root/.cache/torch/checkpoints/

RUN wget "https://download.pytorch.org/models/resnet50-19c8e357.pth"  -O /root/.cache/torch/checkpoints/resnet50-19c8e357.pth

# To get the server 
RUN pip install flask-restful

RUN echo 'export WERKZEUG_DEBUG_PIN=off'>>~/.bashrc

# Change this to your model
ADD referit_try.pth model.pth

ADD run.py code/.


ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "zsgnet", "python", "code/run.py"]
