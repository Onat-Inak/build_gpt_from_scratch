# Create a working directory
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# proxy needs to be provided from the outside
ARG INSIDE_DOCKER_PROXY
ARG HOST_UNAME
ARG HOST_UID
ARG HOST_GID

# ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe
ARG HOST_UNAME
ARG HOST_UID
ARG HOST_GID

# set proxy
ENV http_proxy="${INSIDE_DOCKER_PROXY}"
ENV https_proxy="${INSIDE_DOCKER_PROXY}"
ENV HTTP_PROXY="${INSIDE_DOCKER_PROXY}"
ENV HTTPS_PROXY="${INSIDE_DOCKER_PROXY}"
ENV all_proxy="${INSIDE_DOCKER_PROXY}"

# Install some basic utilities
RUN rm -f /etc/apt/sources.list.d/*.list \
    && rm -vf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /deps

COPY docker/inputrc /root/.inputrc

SHELL ["/bin/bash", "-c"]

RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
    
ENV PATH=/opt/conda/bin:$PATH


# bootstrap conda
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate base \
    && conda update -n base conda -y \
    && conda install python=3.8 \
    && python --version \
    && source /opt/conda/etc/profile.d/conda.sh \
    && conda init bash \
    && conda config --set auto_activate_base true

ENV FORCE_CUDA="1"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# We use cuda-11.1 by default
RUN conda install cudatoolkit=11.1.1 -c conda-forge --yes \
&& pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install python packages
RUN pip3 install -U pip && pip3 install -U setuptools==59.5.0

RUN pip install wandb

# # install mmdet universe
# ARG MMCV="1.7.0"
# ARG MMDET="2.28.0"
# ARG MMSEG="0.30.0"
# ARG MMDET3D="1.0.0rc6"


# # install deps in dev mode (for developement / debugging)
# RUN cd /deps \
#     && git clone https://github.com/open-mmlab/mmcv.git \
#     && cd mmcv \
#     && git checkout tags/v${MMCV} \
#     && MMCV_WITH_OPS=1 pip install -e .

# RUN cd /deps \
#     && git clone https://github.com/open-mmlab/mmdetection.git \
#     && cd mmdetection \
#     && git checkout tags/v${MMDET} \
#     && pip install -e .

# RUN cd /deps \
#     && git clone https://github.com/open-mmlab/mmsegmentation.git \
#     && cd mmsegmentation \
#     && git checkout tags/v${MMSEG} \
#     && pip install -e .

# RUN cd /deps \
#     && git clone https://github.com/open-mmlab/mmdetection3d.git \
#     && cd mmdetection3d \
#     && git checkout tags/v${MMDET3D} \
#     && pip install scipy==1.7.3 \
#     && pip install scikit-image==0.20.0 \
#     && pip install -v -e .

# # new versions of yapf cause issues with mmcv config -> pin
# RUN pip install flash-attn==0.2.2 ipython \
#     && pip install yapf==0.40.1 \
#     && pip install wandb

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate base \
    && conda install -c conda-forge \
        matplotlib \
        tqdm \
        flatten-dict \
        imageio \
        joblib \
        numba \
        numpy \
        pandas \
        psutil \
        pyyaml \
        scipy \
        click \
        scikit-optimize \
        plotly \
        seaborn \
        tensorboard \
    && pip install datadiff \
        tiktoken

# add user of current host system
RUN groupadd -g ${HOST_GID} -o ${HOST_UNAME}
RUN useradd -l -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_UNAME}

# make sure conda env is always active on startup there as well
USER ${HOST_UNAME}


RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda init bash \
    && conda config --set auto_activate_base true

# switch back to root as default user
USER root

# copy inputrc to user for a nicer bash experience
RUN cp /root/.inputrc /home/${HOST_UNAME}/.inputrc \
    && chown ${HOST_UNAME}:${HOST_GID} /home/${HOST_UNAME}/.inputrc

WORKDIR /workspace





# FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
# # proxy needs to be provided from the outside
# ARG INSIDE_DOCKER_PROXY
# ARG HOST_UNAME
# ARG HOST_UID
# ARG HOST_GID

# # need this for apt to not get stuck prompting the user for locale info
# ARG DEBIAN_FRONTEND=noninteractive

# # set proxy
# ENV http_proxy="${INSIDE_DOCKER_PROXY}"
# ENV https_proxy="${INSIDE_DOCKER_PROXY}"
# ENV HTTP_PROXY="${INSIDE_DOCKER_PROXY}"
# ENV HTTPS_PROXY="${INSIDE_DOCKER_PROXY}"
# ENV all_proxy="${INSIDE_DOCKER_PROXY}"

# # Install conda
# COPY docker/conda.gpg /certs/conda.gpg
# RUN install -o root -g root -m 644 /certs/conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg
# RUN gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806
# RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list

# # fix cuda/nvidia key issue (https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112888486)
# # RUN rm /etc/apt/sources.list.d/cuda.list
# # RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# # RUN apt-key del 7fa2af80
# # RUN apt-get update && apt-get install -y --no-install-recommends wget
# # RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
# # RUN dpkg -i cuda-keyring_1.0-1_all.deb

# # COPY apt.sh /tmp/apt.sh

# # Install essentials (debian packages for Ubuntu 18.04)
# RUN set -ex && \
#     DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
#     # install debian packages
#     apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libfreetype6-dev \
#     libhdf5-serial-dev \
#     libpng-dev \
#     libzmq3-dev \
#     pkg-config \
#     software-properties-common \
#     unzip \
#     sudo \
#     libtiff-dev \
#     iproute2 \
#     # python
#     python3 \
#     python3-pip \
#     conda \
#     # nice to have: common editors
#     less \
#     nano \
#     vim \
#     # nice to have: copy files around
#     rsync \
#     libflann-dev \
#     liblz4-dev \
#     # nice to have: internet access
#     ca-certificates \
#     libcurl3-gnutls \
#     curl \
#     wget \
#     # nice to have: git
#     git \
#     # build tools
#     cmake \
#     make \
#     clang-6.0 \
#     unixodbc-dev \
#     apt-utils \
#     apt-transport-https \
#     gnupg \
#     # nice to have: libGL.so.1 required by python package cv2
#     libgl1-mesa-dev \
#     #
#     && \
#     # apt-get clean
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Setup and activate conda
# SHELL ["/bin/bash", "-c"]
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda -V
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda activate base \
#     && conda update conda
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda activate base \
#     && python --version
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda activate base \
#     && conda install python=3.8
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda activate base \
#     && python --version
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda activate base \
#     && conda install -c conda-forge \
#         matplotlib \
#         tqdm \
#         flatten-dict \
#         imageio \
#         joblib \
#         numba \
#         numpy \
#         pandas \
#         psutil \
#         pyyaml \
#         scipy \
#         click \
#         scikit-optimize \
#         plotly \
#         seaborn \
#         tensorboard \
#     && pip install datadiff
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda activate base \
#     && pip install mlflow wandb

# ENV FORCE_CUDA="1"

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda activate base \
#     && conda install cudatoolkit=11.1.1 -c conda-forge --yes \
#     && pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda activate base \
#     && conda install -c pyg -c conda-forge pyg

# # Setup python
# RUN ln -s /usr/bin/python3 /usr/bin/python
# RUN python --version
# RUN which python
# RUN whereis python

# # Install python packages
# RUN pip3 install -U pip
# RUN pip3 install -U setuptools

# # make sure this conda env is always active on startup
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda init bash \
#     && conda config --set auto_activate_base true

# # add user of current host sytem
# RUN groupadd -g ${HOST_GID} -o ${HOST_UNAME}
# RUN useradd -l -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_UNAME}

# # make sure conda env is always active on startup there as well
# USER ${HOST_UNAME}
# RUN source /opt/conda/etc/profile.d/conda.sh \
#     && conda init bash \
#     && conda config --set auto_activate_base true

# # switch back to root as default user
# USER root

# WORKDIR /workspace

# unset proxy
# ENV all_proxy=
# ENV http_proxy=
# ENV https_proxy=
# ENV HTTPS_PROXY=
# ENV HTTP_PROXY=
# ENV no_proxy=