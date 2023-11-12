# Copyright NavInfo Europe 2023. Adapted from Packnet-sfm Toyota Research Institute.

FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV PROJECT=mtsfmlearner
ENV PYTORCH_VERSION=1.7.1
ENV TORCHVISION_VERSION=0.8.2
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
ENV NCCL_VERSION=2.4.8-1+cuda10.1
ENV HOROVOD_VERSION=65de4c961d1e5ad2828f2f6c4329072834f27661
ENV TRT_VERSION=6.0.1.5
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG python=3.7
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libcudnn7=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg

# Instal Python and pip
RUN if [[ "${PYTHON_VERSION}" == "3.7" ]]; then \
    apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Pydata and other deps
RUN pip install future typing numpy pandas matplotlib jupyter h5py \
    awscli boto3 tqdm termcolor path.py pillow-simd opencv-python-headless \
    pycuda yacs cython==0.29.10

# Install PyTorch
RUN pip install torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} && ldconfig

# install horovod (for distributed training)
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    #HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir git+https://github.com/horovod/horovod.git@65de4c961d1e5ad2828f2f6c4329072834f27661 && \
    ldconfig

# Install DGP (dataset utils)
WORKDIR /workspace
RUN pip install nvidia-ml-py3==7.352.0 opencv-python==4.1.0.25 opencv-python-headless==4.5.1.48 packaging==20.8 \
    pandas==1.1.5 pandocfilters==1.4.3 path==15.0.1 path.py==12.5.0 pickleshare==0.7.5 Pillow==6.2.0 Pillow-SIMD==7.0.0.post3 \
    pip==20.3.3 prometheus-client==0.9.0 protobuf==3.14.0 psutil==5.8.0 pycocotools==2.0.0 pycparser==2.20 Pygments==2.7.4 \
    pyparsing==2.4.7 pyquaternion==0.9.5 pyrsistent==0.17.3 python-dateutil==2.8.0 python-gflags==3.1.2 pytools==2021.1 \
    pytz==2019.3 PyYAML==5.3.1 pyzmq==21.0.0 qtconsole==5.0.1 QtPy==1.9.0 requests==2.25.1 rsa==4.5 scikit-learn==0.24.1 \
    termcolor==1.1.0 tqdm==4.29.1 traitlets==5.1.1 typing==3.7.4.3 typing-extensions==3.7.4.3 \
    virtualenv==16.7.5 webencodings==0.5.1 wheel==0.36.2 zipp==3.4.0
ENV PYTHONPATH="/workspace/dgp:$PYTHONPATH"

# Override DGP wandb with required version
RUN pip install wandb==0.8.21
