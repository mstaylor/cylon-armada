FROM ubuntu:22.04
MAINTAINER cylondata@googlegroups.com

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

ENV CYLON_HOME=/cylon
ENV CYLON_PREFIX=/cylon/install

ENV BOOST_VERSION=1.81.0
ENV BOOST_VERSION_UNDERSCORE=1_81_0


#give ARG EXPOSE_ENV a default value
ARG EXPOSE_ENV=33000

#assign the EXPOSE_ENV arg to the EXPOSE_ENV ENV so that it can be accessed
#by the subsequent RUN call within the container
ENV EXPOSE_ENV=$EXPOSE_ENV




RUN DEBIAN_FRONTEND=noninteractive apt-get update -y --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl openssl git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh && \
   /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    #/opt/conda/bin/conda clean -a && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate cylon_dev" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=$CYLON_PREFIX"/lib >> ~/.bashrc

ADD https://google.com cache_bust

ARG CYLON_FMI_BRANCH_REF=39c14b7df9a0c3eb260b85c155650ec176f3aab3

WORKDIR $CYLON_HOME


RUN git clone https://github.com/mstaylor/cylon.git $CYLON_HOME

RUN cd $CYLON_HOME && git checkout -b fmi-ds5110-support ${CYLON_FMI_BRANCH_REF}

SHELL ["/bin/bash", "--login", "-c"]

RUN conda env create -f $CYLON_HOME/conda/environments/fmi.yml

#RUN echo "conda activate cylon_dev" >> ~/.bashrc
#ENV PATH /opt/conda/envs/cylon_dev/bin:$PATH


#RUN /opt/miniconda/bin/conda env create --file $CYLON_HOME/conda/environments/cylon_NoUCX.yml --prefix /opt/conda-env

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    conda install -n base conda-libmamba-solver && conda config --set solver libmamba && \
    conda install -c anaconda boto3 -y && pip install cloudmesh-openstack && \
    pip install awslambdaric && \
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install timm==0.4.12 --no-deps && \
    conda remove -y pyopenssl || true && \
    pip uninstall -y pyOpenSSL || true && \
    pip install pyOpenSSL==23.2.0 --force-reinstall

#RUN ln -sf /opt/conda-env/bin/python /usr/bin/python

RUN apt-get update &&  apt-get install autoconf -y &&  apt-get install libtool -y && apt-get install cmake -y && \
    apt-get install libnuma-dev -y

#UCX install
#Note: Includes support for overriding device eth address

#boost
#ADD https://google.com cache_bust

RUN wget https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERSCORE}.tar.bz2 && \
    tar --bzip2 -xf boost_${BOOST_VERSION_UNDERSCORE}.tar.bz2

RUN cd boost_${BOOST_VERSION_UNDERSCORE}  && \
    ./bootstrap.sh --with-python=python3 && ./b2 -d+2 cxxflags="-fPIC" link=static install

# Clean up
RUN rm -rf /boost_${BOOST_VERSION_UNDERSCORE}.tar.bz2

#RUN apt-get install zlib1g-dev libcurl4-openssl-dev libssl-dev -y

RUN apt-get install zlib1g-dev libssl-dev -y


ARG AWS_SDK_CPP_REF=1.11.909

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    git clone --depth 1 --branch ${AWS_SDK_CPP_REF} --recurse-submodules --shallow-submodules https://github.com/aws/aws-sdk-cpp.git && cd aws-sdk-cpp && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3" -DBUILD_SHARED_LIBS="off" -DENABLE_TESTING=OFF -DAUTORUN_UNIT_TESTS=OFF .. && make -j6 install && cd ../.. && rm -rf aws-sdk-cpp


#REDIS install
RUN git clone https://github.com/redis/hiredis.git /hiredis && \
    cd /hiredis && \
    make && \
    make install


#fmi install
#ADD https://google.com cache_bust

ARG FMI_REF=94ccfa1946fe4b02976a3ae64f49d04c343067da

RUN git clone https://github.com/mstaylor/fmi.git /fmi && cd /fmi && git checkout ${FMI_REF}

RUN cd /fmi && \
    cd python && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make

RUN cd $CYLON_HOME


EXPOSE $EXPOSE_ENV

#ENTRYPOINT ["/cylon/aws/scripts/runCylon.sh"]

ENV PYTHONPATH="/fmi/python/build:$CYLON_HOME/aws/scripts:$PYTHONPATH"

ENV LD_LIBRARY_PATH="/fmi/python/build:/opt/conda/envs/cylon_dev/lib:${LD_LIBRARY_PATH}"

RUN chmod 0755 /cylon/aws/scripts/runCyloninLambda.sh

#RUN chmod 0755 /natchecker/client/natcheck
#ADD https://google.com cache_bust

ARG ARROW_VERSION=16.1.0

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    PY_VERSION=$(python -c 'import platform; print(platform.python_version())') && \
    NUMPY_VERSION=$(python -c 'import numpy; print(numpy.__version__)') && \
    conda install -y -c conda-forge \
        "python=${PY_VERSION}" "numpy=${NUMPY_VERSION}" \
        "pyarrow=${ARROW_VERSION}" "libarrow==${ARROW_VERSION}.*" \
        "libarrow-acero==${ARROW_VERSION}.*" "libarrow-dataset==${ARROW_VERSION}.*" \
        glog openmpi mpi4py "cython>=0.29.31" "pandas>=2.0,<2.2.3dev" fsspec && \
    { conda remove -y --force pyopenssl || true; } && \
    pip install pyOpenSSL==23.2.0 --force-reinstall && \
    python -c "import numpy, torch, timm, torchvision; \
assert numpy.__version__ == '${NUMPY_VERSION}', numpy.__version__; \
assert torch.__version__.split('+')[0] == '2.5.0', torch.__version__; \
assert torchvision.__version__.split('+')[0] == '0.20.0', torchvision.__version__; \
assert timm.__version__ == '0.4.12', timm.__version__" && \
    conda clean -afy

ARG REDIS_PLUS_PLUS_REF=1.3.15

RUN git clone https://github.com/sewenew/redis-plus-plus.git /redis-plus-plus && \
    cd /redis-plus-plus && git checkout ${REDIS_PLUS_PLUS_REF} && \
    mkdir build && cd build && \
    cmake -DREDIS_PLUS_PLUS_CXX_STANDARD=11 .. && make && make install && \
    rm -rf /redis-plus-plus

ARG CYLON_MAIN_REF=3526bfbabd8659144b68a9ea274d2d59834e897f
ENV CYLON_MAIN_HOME=/cylon-main
ENV CYLON_MAIN_PREFIX=/cylon-main/install

RUN git clone https://github.com/mstaylor/cylon.git $CYLON_MAIN_HOME && \
    cd $CYLON_MAIN_HOME && git checkout ${CYLON_MAIN_REF} && rm -rf .git

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    cd $CYLON_MAIN_HOME && \
    python build.py \
        -cmake-flags="-DCYLON_USE_REDIS=1 -DCYLON_FMI=1 -DCYLON_SIMD=1 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX" \
        -ipath="$CYLON_MAIN_PREFIX" --cpp --python && \
    rm -rf $CYLON_MAIN_HOME/build $CYLON_MAIN_HOME/python/pycylon/build

ENV CYLON_ARMADA_HOME=/cylon-armada
ENV CYLON_ARMADA_PREFIX=/cylon-armada/install

COPY cpp/CMakeLists.txt $CYLON_ARMADA_HOME/cpp/CMakeLists.txt
COPY cpp/src $CYLON_ARMADA_HOME/cpp/src

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    mkdir -p /tmp/cylon-armada-build && cd /tmp/cylon-armada-build && \
    cmake -DCYLON_PREFIX=$CYLON_MAIN_PREFIX \
          -DCYLON_USE_REDIS=1 \
          -DCYLON_FMI=1 \
          -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
          -DCMAKE_INSTALL_PREFIX=$CYLON_ARMADA_PREFIX \
          -DCMAKE_BUILD_TYPE=Release \
          $CYLON_ARMADA_HOME/cpp && \
    cmake --build . --parallel $(nproc) && \
    cmake --install . && \
    rm -rf /tmp/cylon-armada-build

COPY python/setup.py $CYLON_ARMADA_HOME/python/setup.py
COPY python/cylon_armada $CYLON_ARMADA_HOME/python/cylon_armada

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    cd $CYLON_ARMADA_HOME/python && \
    rm -rf build cylon_armada/*.cpp cylon_armada/*.so cylon_armada/__pycache__ && \
    LD_LIBRARY_PATH=$CYLON_MAIN_PREFIX/lib:$CYLON_ARMADA_PREFIX/lib:/usr/local/lib:$LD_LIBRARY_PATH \
    CYLON_PREFIX=$CYLON_MAIN_PREFIX \
    CYLON_ARMADA_PREFIX=$CYLON_ARMADA_PREFIX \
    CYLON_HOME=$CYLON_MAIN_HOME \
    CYLON_REDIS=1 \
    CYLON_FMI=1 \
    python setup.py build_ext --inplace && \
    rm -rf build

ENV PYTHONPATH="${CYLON_ARMADA_HOME}/python:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="${CYLON_MAIN_PREFIX}/lib:${CYLON_ARMADA_PREFIX}/lib:/usr/local/lib:${LD_LIBRARY_PATH}"

COPY docker/cosmic-ai-lambda/lambda_entry3.py $CYLON_HOME
COPY docker/cosmic-ai-lambda/fmi.json $CYLON_HOME

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    python -c "import mpi4py; mpi4py.rc.initialize = False; mpi4py.rc.finalize = False; \
import torch, timm; \
from pycylon.net.fmi_config import FMIConfig; \
from pycylon.net.reduce_op import ReduceOp; \
from pycylon.frame import CylonEnv; \
from cylon_armada.context_table import ContextTable; \
print('embedding', ContextTable(8).to_arrow().schema.field('embedding').type); \
print('torch', torch.__version__, 'timm', timm.__version__)" && \
    md5sum $CYLON_HOME/lambda_entry3.py $CYLON_HOME/fmi.json

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/cylon/aws/scripts/runCyloninLambda.sh" ]
#ENTRYPOINT ["/lambda-entrypoint.sh"]
# Pass the name of the function handler as an argument to the runtime
CMD [ "python", "-m", "awslambdaric", "lambda_entry3.handler" ]