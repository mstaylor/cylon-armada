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

WORKDIR $CYLON_HOME


RUN git clone https://github.com/mstaylor/cylon.git $CYLON_HOME

RUN cd $CYLON_HOME && git checkout -b fmi-ds5110-support origin/fmi-ds5110-support

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


RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git && cd aws-sdk-cpp && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3" -DBUILD_SHARED_LIBS="off" .. && make -j6 install && cd ../.. && rm -rf aws-sdk-cpp


#REDIS install
RUN git clone https://github.com/redis/hiredis.git /hiredis && \
    cd /hiredis && \
    make && \
    make install


#fmi install
#ADD https://google.com cache_bust

RUN git clone https://github.com/mstaylor/fmi.git /fmi

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

COPY lambda_entry3.py $CYLON_HOME
COPY fmi.json $CYLON_HOME

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/cylon/aws/scripts/runCyloninLambda.sh" ]
#ENTRYPOINT ["/lambda-entrypoint.sh"]
# Pass the name of the function handler as an argument to the runtime
CMD [ "python", "-m", "awslambdaric", "lambda_entry3.handler" ]