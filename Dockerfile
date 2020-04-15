FROM rust:slim

RUN rustup override set nightly && rustup update 

ADD . /home/grandma
WORKDIR /home/grandma

# install dependencies
RUN apt-get update \
  && apt-get install --no-install-recommends --yes curl unzip python3.7-minimal python3-pip
  && PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip
  && curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$PROTOC_ZIP
  && unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
  && unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
  && rm -f $PROTOC_ZIP
  && rm -rf /var/lib/apt/lists/*
# (last line can reduce image size after an apt-get update)

# build the base lib
RUN cargo clean && cargo build

WORKDIR /home/grandma/pygrandma

# install python dependencies
RUN pip3 install --no-cache-dir -v -e .
  && pip3 install --no-cache-dir numpy scikit-learn 

# docker build -t grandma .
# docker run -it grandma
