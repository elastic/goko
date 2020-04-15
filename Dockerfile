FROM rust:slim 

RUN rustup override set nightly 

# install dependencies
RUN (apt-get update 2>&1; exit 0) \
  && apt-get install --no-install-recommends --yes curl unzip \
  && PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip \
  && curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$PROTOC_ZIP \
  && unzip -o $PROTOC_ZIP -d /usr/local bin/protoc \
  && unzip -o $PROTOC_ZIP -d /usr/local 'include/*' \
  && rm -f $PROTOC_ZIP \
  && rm -rf /var/lib/apt/lists/*
# (last line can reduce image size after an apt-get update)

ADD . /home/grandma
WORKDIR /home/grandma

RUN cargo build --release

# docker build -t grandma .
# docker run -it grandma
CMD /bin/bash
