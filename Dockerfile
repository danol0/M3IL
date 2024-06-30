FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y git
WORKDIR /app
RUN git clone https://github.com/danol0/mtpf .
RUN conda env update -f env.yml --name base