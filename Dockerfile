FROM ubuntu:18.04

RUN ln -sf /usr/share/zoneinfo/EST /etc/localtime && \
    apt-get update && apt-get install -y vim \
    xvfb \
    python \
    python-pip \
    python-tk


# Copy requirements and install before copying the code
WORKDIR /var/rom
COPY requirements.txt /var/rom/requirements.txt
RUN pip install -r requirements.txt

COPY . /var/rom


