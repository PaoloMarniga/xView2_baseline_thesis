#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          #
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, #
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common curl git apt-transport-https

RUN apt-get update && \
    apt-get install -y python3.6 python3.6-dev python3-pip

RUN ln -sf /usr/bin/python3.6 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

RUN python --version && pip --version

RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser && \
    mkdir -p /home/appuser/.keras/models && \
    chown -R appuser:appuser /home/appuser

ADD https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 /home/appuser/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

RUN chmod -R 777 /home/appuser/

USER appuser

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /home/appuser/code/xview-2
RUN mkdir -p /home/appuser/submission
RUN mkdir -p /home/appuser/output

RUN pip install --upgrade "pip<21" setuptools wheel
RUN pip install numpy
RUN pip install "tensorflow==1.15.5"
RUN pip install matplotlib tqdm scipy Pillow scikit-image opencv-python-headless==4.5.5.64 imgaug IPython geopandas keras imantics scikit-learn chainer tensorboard tensorboardX

RUN git clone https://github.com/DIUx-xView/xview2-baseline.git /home/appuser/code/xview-2

RUN curl -L https://github.com/DIUx-xView/xView2/releases/download/v1.0/localization.h5 -o /home/appuser/code/xview-2/weights/localization.h5
RUN curl -L https://github.com/DIUx-xView/xView2/releases/download/v1.0/classification.hdf5 -o /home/appuser/code/xview-2/weights/classification.hdf5

RUN chmod +x /home/appuser/code/xview-2/submission/run.sh
ENTRYPOINT ["/home/appuser/code/xview-2/submission/run.sh", "/home/appuser/code/xview-2/"]