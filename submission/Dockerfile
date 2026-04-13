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

COPY submission/assets/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 /home/appuser/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

RUN mkdir -p /home/appuser/output
RUN chmod -R 777 /home/appuser/

SHELL ["/bin/bash", "-c"]

RUN pip3 install --upgrade "pip<21" setuptools wheel
RUN pip3 install numpy
RUN pip3 install "tensorflow==1.15.5"
RUN pip3 install "keras==2.2.4" "h5py<3"
RUN pip3 install matplotlib tqdm scipy Pillow scikit-image imgaug IPython geopandas imantics scikit-learn chainer tensorboard tensorboardX
RUN pip3 install opencv-python-headless==4.3.0.38

COPY . /home/appuser/code/xview-2
RUN chmod +x /home/appuser/code/xview-2/submission/run.sh
RUN chown -R appuser:appuser /home/appuser/code/xview-2

USER appuser

ENTRYPOINT ["/home/appuser/code/xview-2/submission/run.sh", "/home/appuser/code/xview-2/"]