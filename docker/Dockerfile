#*******************************************************************************
# libonvif/docker/Dockerfile
#
# Copyright (c) 2022  Vladislav Vlsarro
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#*******************************************************************************/

FROM ubuntu:18.04

# Setup build environment
RUN export DEBIAN_FRONTEND=noninteractive; \
    export DEBCONF_NONINTERACTIVE_SEEN=true; \
    apt-get update -qqy && apt-get install -qqy --option Dpkg::Options::="--force-confnew" --no-install-recommends \
    autoconf automake build-essential pkgconf libtool libxml2-dev git ca-certificates wget && \
    apt-get --quiet autoremove --yes && \
    apt-get --quiet --yes clean && \
    rm -rf /var/lib/apt/lists/*

# install cmake 3.17
RUN wget -qO- "https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

COPY . /libonvif

RUN cd /libonvif  && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    ldconfig

CMD ["/bin/bash"]
