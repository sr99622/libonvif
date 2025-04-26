FROM ubuntu:24.10
RUN apt-get update
RUN apt-get install -y wget cmake g++ git python3.12-dev python3-pip virtualenv libxml2-dev libavdevice-dev libsdl2-dev '^libxcb.*-dev' libxkbcommon-x11-dev

RUN wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.8.3/intel-igc-core-2_2.8.3+18762_amd64.deb
RUN wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.8.3/intel-igc-opencl-2_2.8.3+18762_amd64.deb
RUN wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-level-zero-gpu-dbgsym_1.6.32961.7_amd64.ddeb
RUN wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-level-zero-gpu_1.6.32961.7_amd64.deb
RUN wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-opencl-icd-dbgsym_25.09.32961.7_amd64.ddeb
RUN wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-opencl-icd_25.09.32961.7_amd64.deb
RUN wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/libigdgmm12_22.6.0_amd64.deb

RUN dpkg -i *.deb
RUN /bin/bash -c "rm *deb"
RUN virtualenv onvif-gui-env
RUN /bin/bash -c "source /onvif-gui-env/bin/activate && pip install --upgrade pip && pip install onvif-gui openvino && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
EXPOSE 5000
CMD ["/onvif-gui-env/bin/onvif-gui"]

# these work with ubuntu 22.04
# RUN wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb
# RUN wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb
# RUN wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb
# RUN wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb
# RUN wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb
# RUN wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb
# RUN wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb
