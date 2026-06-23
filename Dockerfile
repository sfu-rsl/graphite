# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04
# FROM nvidia/cuda:13.0.2-runtime-ubuntu22.04


# Install git, compiler, cmake
RUN apt-get update && apt-get -y install build-essential cmake git git-lfs gdb


# Install other ORB-SLAM3 and CUDA dependencies
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get -y install libopencv-dev libopencv-core-dev libeigen3-dev libboost-serialization-dev libssl-dev 


# Install CUDA Toolkit
# RUN apt-get -y install cuda-toolkit-12-6 cuda-gdb-12-6
RUN apt-get update && apt update && apt-get -y install cuda-toolkit-12-8 cuda-gdb-12-8 cudss-cuda-12
# RUN apt update && apt-get -y install cuda-toolkit-13-0 cuda-gdb-13-0 cudss

# Install Python
RUN apt update && apt-get -y install python3 python3-pip && pip3 install wrenfold

RUN apt update && apt-get -y install clang-format doxygen breathe-doc graphviz