ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION

RUN apt-get update -y
# RUN apt-get install -y python3-pip python-dev build-essential
RUN apt-get install -y python3-pip build-essential
RUN apt-get install -y git
RUN apt-get install -y ffmpeg

# RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /

RUN pip3 install -r requirements.txt

# Docker-specific changes
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX;8.9;9.0"
RUN export CUDA_HOME=/usr/local/cuda
# RUN git clone https://github.com/PanQiWei/AutoGPTQ
# RUN pip3 install AutoGPTQ/.
# RUN pip3 install "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.4.2"
# RUN pip3 install auto-gptq==0.4.2
RUN pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
RUN pip3 install soundfile
RUN apt-get install -y libportaudio2
RUN apt-get install -y libsndfile1

# Python version incompatibility
COPY rutts_gruut_fixed.py /usr/local/lib/python3.8/dist-packages/RUTTS/tokenizer/gruut/tokenizer.py
COPY rutts_g2p_fixed.py /usr/local/lib/python3.8/dist-packages/RUTTS/tokenizer/g2p/tokenizer.py

COPY *.py /

WORKDIR /

EXPOSE 7860

# ENTRYPOINT python3 main.py

# docker build -t fttftf_docker:gradio_gpu .
# docker run --gpus all -it -p 7860:7860 fttftf_docker:gradio_gpu
# python3 main.py (optional if entrypoint commented above)