FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG PRELOAD_PG="224 448"
ARG BV_SOURCE="lucasb-eyer"
# Alternatively, use google-research for the official one.

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set the working directory
WORKDIR /home

RUN apt-get update && apt install --yes --no-install-recommends rsync tmux fish nvtop vim pipx jq && apt-get clean && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

# The btop on apt is older than the version needed for GPU.
# It currently crashes on >6 GPUs, so we also need nvitop.
RUN apt-get update && apt install --yes --no-install-recommends lowdown && apt-get clean && \
    git clone https://github.com/aristocratos/btop.git && \
    cd btop && make GPU_SUPPORT=true && make install && cd - && \
    pipx install nvitop

# A simple GPU-aware task-spooler (job-queue). Super useful.
RUN git clone https://github.com/justanhduc/task-spooler && \
    cd task-spooler && env CUDA_HOME=/usr/local/cuda make && env CUDA_HOME=/usr/local/cuda make install && cd -

# Let's install everything into an env. JAX and some key packages first, as a checkpoint.
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install -U --no-cache-dir pip && \
    pip install -U --no-cache-dir "jax[cuda12]" && \
    pip install matplotlib pandas xarray

# And now big_vision on top of that.
RUN git clone https://github.com/lucasb-eyer/big_vision.git && \
    . venv/bin/activate && \
    pip install -r big_vision/big_vision/requirements.txt

# Get PaliGemma model via HF
RUN --mount=type=secret,id=HF_TOKEN,env=HF_TOKEN \
  . venv/bin/activate && \
  if [ -n "${HF_TOKEN}" ]; then \
    pip install -U "huggingface_hub[cli]" && \
    if [[ "${PRELOAD_PG}" == *"224"* ]]; then \
      huggingface-cli download google/paligemma-3b-pt-224-jax paligemma-3b-pt-224.npz --local-dir /home --token=${HF_TOKEN} && \
      mv /home/{paligemma-3b-pt-224,pt_224}.npz ; \
    fi ; \
    if [[ "${PRELOAD_PG}" == *"448"* ]]; then \
      huggingface-cli download google/paligemma-3b-pt-448-jax paligemma-3b-pt-448.npz --local-dir /home --token=${HF_TOKEN} && \
      mv /home/{paligemma-3b-pt-448,pt_448}.npz ; \
    fi ; \
    if [[ "${PRELOAD_PG}" == *"896"* ]]; then \
      huggingface-cli download google/paligemma-3b-pt-896-jax paligemma-3b-pt-896.npz --local-dir /home --token=${HF_TOKEN} && \
      mv /home/{paligemma-3b-pt-896,pt_896}.npz ; \
    fi \
  fi

# Copy the README.md
COPY README.md /usr/share/nginx/html/README.md
COPY README.md /home
COPY README.md /root

# We could insert pre_start and post_start scripts here if we needed some.
COPY pre_start.sh /pre_start.sh
# COPY post_start.sh /post_start.sh
RUN chmod +x /pre_start.sh
# RUN chmod +x /post_start.sh

# Set the default command for the container
CMD [ "/start.sh" ]
