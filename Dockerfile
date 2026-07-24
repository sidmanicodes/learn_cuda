FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    wget \
    vim \
    gdb \
    ninja-build \
    openssh-server \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

RUN sed -i \
    -e 's/#\?PermitRootLogin.*/PermitRootLogin yes/' \
    -e 's/#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' \
    -e 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' \
    /etc/ssh/sshd_config

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed --upgrade pip

# Pinned to the latest stable PyTorch (2.13.0) built against CUDA 13.0
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

RUN python3 -m pip install --no-cache-dir --break-system-packages numpy

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

RUN echo 'cd /workspace 2>/dev/null || true' >> /root/.bashrc

WORKDIR /workspace

EXPOSE 22

CMD ["/usr/local/bin/start.sh"]
