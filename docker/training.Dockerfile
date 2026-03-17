ARG         BASE_IMAGE=ubuntu:latest
FROM        ${BASE_IMAGE}

ARG         DEBIAN_FRONTEND=noninteractive

RUN         apt update && \
            apt install --no-install-recommends -y \
                ca-certificates \
                git \
                curl \
                wget \
                htop \
                nano \
                build-essential \
                cmake \
                libreadline-dev \
                libncursesw5-dev \
                libssl-dev \
                libsqlite3-dev \
                libgdbm-dev \
                libc6-dev \
                libbz2-dev \
                libffi-dev \
                libpq-dev \
                liblzma-dev \
                libopenblas-dev \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                tk-dev \
                && \
            apt clean && \
            rm -rf /var/lib/apt/lists/*

RUN         wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
            dpkg -i cuda-keyring_1.1-1_all.deb && \
            apt update && \
            apt install --no-install-recommends -y cuda-compiler-12-8 cuda-libraries-dev-12-8 && \
            apt clean && \
            rm -rf /var/lib/apt/lists/* && \
            rm cuda-keyring_1.1-1_all.deb

# Build and install VVenc
ARG         VVENC_VERSION=v1.13.1
RUN         git clone --depth 1 --branch ${VVENC_VERSION} https://github.com/fraunhoferhhi/vvenc.git /tmp/vvenc && \
            cd /tmp/vvenc && \
            echo "Cloned commit: $(git log -1 --oneline)" && \
            echo "Tag: $(git describe --tags --exact-match || echo 'No exact tag')" && \
            cmake -S . -B build/release-static -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
            cmake --build build/release-static -j"$(nproc)" && \
            cmake --build build/release-static --target install && \
            cd / && rm -rf /tmp/vvenc

# Build and install VVdec (with vvdecapp)
ARG         VVDEC_VERSION=v3.0.0
RUN         git clone --depth 1 --branch ${VVDEC_VERSION} https://github.com/fraunhoferhhi/vvdec.git /tmp/vvdec && \
            cd /tmp/vvdec && \
            echo "Cloned commit: $(git log -1 --oneline)" && \
            echo "Tag: $(git describe --tags --exact-match || echo 'No exact tag')" && \
            cmake -S . -B build/release-static -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX=/usr/local \
                -DVVDEC_INSTALL_VVDECAPP=ON && \
            cmake --build build/release-static -j"$(nproc)" && \
            cmake --build build/release-static --target install && \
            cd / && rm -rf /tmp/vvdec

ARG         VERSION_PYTHON
RUN         git clone "https://github.com/pyenv/pyenv.git" ./pyenv && \
            (cd pyenv/plugins/python-build && ./install.sh) && \
            rm -rf pyenv

RUN         python-build --no-warn-script-location ${VERSION_PYTHON} /opt/python
ENV         PATH="/opt/python/bin:${PATH}"

RUN         python -m pip install --no-cache-dir --upgrade pip
RUN         python -m pip install --no-cache-dir --upgrade poetry

COPY        pyproject.toml .
COPY        poetry.lock .
RUN         poetry config virtualenvs.create false --local
RUN         --mount=type=secret,id=PYPI_USERNAME,env=POETRY_HTTP_BASIC_PRIVATE_USERNAME \
            --mount=type=secret,id=PYPI_PASSWORD,env=POETRY_HTTP_BASIC_PRIVATE_PASSWORD \
            poetry install --only main --no-root --no-interaction --extras extras && \
            rm -r poetry.toml pyproject.toml poetry.lock /root/.cache
RUN         python -m pip install --no-cache-dir --no-build-isolation --upgrade flash-attn==2.6.3

ENV         PYTHONUNBUFFERED=1
ENTRYPOINT  ["mlflow", "run", "--env-manager", "local"]
