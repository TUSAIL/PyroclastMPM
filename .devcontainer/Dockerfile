FROM ubuntu:22.04

WORKDIR /workspaces

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update --no-install-recommends \
    && apt-get -y install --no-install-recommends \
    build-essential=12.9ubuntu3 \
    curl=7.81.0-1ubuntu1.10 \
    git=1:2.34.1-1ubuntu1.9 \
    vtk9=9.1.0+really9.1.0+dfsg2-3build1 \
    libvtk9-dev=9.1.0+really9.1.0+dfsg2-3build1 \
    libthrust-dev=1.15.0-1 \
    cmake=3.22.1-1ubuntu1.22.04.1 \
    python-is-python3=3.9.2-2 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://get.trunk.io -fsSL | bash -s -- -y


ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# to run poetry directly as soon as it's installed
ENV PATH="$POETRY_HOME/bin:$PATH"

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -