FROM ubuntu:22.04

WORKDIR /workspaces

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get -y update --no-install-recommends \
    && apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    git \
    vtk9 \
    libvtk9-dev \
    libthrust-dev \
    cmake \
    python-is-python3 \
    && apt-get autoremove -y \
    && apt-get clean -y


ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# to run poetry directly as soon as it's installed
ENV PATH="$POETRY_HOME/bin:$PATH"

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
