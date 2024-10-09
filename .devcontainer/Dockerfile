FROM  mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

RUN apt-get update && \  
    apt-get install -y curl && \
    apt-get clean

RUN python3 -m pip install --user pipx && \
    python3 -m pipx ensurepath

RUN  pipx ensurepath --global

RUN pipx install poetry

RUN poetry completions bash >> ~/.bash_completion