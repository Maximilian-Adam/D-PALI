# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src  


# System dependencies (MuJoCo, OpenGL, etc.)

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglew2.1 \
      libosmesa6-dev \
      patchelf \
    && rm -rf /var/lib/apt/lists/*
    
# Install pip requirements

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY src/      ./src/
COPY assets/   ./assets/
COPY scripts/  ./scripts/
COPY training/ ./training/

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["python", "training\train_sac_her.py"]
CMD [bash]