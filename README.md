# MTG Recognition Docker Usage Guide

Run the MTG Recognition Streamlit app locally with Docker.

## Prerequisites

- Install Docker Engine: <https://docs.docker.com/engine/install/>
- Confirm Docker is running:

```bash
docker --version
```

## Quick Start

### 1) Clone the repository

HTTPS (recommended):

```bash
git clone https://github.com/decksray/MTG-Recognition.git
```

SSH (if your SSH keys are configured):

```bash
git clone git@github.com:decksray/MTG-Recognition.git
```

### 2) Enter the project directory

```bash
cd MTG-Recognition
```

### 3) Build the Docker image

Standard build (uses Docker cache for faster rebuilds):

```bash
docker build -t mtg-recognition:local .
```

Optional clean rebuild (slower, useful for troubleshooting):

```bash
docker build --progress=plain --no-cache -t mtg-recognition:local .
```

### 4) Run the container

```bash
docker run --rm -p 8501:8501 --name mtg-recognition mtg-recognition:local
```

### 5) Open the app

Open in your browser:

```text
http://localhost:8501
```

## Notes

- The first startup may take longer while models and services initialize.
- The container runs as a non-root user and includes a Docker healthcheck.

## Useful Commands

View live logs:

```bash
docker logs -f mtg-recognition
```

Check health status:

```bash
docker inspect --format='{{.State.Health.Status}}' mtg-recognition
```

Stop the app:

```bash
docker stop mtg-recognition
```
