# Stage 1: Build stage
# Build Stage: It installs python3-venv (needed to create a virtual environment) and sets up a working directory.

FROM python:3.11-slim AS build

# Set working directory
WORKDIR /app

COPY . .

# Install dependencies for venv creation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Final image
FROM python:3.11-slim
# Final Stage: Copies only the necessary files (including the Python environment) from the build stage to the final image.

# Set working directory
WORKDIR /app

# Copy only the venv from build stage
COPY --from=build /app /app

# Set environment variable to avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and activate the virtual environment
RUN python3 -m venv /app/venv

#Install dependencies from requirements.txt inside the venv
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#Entrypoint to activate the venv
# ENTRYPOINT ["./init.sh"]
