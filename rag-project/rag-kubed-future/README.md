# README.md

# RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI and a vector database. The main goal is to efficiently embed and index documents for quick retrieval and response generation.

## Features

- FastAPI for building the API
- Integration with a vector database for efficient nearest neighbor search
- Document chunking and embedding using Sentence Transformers
- Kubernetes deployment configuration for container orchestration
- Docker support for easy containerization

## Directory Structure

- `src/api`: Contains the API routes and models.
- `src/core`: Core functionalities including document processing and embedding.
- `src/services`: Services for interacting with the vector database.
- `tests`: Unit tests for the application.
- `kubernetes`: Kubernetes configuration files for deployment.
- `docker`: Dockerfile for building the application image.

## Requirements

- Python 3.11
- FastAPI
- Sentence Transformers
- FAISS
- Other dependencies listed in `requirements.txt`

## Getting Started

1. Clone the repository.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   uvicorn src.main:app --reload
   ```

## Docker

To build and run the application using Docker:
1. Build the Docker image:
   ```
   docker build -t rag-project .
   ```
2. Run the Docker container:
   ```
   docker run -p 8000:8000 rag-project
   ```

## Kubernetes

To deploy the application on Kubernetes, apply the configuration files in the `kubernetes` directory:
```
kubectl apply -f kubernetes/
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.