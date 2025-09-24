rag-project
├── src
│   ├── api
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── models.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── document_loader.py
│   │   ├── embed_and_index.py
│   │   ├── retriever.py
│   │   └── config.py
│   ├── services
│   │   ├── __init__.py
│   │   └── vector_store.py
│   └── main.py
├── tests
│   ├── __init__.py
│   ├── test_api.py
│   └── test_core.py
├── kubernetes
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── docker
│   └── Dockerfile
├── requirements.txt
├── docker-compose.yml
├── .dockerignore
├── .env.example
└── README.md