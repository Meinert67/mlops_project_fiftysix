# Change from latest to a specific version if your requirements.txt
FROM python:3.12.8-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY models models/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENV PYTHONPATH="/src/project"

CMD ["uvicorn", "src.project.api:app", "--host", "127.0.0.1", "--port", "80"]
# ENTRYPOINT ["uvicorn", "src/project/api:app", "--host", "0.0.0.0", "--port", "8000"]
