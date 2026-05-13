FROM python:3.11-slim
WORKDIR /app
COPY pipeline.py ./pipeline.py
RUN pip install --no-cache-dir numpy pandas scikit-learn tensorflow
ENTRYPOINT ["python", "/app/pipeline.py"]
