FROM python:3.11-slim
WORKDIR /workspace/training
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt
CMD ["python", "train_temporal.py"]
