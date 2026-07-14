# One image for both training and serving. The trade-off here is honest:
# the serving image ends up bigger than strictly necessary (it carries
# training-only deps like matplotlib), but for a project this size,
# maintaining two near-identical Dockerfiles is more complexity than it's
# worth. Splitting into a lean inference-only image is a reasonable next
# step if this needed to scale (see README "Future Improvements").
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
