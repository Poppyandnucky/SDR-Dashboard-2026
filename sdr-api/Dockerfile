FROM python:3.11-slim

WORKDIR /app

COPY sim/requirements.txt /tmp/sim-requirements.txt
COPY sdr-api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/sim-requirements.txt -r /tmp/api-requirements.txt

COPY sim/ /app/sim/
COPY sdr-api/ /app/sdr-api/

ENV PYTHONPATH=/app/sim
ENV PORT=8000

WORKDIR /app/sdr-api

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
