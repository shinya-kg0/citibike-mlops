FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*


RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

COPY .bashrc /home/appuser/.bashrc
RUN chown -R appuser:appuser /app /home/appuser/.bashrc

ENV PYTHONUNBUFFERED=1

EXPOSE 8888

USER appuser

CMD ["bash"]