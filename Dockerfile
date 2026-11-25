FROM mambaorg/micromamba:1.5.8
USER root
WORKDIR /app
COPY environment.yml .
RUN micromamba install -y -n base -f environment.yml && micromamba clean --all --yes
COPY . /app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV MAMBA_DOCKERFILE_ACTIVATE=1
CMD ["micromamba", "run", "-n", "base", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
