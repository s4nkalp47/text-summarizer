# Serves the summarizer API on CPU. The model is downloaded from the Hugging Face Hub at startup
# (set MODEL_ID); weights are never baked into the image.
#
#   docker build -t text-summarizer .
#   docker run -p 7860:7860 -e MODEL_ID=<your-username>/pegasus-samsum text-summarizer
FROM python:3.11-slim

# Hugging Face Spaces runs containers as user 1000, so install and run as that user.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PORT=7860
# Create the app folder as `user` (WORKDIR alone may create it owned by root); the app writes logs/ there.
RUN mkdir -p /home/user/app
WORKDIR /home/user/app

# CPU-only PyTorch (~200 MB instead of several GB of CUDA libraries), same version as requirements.txt.
COPY --chown=user requirements.txt .
RUN pip install --user "$(grep '^torch==' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu

# Only the packages the API needs, at the versions pinned in requirements.txt
# (training/evaluation libraries like datasets, pandas and matplotlib stay out of the image).
RUN grep -E '^(transformers|sentencepiece|protobuf|fastapi|uvicorn|PyYAML|python-box)==' requirements.txt > requirements-serve.txt \
    && pip install --user -r requirements-serve.txt

COPY --chown=user pyproject.toml params.yaml app.py ./
COPY --chown=user config ./config
COPY --chown=user src ./src
COPY --chown=user static ./static
RUN pip install --user --no-deps .

EXPOSE 7860
# Shell form so $PORT is expanded; `exec` makes uvicorn PID 1 so it receives stop signals.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
