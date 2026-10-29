FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.13.5

WORKDIR /app

# Install only the runtime dependencies from the lockfile (no notebook/dev packages)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py model.pkl ./
COPY static/ ./static/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
