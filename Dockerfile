FROM python:3.12-slim

RUN pip install --no-cache-dir "uv>=0.11,<0.12"

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"

# Dependencies first so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

# One image, three roles: api (default), mcp and ui override the command in docker-compose.yml.
EXPOSE 8000 8001 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
