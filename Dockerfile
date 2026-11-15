FROM node:24-bookworm-slim AS ui
WORKDIR /app/frontend
RUN npm install -g pnpm@11
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend/ backend/
COPY scripts/ scripts/
ARG INSTALL_OPEN_MODELS=0
RUN if [ "$INSTALL_OPEN_MODELS" = "1" ]; then python -m pip install --no-cache-dir ".[open-models]"; else python -m pip install --no-cache-dir .; fi
COPY --from=ui /app/frontend/dist frontend/dist
RUN useradd --create-home neuroscope && mkdir -p /app/data && chown -R neuroscope:neuroscope /app
USER neuroscope
ENV NEUROSCOPE_HOST=0.0.0.0 NEUROSCOPE_DATA_DIR=/app/data NEUROSCOPE_PROJECT_ROOT=/app PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "neuroscope"]
