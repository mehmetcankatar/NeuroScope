import os
from pathlib import Path
import subprocess
import sys


def main():
    # A small dependency-free .env loader; existing environment wins.
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    import uvicorn
    worker = subprocess.Popen([sys.executable, "-m", "neuroscope.worker"])
    try:
        uvicorn.run("neuroscope.api:app", host=os.getenv("NEUROSCOPE_HOST", "127.0.0.1"),
                    port=int(os.getenv("NEUROSCOPE_PORT", "8000")), access_log=False)
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=8)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()


if __name__ == "__main__":
    main()
