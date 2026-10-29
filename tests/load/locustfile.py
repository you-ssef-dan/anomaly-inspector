"""Load test: simulated cameras sending part photos to a running API.

  uv run uvicorn anomaly_inspector.serving.app:app --port 8000      # in another terminal
  uv run locust -f tests/load/locustfile.py --host http://localhost:8000 \
      --headless -u 10 -r 2 -t 2m --csv reports/load/<setting>

Each user is one camera: it sends a holdout photo of the category the API serves, then waits
0.5 to 1.5 s, like a part arriving on a line about once a second.
"""

import random
from pathlib import Path

import pandas as pd
import requests
from locust import HttpUser, between, events, task
from locust.env import Environment

MANIFEST = Path("data/processed/manifest.parquet")
PHOTOS: list[tuple[str, bytes]] = []  # (file name, PNG bytes), read once for all users


@events.test_start.add_listener  # type: ignore[untyped-decorator]
def load_photos(environment: Environment, **_: object) -> None:
    """The holdout photos (good and defective) of the category the API serves."""
    category = requests.get(f"{environment.host}/v1/model", timeout=10).json()["category"]
    rows = pd.read_parquet(MANIFEST)
    rows = rows[(rows["category"] == category) & (rows["split"] == "holdout")]
    PHOTOS[:] = [(Path(p).name, Path(p).read_bytes()) for p in rows["path"]]


class Camera(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def inspect_a_part(self) -> None:
        name, photo = random.choice(PHOTOS)
        self.client.post("/v1/predict", files={"file": (name, photo, "image/png")})
