"""FastAPI serving layer.

Only /health exists so far. /predict and /hotspots are intentionally not stubbed:
a stub returning invented risk scores is indistinguishable from a working endpoint
until someone trusts it. They get built once a model is trained and its output
shape is settled.
"""

from fastapi import FastAPI

app = FastAPI(
    title="RoadRisk TH API",
    description="Accident risk prediction for Thailand.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# TODO: GET /hotspots - top-N risky H3 cells for a given time bucket.
# TODO: POST /predict - severity probability for a set of conditions.
# Both need pydantic request models and input validation before use.
