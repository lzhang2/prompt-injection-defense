# ml-platform

Shared infrastructure for ML apps on this machine: an **MLflow** tracking server and
model registry, and a **Ray cluster** that apps deploy their Ray Serve endpoints onto.
Apps (such as `../genai-safety`) live in their own folders and never run these services
themselves.

```
                 ml-platform network (docker)
 ┌──────────────────────────────────────────────────────────────┐
 │  mlflow  :5000   ◄──── model loads ────  ray-head  :8265 :8000│
 │  data/mlflow/                            (runs the app image) │
 └──────▲───────────────────────────────────────▲──────────────▲─┘
        │ registry / tracking                   │ serve deploy │ HTTP
   app pipeline jobs                       app deploy job    your Mac
   (e.g. genai-safety)                                       127.0.0.1:8100
```

Containers reach each other by **service name** on the `ml-platform` network
(`http://mlflow:5000`, `http://ray-head:8265`). `127.0.0.1` inside a container means
that container itself. App compose projects join the network as `external`.

| Service | On your Mac | On the network |
|---|---|---|
| MLflow UI / API | http://127.0.0.1:5100 | `http://mlflow:5000` |
| Ray dashboard | http://127.0.0.1:8365 | `http://ray-head:8265` |
| Ray Serve (app endpoints) | http://127.0.0.1:8100 | `http://ray-head:8000` |

Ports are bound to `127.0.0.1`, so nothing is exposed to your network. They avoid
5000 (macOS AirPlay Receiver) and the non-Docker MLflow/Ray on 5001, 8000, 8265.

## Start, in order

```bash
cp .env.example .env               # set RAY_IMAGE
cp ../genai-safety/configs/secrets.example.yaml ../genai-safety/configs/secrets.yaml
                                   # fill in gemini_api_key
docker build -t ml-platform/ray-base:2.60.0 ray/

# 1. MLflow
docker compose up -d mlflow

# 2. Build the app image FROM ray-base (in the app's folder)
(cd ../genai-safety && docker compose build)

# 3. Ray cluster, running that app image (no app deployed yet)
docker compose up -d ray-head

# 4. Deploy the app onto the cluster (in the app's folder)
(cd ../genai-safety && docker compose run --rm deploy)
```

## Why the Ray cluster runs an app's image

Ray replicas import the app's code from the image the cluster runs; `serve deploy`
sends only config. So apps build `FROM ml-platform/ray-base`, and `ray-head` runs that
image (`RAY_IMAGE`). With several apps you either build one image with all of them, or
give each app its own Ray cluster, which is what KubeRay `RayService` does on Kubernetes.

App **code** changes therefore mean: rebuild the app image, then
`docker compose up -d ray-head` here (restarts the cluster), then redeploy the app.
Config-only changes (model version, replicas) are just a redeploy from the app.

## Data

- `data/mlflow/`: `mlflow.db` (runs, registry, aliases) and `mlartifacts/` (model files).
- `data/ray-artifacts/`: files apps write at `/app/artifacts` inside the cluster,
  such as request logs.

Both are git-ignored. To import an existing local registry without retraining:

```bash
scripts/seed_mlflow.sh ../genai-safety/mlflow.db ../genai-safety/mlartifacts
```

This works because artifact paths in `mlflow.db` are relative (`mlflow-artifacts:/...`).
It is a snapshot: after this, train and promote against this MLflow only.

## Secrets

API keys live in the app's `configs/secrets.yaml` (git-ignored). `ray-head` mounts it
read-only at `/app/configs/secrets.yaml`, next to the config the app loads, so replicas
read it at startup. Set `APP_SECRETS` in `.env` to use a different file. Keys are never
in an image or in `.env`. Create the file before `up`: if it is missing, Docker
creates an empty directory in its place.

## Versions

`ray/requirements.txt` pins Ray, MLflow, torch and transformers to the versions that
trained the current models; `mlflow/Dockerfile` pins the server to the same MLflow.
Change them together with a retrain.

## Production

Same split, managed pieces: MLflow on Postgres (`--backend-store-uri postgresql://...`)
with S3/GCS artifacts (`--artifacts-destination s3://...`); Ray on Kubernetes with
KubeRay, one `RayService` per app pointing at its image and Serve config; secrets from
a secret manager; app endpoints behind a gateway with auth and TLS.
