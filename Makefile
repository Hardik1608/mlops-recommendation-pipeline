# Convenience targets (Linux/macOS/WSL). On Windows run the commands directly.
PY ?= python
DATASET ?= ml-latest-small

.PHONY: install lint format test test-fast data features train benchmark serve stream drift \
        loadtest up down logs helm-lint promtool clean

install:            ## dev environment (run inside a virtualenv)
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install --no-deps -e .

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:               ## full suite: unit + integration (MLflow, Feast) - synthetic data, no network
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

data:               ## download (checksum-verified) + validate + temporal split
	recsys download --dataset $(DATASET)
	recsys prepare --dataset $(DATASET)

features:           ## batch features -> Feast registry -> online store
	recsys features --dataset $(DATASET)
	recsys feast-apply
	recsys feast-materialize

train:              ## train, evaluate vs baseline, register in MLflow, run the promotion gate
	recsys train --dataset $(DATASET)

benchmark:          ## offline benchmark table -> reports/benchmark_<dataset>.md
	recsys prepare --dataset ml-1m
	recsys benchmark --dataset $(DATASET)
	recsys benchmark --dataset ml-1m

serve:
	recsys serve --host 0.0.0.0 --port 8000

stream:             ## replay the test window through the stream processor (no broker needed)
	recsys stream --dataset $(DATASET)

drift:
	recsys drift --dataset $(DATASET)

loadtest:           ## requires `make serve` in another shell
	$(PY) -m recsys.loadtest --concurrency 32 --duration 30 --label closed-c32
	$(PY) -m recsys.loadtest --rate 300 --duration 30 --label open-300rps

up:                 ## full stack: kafka, redis, mlflow, trainer, api, consumer, producer, monitoring
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f api consumer

helm-lint:
	helm lint deploy/helm/recsys --strict
	helm template ci deploy/helm/recsys > /dev/null

promtool:
	docker run --rm -v "$(CURDIR)/deploy/prometheus:/etc/prometheus" -w /etc/prometheus \
	  --entrypoint promtool prom/prometheus:v3.13.4 test rules alerts_test.yml

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml
