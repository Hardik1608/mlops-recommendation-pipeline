# Design document — MLOps recommendation pipeline

> Scope: a movie recommender (MovieLens) built the way a production recommender is built —
> streaming ingestion, a feature store, tracked + gated models, an online API with A/B routing,
> monitoring and Kubernetes deployment. Numbers quoted here were measured in this repo; the
> commands that reproduce them are in the [README](../README.md#results). Capacity figures that
> are *estimates* are labelled as such.

---

## 1. Requirements

**Functional**

| # | Requirement | Where |
|---|---|---|
| F1 | `GET /recommendations/{user_id}?k=10` returns a ranked list (id, title, genres, score) | `serving/app.py` |
| F2 | A user's new interactions influence recommendations within seconds, without retraining | stream processor → Feast push → fold-in |
| F3 | Unknown users get sensible results (cold start) | fold-in from streamed events, else popularity |
| F4 | Never re-recommend what the user already consumed | training history + streamed events masked |
| F5 | Safe model rollout: offline gate, online A/B, one-command promotion and rollback | MLflow aliases + hash split |
| F6 | Daily retraining and drift-triggered retraining | Airflow DAGs |

**Non-functional**

| # | Requirement | Design response |
|---|---|---|
| N1 | P95 < 100 ms inside the service | in-process scoring (no network hop for the model), Redis cache, alert on P95 > 100 ms |
| N2 | Horizontal scalability | stateless API pods, HPA on CPU, one uvicorn worker per pod |
| N3 | Graceful degradation | every dependency (cache, feature store, registry) is optional at request time |
| N4 | Reproducibility / lineage | data checksum, split cutoffs, params, metrics, code all attached to the MLflow version |
| N5 | No train/serve skew on the evaluation | strict global temporal split; the same `recommend()` code path is used offline and online |

---

## 2. Architecture

Two planes that share the model registry and the feature store:

* **Offline / batch** (Airflow, daily): ingest → validate → temporal split → batch features →
  Feast `apply` + `materialize` → train (implicit ALS) → evaluate vs popularity baseline →
  log + register in MLflow → **gate** (assign `challenger` alias, or bootstrap `champion`).
* **Online / streaming**: rating events → Kafka `ratings.v1` (key = `user_id`) → stream
  processor → (a) Feast push → Redis online store (`user_realtime` view), (b) Parquet event
  lake partitioned by date, (c) cache invalidation. The API reads the online store on every
  cache miss.

```
             +---------- Airflow (daily) ----------+         +-- Airflow (hourly) --+
raw data --> | validate -> split -> features -> train|--MLflow-| drift (PSI) -> retrain|
             +-------------------|-----------------+ registry +----------|-----------+
                                 v (materialize)       |  aliases        v Pushgateway
events --> Kafka ratings.v1 --> consumer --> Feast/Redis online store    Prometheus/Grafana
                                    |                  ^    |                 ^
                                    v                  |    v                 |
                              Parquet event lake   API pods (FastAPI) --------+
                                                  (A/B hash, cache, fold-in)
```

(The README has the same picture as a Mermaid diagram.)

---

## 3. Request path (serving)

1. **Variant**: `bucket = sha256(salt:user_id)[:8] / 2^64`; `challenger` if `bucket < share`.
   Sticky per user, stateless, re-randomised per experiment by changing the salt.
2. **Cache**: Redis hash `recs:v1:<user_id>`, field `<variant>:<model_version>:<k>`, TTL 300 s.
   The model version in the field means a promotion never serves stale lists; the stream
   processor deletes the user's key when new events arrive (event-driven invalidation).
   A circuit breaker skips Redis for 30 s after an error, so an outage costs one timeout.
3. **Real-time features**: `user_realtime` from Feast (last 20 interactions, ratings, event
   times). Only events *newer than the model's training cutoff* are used as context, so
   retrained models do not double-count history.
4. **Scoring** (in-process, numpy):
   * known user, no new events → `u · Vᵀ` (batch embedding)
   * known user + new events → fold-in: re-solve the user's ALS normal equations over
     history + new events (one k×k solve) → `personalized_realtime`
   * unknown user + events → fold-in from the events only → `fold_in`
   * otherwise → popularity (`popularity`)
   * hybrid: per-user max-normalised ALS score + β · log-popularity prior with a short
     half-life ("trending now"); seen items are masked; `argpartition` top-k.
5. **Response** is cached; Prometheus records latency (by route + variant), cache result,
   strategy mix, cold starts, model compute time and loaded model versions.

## 4. Streaming path

* **Contract**: `RatingEvent` v1 (pydantic). Major version in the topic name; additive optional
  fields allowed (consumers ignore unknown fields). Invalid payloads go to `ratings.v1.dlq`.
* **Ordering**: key = `user_id` → all events of a user land in one partition → one consumer
  owns a user's state (keyed state, like Flink) → no cross-consumer races on `recent_items`.
* **Delivery**: idempotent producer (`acks=all`); consumer commits offsets manually *after*
  the online-store write and the Parquet flush → at-least-once. Event ids are UUIDv5 of
  `(user, item, timestamp)`, so replays and redeliveries are de-duplicated (tested).
* **State recovery**: after a restart/rebalance the aggregator bootstraps a user's state from
  the online store the first time it sees them.
* **Late / out-of-order events** are inserted in event-time order; the newest N are kept.

## 5. Model choice

| | Explicit ALS (biased MF) | Implicit ALS (Hu et al. 2008) | Hybrid (served) | SASRec (shadow) |
|---|---|---|---|---|
| Objective | squared error on ratings | confidence-weighted preference | implicit ALS + trending prior | next-item cross-entropy over the catalogue |
| Good at | RMSE | ranking | ranking under temporal drift | next item of a session; catalogue coverage |
| Cold start | bias fallback | fold-in from events | fold-in from events | reads the streamed events as a sequence |
| Trained / served on | CPU | CPU | CPU | GPU (PyTorch) / CPU (ONNX Runtime) |

Measured: explicit ALS has the best RMSE but by far the worst NDCG — optimising rating error
does not optimise ranking. Implicit ALS uses a conjugate-gradient solver (3 CG steps per
row, Takács et al. 2011) implemented with batched numpy; it was ~4× faster than the exact
normal-equation solve on this machine and is tested against it.

**SASRec** (Kang & McAuley 2018) is a causal transformer over each user's time-ordered
interactions, trained with full-softmax cross-entropy (Klenitskiy & Vasilev 2023), computed in
position slices with activation recomputation so 59k-movie catalogues fit in 12 GB. Serving
does not depend on PyTorch: the trained network is exported to ONNX, checked against PyTorch
on real sequences, and stored inside the model package; the API scores it with ONNX Runtime.
Real-time events are appended to the user's sequence, so streaming needs no fold-in.

## 6. Evaluation protocol (and why it matters)

* **Global temporal split** (80/10/10 by timestamp): every training event precedes every test
  event. Hyper-parameters are selected on train→val only; the reported model is fit on
  train+val and evaluated once on test.
* Users are reported in **warm** (seen in training) and **cold** segments. On
  ml-latest-small 65% of test users are cold — a random split would hide this completely.
* **Protocol check**: on a *random* 80/20 split, implicit ALS beats popularity by a large
  factor; on the temporal split the gap shrinks sharply and a tuned (time-decayed) popularity
  baseline becomes hard to beat. The README table has both numbers.
* **Leave-one-out check** (`benchmark --loo`): each user's last event is the test item. It
  rewards predicting the next rating of a session, and on MovieLens it ranks SASRec 1.6-4.9×
  above ALS (ml-25m / ml-1m), the opposite of the temporal split. A control with shuffled histories
  (`--order-ablation`) shows the advantage is the order of ratings, which on MovieLens is
  largely set by the site's interface. The served model is chosen on the temporal split only.
* **Significance**: model comparisons are paired over the same returning test users
  (bootstrap over users, 2,000 resamples), reported with 95% intervals. New users are left out
  of the comparison because every model gives them the same popularity fallback.

## 7. Model lifecycle (A/B testing)

```
train -> offline gate --(no champion)-------> champion  (bootstrap)
                      --(beats champion)----> challenger -> X% of users (hash split)
                      --(otherwise)---------> rejected  (tagged with the reason)
challenger --recsys promote--> champion (old champion -> previous_champion)
champion   --recsys rollback-> previous_champion
```

* The gate re-evaluates the **current champion on the same test window** as the candidate,
  so a champion trained on older data is compared like for like.
* API pods poll the aliases (`RECSYS_MODEL_REFRESH_SECONDS`) and hot-swap atomically —
  promotion and rollback need no redeploy.
* Online decision: compare per-variant business metrics (CTR / watch-through, not available
  offline in MovieLens) with a fixed sample size or a sequential test; guardrails are the
  per-variant latency/error panels and the `RecsysChallengerLatencyRegression` alert.
* **Shadow** (`recsys shadow --version N`, alias `shadow`): a candidate that lost the gate, or
  any model whose serving cost is unknown, is scored on a sticky share of uncached requests
  (`RECSYS_SHADOW_TRAFFIC`, default 20%) in a background thread after the response is built.
  Its compute latency (`recsys_model_compute_seconds{variant="shadow"}`) and its overlap with
  the served top-k (`recsys_shadow_overlap_ratio`) are recorded; failures are counted
  (`recsys_shadow_requests_total{result="error"}`) and never reach the user. Shadow scoring
  uses the events newer than the *shadow* model's own training cutoff.

## 8. Failure modes

| Failure | Effect | Mitigation in this design |
|---|---|---|
| Redis (cache) down | cache misses | circuit breaker → served uncached; `RecsysCacheDegraded` alert |
| Online feature store down | no real-time context | request still served from batch embedding / popularity; `recsys_feature_store_errors_total` alert |
| MLflow down | running pods unaffected (models in memory) | hot-swap poller keeps current models; new pods fail the startup probe instead of serving nothing |
| Kafka down | real-time features go stale | serving unaffected; producer retries (idempotent); consumer resumes from committed offsets |
| Consumer crash mid-batch | redelivery | at-least-once + UUIDv5 de-dup; state bootstrapped from the online store |
| Poison message | — | schema validation → DLQ topic, consumer continues |
| Bad data upstream | bad model | validation task fails the DAG before training |
| Bad model | quality regression | offline gate; challenger only gets 10%; rollback is an alias move |
| Data drift | silent quality decay | hourly PSI job → Pushgateway → alert → retraining DAG triggered |
| Traffic spike | latency | HPA (scale up 100%/30 s), PDB keeps ≥1 pod during node drains |

## 9. Capacity estimate (back-of-envelope)

*Assumptions (illustrative, not measured):* 1M daily active users, 20 recommendation requests
and 50 interaction events per user per day, peak = 5× average.

| Quantity | Estimate |
|---|---|
| Requests | 20M/day ≈ 230 req/s average, ≈ 1,150 req/s peak |
| API pods | measured: one worker saturates at ~656 req/s on a laptop core (batch path); at the 60% HPA target ≈ 390 req/s per pod → **3 pods** for 1,150 req/s, plus headroom. With the Feast SDK lookup on every request (~317 req/s per worker) ≈ 6 pods — re-measure on the production node type and Redis |
| Events | 50M/day ≈ 580 events/s average, ≈ 2,900/s peak; ~250 B each ≈ 0.7 MB/s peak |
| Kafka | 6 partitions are ample (a partition handles MB/s); 7-day retention ≈ 90 GB |
| Online features | 20 recent items × (8 + 4 + 8 B) + overhead ≈ 1 KB/user → ~1 GB for 1M users |
| Cache | ~2 KB per cached list; with TTL 300 s only recently active users are resident |
| Model in memory | items × factors × 4 B: 10k × 64 → 2.5 MB; user embeddings 1M × 64 × 4 B = 256 MB |

## 10. What changes at 10× traffic (and 100× catalogue)

1. **API**: HPA already scales pods horizontally; at 10× the bottleneck is CPU per request.
   Keep one worker per pod; move JSON serialisation to `orjson`; raise cache hit-rate with
   TTL jitter and per-user precomputed top-N for heavy users.
2. **Scoring**: brute-force `u·Vᵀ` is O(items × k) per request (fine for 10k items). At
   100k–1M items switch to **two-stage retrieval**: ANN candidate generation (FAISS/ScaNN/HNSW
   over item factors, ~500 candidates) → **learning-to-rank** stage (e.g. LightGBM) using the
   `user_stats`/`item_stats` Feast views that are already registered and materialized.
3. **User embeddings** no longer fit comfortably in every pod at 10M+ users → store them in the
   online store (or compute by fold-in from history at request time, already implemented).
4. **Streaming**: more partitions + consumers (one per partition); move windowed aggregates
   (e.g. trending counts that feed the popularity prior in real time) to Flink / Spark
   Structured Streaming.
5. **Stores**: Redis Cluster, separate clusters for cache and features (different eviction
   policies — cache `volatile-lru`, features never evicted); Feast SQL registry on Postgres.
6. **MLflow**: Postgres backend + object-storage artifacts; models pulled once per rollout.
7. **Training**: Spark MLlib ALS or GPU `implicit` for 25M+ interactions; incremental
   retraining warm-started from the previous factors.
8. **Experimentation**: an experiment service (assignment + exposure logging) instead of a
   single salt; sequential testing so experiments can stop early.

## 11. Security & operations

* Containers run as non-root; Helm sets `readOnlyRootFilesystem`, drops all capabilities,
  `seccompProfile: RuntimeDefault`; secrets (Feast registry URL, credentials) only via an
  existing Kubernetes Secret, never in values or images.
* JSON logs in containers; Prometheus metrics with bounded label cardinality (route templates,
  two variants).
* CI: ruff, pytest (unit + integration with MLflow and Feast), end-to-end smoke (CLI → API →
  load test), Docker build + hadolint, compose validation, Helm lint + kubeconform, promtool
  rule unit tests, Airflow DAG integrity.
