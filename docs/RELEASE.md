# CrossMind Release & Versioning Notes

## 1. Versioning Scheme

CrossMind uses **Semantic Versioning** (SemVer) with the format `MAJOR.MINOR.PATCH`.

- **MAJOR**: Incompatible API changes or architectural shifts
- **MINOR**: New functionality in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes

### Current Version

```
VERSION = "1.0.0"  (defined in config.py)
```

> **Note**: The version string in `config.py` does not necessarily match the project's release cadence. This is documentation debt.

## 2. Component Versions

| Component | Version Source | Current Value |
|-----------|---------------|---------------|
| Project | `config.py:VERSION` | `1.0.0` |
| Frontend | `frontend/package.json:version` | `1.0.0` |
| Qdrant | `docker-compose.yml` | `qdrant/qdrant:v1.7.4` |
| Neo4j | `docker-compose.yml` | `neo4j:5.18` |
| Jaeger | `docker-compose.yml` | `jaegertracing/all-in-one:1.57` |
| Prometheus | `docker-compose.yml` | `prom/prometheus:v2.53.1` |
| Grafana | `docker-compose.yml` | `grafana/grafana:11.1.0` |
| Python | `Dockerfile` | `3.10-slim` |

## 3. Release Checklist

### Pre-Release

- [ ] Run full test suite: `python -m pytest tests` (147 passed)
- [ ] Run packaging validation: `python -c "import tomli; tomli.load(open('pyproject.toml','rb'))"`
- [ ] Run YAML validation: `python scripts/validate_yaml.py .github/workflows monitoring kubernetes helm`
- [ ] Run security check: `python scripts/security_check.py --skip-pip-audit`
- [ ] Compile sources: `python -m compileall app ingestion reasoning vector_store dashboard scripts migrations`
- [ ] Verify health check passes: `python health_check.py`
- [ ] Update `config.py` version if bumping
- [ ] Update `frontend/package.json` version if frontend changes
- [ ] Review `docker-compose.yml` for image tag updates
- [ ] Validate `monitoring/grafana/provisioning/dashboards/json/crossmind-overview.json` queries against live metrics

### Release

- [ ] Tag release in Git: `git tag v1.X.Y && git push --tags`
- [ ] Build and push Docker images:
  ```bash
  docker build -t crossmind/api:1.X.Y .
  docker build -f Dockerfile.dashboard -t crossmind/dashboard:1.X.Y .
  ```
- [ ] Deploy Kubernetes manifests with new image tags
- [ ] Verify HPA scaling behavior under load
- [ ] Confirm Grafana dashboards render correctly

### Post-Release

- [ ] Monitor Prometheus alerts for 24 hours
- [ ] Validate Jaeger traces for new request patterns
- [ ] Check DLDB growth and retention policies
- [ ] Review query latency p95 in Grafana

## 4. Feature Flag Matrix

The following flags control feature availability. They are defined in `config.py`.

| Flag | Default | Type | Impact if disabled |
|------|---------|------|-------------------|
| `REACT_UI_ENABLED` | `False` | UI | React frontend unavailable |
| `VLLM_ENABLED` | `False` | Model | vLLM serving disabled; falls back to simulator |
| `NEO4J_ENABLED` | `False` | Infra | GraphRAG uses in-memory fallback |
| `ASYNC_INGESTION_ENABLED` | `False` | Infra | No Celery background tasks |
| `SCALLOP_ENABLED` | `False` | Reasoning | Scallop reasoner skipped |
| `DEFORESTVIS_ENABLED` | `False` | Reasoning | Deforest visualization skipped |
| `MULTIVECTOR_SEARCH_ENABLED` | `False` | Retrieval | Multi-vector search disabled |
| `SPARSE_VECTOR_ENABLED` | `False` | Retrieval | Sparse vector params disabled in Qdrant |
| `TENSOR_3D_INDEXING_ENABLED` | `False` | Retrieval | 3D tensor indexing disabled |
| `COLBERT_RERANKING_ENABLED` | `False` | Retrieval | ColBERT reranking skipped |
| `SSE_COMPRESSION_ENABLED` | `False` | API | SSE responses not compressed |
| `DLDB_BACKUP_ENABLED` | `False` | Storage | No automatic DLDB backups |
| `DASHBOARD_AUTH_ENABLED` | `False` | UI | Dashboard auth gate skipped |
| `RBAC_ENABLED` | `True` | Security | Role-based access control disabled |
| `OPENTELEMETRY_ENABLED` | `True` | Observability | No trace export |
| `PROMETHEUS_ENABLED` | `True` | Observability | No metrics exposure |
| `HOT_CACHE_ENABLED` | `True` | Performance | In-memory hot cache disabled |
| `DISK_CACHE_ENABLED` | `True` | Performance | Disk-based query cache disabled |
| `ACTIVE_LEARNING_ENABLED` | `True` | Learning | Active learning loop disabled |
| `PLUGIN_AUTO_DISCOVER_ENABLED` | `True` | Extensibility | Plugin discovery disabled |
| `REQUEST_DEDUPLICATION_ENABLED` | `True` | Performance | Duplicate request suppression disabled |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Resilience | Circuit breaker sensitivity changed |
| `USE_LOCAL_SIMULATOR_FALLBACK` | `True` | Resilience | Simulator fallback disabled |

## 5. Breaking Changes Policy

### Version Bump Rules

1. **PATCH**: Bug fixes, documentation updates, dependency patches
2. **MINOR**: New feature flags, new optional modules, new test cases, backward-compatible API additions
3. **MAJOR**: 
   - Removing or renaming public modules
   - Changing Pydantic schema field names or types
   - Changing default behavior of existing feature flags
   - Removing support for Python 3.10

### Deprecation Process

1. Announce deprecation in release notes
2. Add `DeprecationWarning` in code
3. Keep deprecated path functional for 2 minor releases
4. Remove in next major version

## 6. Dependency Management

- Python dependencies: `requirements.txt`
- Frontend dependencies: `frontend/package.json`
- Docker base images: pinned in `docker-compose.yml` and `Dockerfile`

> **Note**: `requirements.txt` is the single source of truth for Python dependencies. Do not edit CI/security or packaging files per project constraints.

## 7. Known Limitations Affecting Releases

1. **Empty packages**: `core/`, `infra/`, `services/` are structural placeholders (only `__init__.py`). Releases that depend on these packages will fail at import time until populated.
