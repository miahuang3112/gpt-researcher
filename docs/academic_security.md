# Academic API security boundary

The Academic API supports two explicit deployment modes.

## Trusted-local mode

`ACADEMIC_AUTH_MODE=trusted_local` is the development default. It performs no
application-level authentication. The supplied Compose file binds API, MySQL
and Redis ports to `127.0.0.1` only. Do not expose these ports through a public
load balancer, reverse proxy, tunnel or shared host in this mode.

## API-key mode

For any shared or remotely reachable deployment, set:

```text
ACADEMIC_AUTH_MODE=api_key
ACADEMIC_API_KEY=<a randomly generated secret>
```

Clients must send the secret in `X-API-Key`. Health endpoints remain
unauthenticated for container orchestration. Evaluation endpoints return a
generic HTTP 401 response for missing or incorrect keys. Keys are represented
with Pydantic `SecretStr` and must never be placed in source control or URLs.

This is service authentication, not multi-user authorization. Before adding
separate users, add identity-aware ownership checks to every job operation.

## Artifact retention and deletion

Defaults are configurable with `ACADEMIC_PDF_RETENTION_DAYS` (30),
`ACADEMIC_CHECKPOINT_RETENTION_DAYS` (7), and
`ACADEMIC_RESULT_RETENTION_DAYS` (90). Retention applies only after a job is
completed or failed. Run the following command from the API runtime on a daily
schedule:

```text
python -m backend.academic_api.services.retention
```

The command removes expired managed files and result payloads but retains the
minimal job audit row. `DELETE /api/academic/evaluations/{job_id}` permanently
deletes a terminal job, its result, outbox rows, job-owned PDF and checkpoint;
it returns 409 for queued/running jobs. File deletion is confined to configured
managed roots, so a legacy or corrupted database path cannot delete arbitrary
server files. Deleted files are not recoverable unless the deployment provides
separate backups.
