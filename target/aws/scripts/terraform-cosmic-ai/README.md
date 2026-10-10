# Cosmic AI Terraform migration

**Parked.** This module targets a separate migration-destination account
(`cosmicai` profile, 881908115028) — the intent was to replicate the
click-ops `DataParallel-CosmicAI` state machine there. That account's IAM
user only has `PowerUserAccess`, which deliberately excludes
`iam:CreateRole`, so it cannot provision the roles this module needs, and
no existing role in that account is trusted by `lambda.amazonaws.com`/
`states.amazonaws.com` either. Running the actual Cosmic AI experiments
went to the main account (448324707516) instead, as an addition to
`../terraform/` (the main cylon-armada module) rather than this one — see
that module's own resources for the live path. This module is kept for if
the cosmicai account's IAM restrictions are ever lifted, or for migrating
to some other account in the future.

Independent of cylon-armada's own `../terraform/` module — apply/destroy
separately. Covers only the published Arm A baseline (S3-only); the FMI
direct-communication Arm B (`initializer_FMI.py`/`inference_FMI.py` in
AI-for-Astronomy) is separate, unfinished work and is not covered here.

## Architecture, as actually deployed (not as it first appeared)

Three Lambdas, driven by a Step Functions Distributed Map:

- **init** (`initializer.py`) and **summarize** (`summarizer.py`) are plain
  boto3 `lambda_handler` functions. Zip-packaged on the native Python
  runtime — no container image, no build step. (An earlier draft of this
  module assumed a shared container image for all three functions; that was
  wrong — these two have no dependency beyond what the Lambda Python
  runtime already bundles.)

- **executor** (invoked by the state machine as `cosmic-executor`) does
  *not* run `inference.py` as its handler. Its real entry point is
  `lambda_entry3.handler`
  (`docker/cosmic-ai-lambda/Dockerfile` in this repo, replicated from
  `cylon/docker/aws/lambda/cosmi-ai/Dockerfile`'s `fmi-ds5110-support`
  branch lineage) — a generic "fetch a script from S3 into `/tmp` and run it
  as a subprocess" launcher. It copies every key of the incoming event into
  `os.environ` (this is how `RANK`/`WORLD_SIZE` reach `inference.py`'s
  `environ_or_required()` lookups), then downloads and runs `inference.py`
  per the `SCRIPT`/`S3_OBJECT_NAME`/`S3_OBJECT_TYPE` fields `initializer.py`
  puts in each rank's payload. This is why `inference.py` is a bare
  `__main__` script with no `lambda_handler` of its own. This image is
  heavy (Cylon/FMI/Boost/AWS-SDK-C++ built from source, plus
  torch/torchvision/timm) and is the one real prerequisite build —
  `executor_image_uri` has no default.

## Fixes made getting this to actually plan/apply

- **Pagination.** `initializer.py`, `initializer_FMI.py`, and
  `summarizer.py` each called `s3_client.list_objects_v2()` once with no
  continuation token — capped at 1000 keys. All three now paginate.
- **Dockerfile build context.** The source Dockerfile's `COPY
  lambda_entry3.py` / `COPY fmi.json` referenced files that did not exist in
  that directory — they live at `docker/aws/lambda/fmi/` on the remote
  `fmi-ds5110-support` branch of the `cylon` repo only. Both are now
  replicated alongside the Dockerfile at `docker/cosmic-ai-lambda/` in this
  repo, so it builds standalone without that other repo checked out.
- **ASL schema.** The click-ops state machine's Distributed Map used
  `"Iterator"` with a sibling `"ProcessorConfig"` — the current AWS API
  rejects this (`Field 'ProcessorConfig' is not supported`). The modern
  field is `ItemProcessor`, with `ProcessorConfig` nested inside it. Fixed
  in `../step_functions/workflow_cosmic_ai.asl.json`; confirmed by a real
  `terraform plan` against the destination account.

## Per-rank stage instrumentation (Track 1 design doc, section 4.1)

Added to `inference.py`, `lambda_entry3.py`, and `summarizer.py`:

- `init_s` — time from `lambda_entry3.handler`'s entry
  (`LAMBDA_ENTRY_TS`, an env var it sets) to `inference.py`'s own process
  starting. This is launcher overhead (S3 script fetch + subprocess spawn),
  **not** Lambda's own container cold-start init duration — that happens
  before `lambda_entry3.handler` is even called, and AWS reports it
  separately in the platform `REPORT` log line. `None` when `inference.py`
  runs outside that launcher (e.g. standalone).
- `payload_fetch_s` — fetching `payload.json` (batch size, data map, result
  path).
- `scatter_s` — the partition download in `load_data()`.
- `inference_s` — reuses the existing profiler-derived `total_time` (model
  execution only), not a second wall-clock measurement of the same thing.
- `barrier_s` — always `0.0`: Arm A has no cross-rank synchronization.
- `publish_s` / `total_s` — a record cannot report its own write duration
  on the write that produces it, so these two are filled by a second, cheap
  overwrite of the same per-rank result key rather than a separate sidecar
  file.
- Aggregator (`summarizer.py`): `aggregate_s`, `ranks_aggregated`,
  `bytes_aggregated`, written into the response body and into a sibling
  `aggregate_metrics.json` at the same prefix.

## Before `apply`

1. Build and push the executor image (from the `cylon-armada` repo root).
   Uses the same shared `cylon-armada` ECR repository name the rest of this
   project uses (per `target/aws/scripts/terraform/variables.tf`'s
   `ecr_repository_name` convention — one repo, one tag per image). It did
   not exist yet in the `cosmicai` account as of this module's last real
   `terraform plan` — `create-repository` is still in the commands below:
   ```bash
   docker build --platform=linux/amd64 \
     -t cosmic-ai-executor \
     -f docker/cosmic-ai-lambda/Dockerfile \
     .

   aws ecr get-login-password --profile cosmicai --region us-east-1 | \
     docker login --username AWS --password-stdin 881908115028.dkr.ecr.us-east-1.amazonaws.com
   aws ecr create-repository --profile cosmicai --region us-east-1 --repository-name cylon-armada
   docker tag cosmic-ai-executor:latest 881908115028.dkr.ecr.us-east-1.amazonaws.com/cylon-armada:cosmic-ai-executor
   docker push 881908115028.dkr.ecr.us-east-1.amazonaws.com/cylon-armada:cosmic-ai-executor
   ```
2. `executor_image_uri`'s default already matches the tag above, so no
   `terraform.tfvars` is required unless you tag differently.
3. `terraform init && terraform plan` — last verified clean (10 to add, 0
   errors) before the IAM `AccessDenied` on `apply` led to parking this
   module. The IAM roles still need resolving in that account before
   `apply` can succeed — see the "Parked" note above.

## Not yet done

- Resolving IAM permissions in the `cosmicai` account (881908115028) —
  `iam:CreateRole` is denied under `PowerUserAccess`, and no existing role
  there is trusted by `lambda.amazonaws.com`/`states.amazonaws.com`.
- Building/pushing the executor image itself (the Dockerfile build was not
  run against this account's ECR this session).
- `terraform apply` has not succeeded against this account.
- Phase 2 items 3 (uncompressed image size vs. Lambda's 10 GB ceiling, for
  `cylon-armada-cosmic-python`) and 4 (FMI/rendezvous smoke test) from
  `docs/superpowers/specs/2026-09-25-cosmicai-lambda-dataplane-design.md`
  are unrelated to this module and still open.