# Track 1 (Cosmic AI Lambda data plane) — handoff, 2026-09-30

Status of the work from this session, with exact next commands. See
`docs/superpowers/specs/2026-09-25-cosmicai-lambda-dataplane-design.md` for
the full design this implements, and
`target/aws/scripts/terraform-cosmic-ai/README.md` for the Terraform
module's own architecture notes.

## Done

- **Phase 1, gate validation** — satisfied by this session's earlier
  photometry_classification reuse-gate work
  (`docs/Reuse_Gate_Validity_Findings_2026-09-28.md`, Parts 10-13).
- **Phase 2 item 1, pagination** — `initializer.py`, `initializer_FMI.py`,
  `summarizer.py` in `/home/parallels/AI-for-Astronomy/aws/lambda/` all
  paginate `list_objects_v2` now.
- **Phase 2 item 2, stage instrumentation** — `init_s`, `payload_fetch_s`,
  `scatter_s`, `inference_s`, `barrier_s`, `publish_s`, `total_s` per rank
  (`inference.py`, `lambda_entry3.py`), `aggregate_s`/`ranks_aggregated`/
  `bytes_aggregated` at the aggregator (`summarizer.py`).
- **Terraform migration module** —
  `target/aws/scripts/terraform-cosmic-ai/` in cylon-armada, validated with
  a real `terraform plan`: clean, 10 to add, 0 errors.
- **Account switched to 448324707516** (`default` profile), not the
  `cosmicai` profile/881908115028 originally planned. That account's IAM
  user only has `PowerUserAccess`, which deliberately excludes
  `iam:CreateRole` — confirmed by an actual `AccessDenied` on `terraform
  apply`, and by checking every existing role in that account for one
  already trusted by `lambda.amazonaws.com`/`states.amazonaws.com` (none
  exists). 448324707516 is where the `cylon-armada` ECR repo and the
  `cosmicai-data-cylon` S3 bucket (the one `inference.py`/`summarizer.py`
  already hardcode) already live, and where you have full IAM access.
- **Module made fully self-contained** — `initializer.py`/`summarizer.py`
  are now also copied into `target/aws/scripts/terraform-cosmic-ai/
  lambda_src/` (byte-identical to the `AI-for-Astronomy` originals), and
  the `archive_file` data sources zip from there instead of reaching into
  a sibling `AI-for-Astronomy` checkout. This was a real bug: the module's
  previous default assumed `/home/parallels/AI-for-Astronomy` existed on
  whatever machine runs `terraform apply`, which failed on a different
  machine with "could not archive missing file."
- `executor_image_uri` now has a default (matching the tag below), so
  `terraform plan`/`apply` no longer prompts for it interactively.
- **Two real bugs found and fixed along the way**:
  - The executor Dockerfile's `COPY lambda_entry3.py` / `COPY fmi.json`
    referenced files that did not exist in its original directory
    (`cylon` repo, `docker/aws/lambda/cosmi-ai/`) — they only existed on
    the remote `fmi-ds5110-support` branch. Both are now in place.
  - The click-ops state machine's Distributed Map used a schema
    (`"Iterator"` + sibling `"ProcessorConfig"`) the current AWS API
    rejects. Fixed to the modern `ItemProcessor` nesting in
    `target/aws/scripts/step_functions/workflow_cosmic_ai.asl.json`.
- **Design correction**: `init`/`summarize` Lambdas do not need a container
  image at all (plain boto3, nothing beyond what the Lambda Python runtime
  bundles) — switched to zip packaging. Only the `executor` Lambda
  (`cosmic-executor`) genuinely needs a container image.
- **Dockerfile replicated into this repo**, at `docker/cosmic-ai-lambda/`
  (Dockerfile + `lambda_entry3.py` + `fmi.json`, byte-identical to the
  `cylon` repo source) — the build no longer needs that other repo checked
  out alongside this one.

## Next: build and deploy the executor image

```bash
docker build --platform=linux/amd64 \
  -t cosmic-ai-executor \
  -f docker/cosmic-ai-lambda/Dockerfile \
  docker/cosmic-ai-lambda
```

This is the slow step — it compiles Boost, AWS SDK C++, and FMI from
source, plus installs torch/torchvision/timm. Not run this session.

Same shared `cylon-armada` ECR repository the rest of this project uses
(per `target/aws/scripts/terraform/variables.tf`'s `ecr_repository_name`
convention — one repo, one tag per image). It already exists in
448324707516 (confirmed via `aws ecr describe-repositories
--repository-names cylon-armada`), so no `create-repository` step is
needed.

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 448324707516.dkr.ecr.us-east-1.amazonaws.com

docker tag cosmic-ai-executor:latest 448324707516.dkr.ecr.us-east-1.amazonaws.com/cylon-armada:cosmic-ai-executor
docker push 448324707516.dkr.ecr.us-east-1.amazonaws.com/cylon-armada:cosmic-ai-executor
```

**Update, 2026-10-01: the live deployment target changed.** `terraform-cosmic-ai/`
is parked (targets the `cosmicai` account, 881908115028, which cannot
`iam:CreateRole`). The Cosmic AI Arms A/B resources — the three Lambdas
(`cosmic_ai_init`/`cosmic_ai_executor`/`cosmic_ai_summarize`), the
`cosmic_ai_workflow` state machine, and the `cosmic_ai_inference` S3 upload
(the fixed `inference.py`, fetched fresh from S3 by `lambda_entry3.handler`
on every invocation) — are now resources inside the main module instead,
reusing its existing shared `aws_iam_role.lambda_execution`/
`aws_iam_role.step_functions_execution` roles. Apply from there:

```bash
cd target/aws/scripts/terraform
terraform init
terraform plan    # last verified clean: 54 to add, 0 errors, against 448324707516
terraform apply
```

## Still open after that

- **Phase 2 item 3** — uncompressed image size of `cylon-armada-cosmic-python`
  against Lambda's 10 GB ceiling:
  ```bash
  docker build --platform=linux/amd64 -t cylon-armada-fmi-python -f docker/Dockerfile.fmi.python .
  docker build --platform=linux/amd64 -t cylon-armada-cosmic-python -f docker/Dockerfile.cosmic.python .
  docker image inspect --format='{{.Size}}' cylon-armada-cosmic-python
  ```
  (from the `cylon-armada` repo root). Compare the result against
  10737418240 bytes (10 GB).

- **Phase 2 item 4** — FMI/rendezvous smoke test at a small rank count on
  Lambda, reusing the existing `rendezvous_test.py` (already deployed,
  already has a documented two-invocation N=2 command):
  ```bash
  aws lambda invoke --function-name cylon-armada-rendezvous-test \
    --payload '{"rank":0,"world_size":2,"comm_name":"test_abc"}' /dev/stdout
  aws lambda invoke --function-name cylon-armada-rendezvous-test \
    --payload '{"rank":1,"world_size":2,"comm_name":"test_abc"}' /dev/stdout
  ```
  This validates the rendezvous infrastructure itself, not the Cosmic-AI
  FMI path specifically (`initializer_FMI.py`/`inference_FMI.py` are
  unfinished — Phase 3 scope, not Phase 2).

- **Phase 3** (Arms A and B on Lambda) and **Phase 4** (Arm C) are separate,
  larger efforts not started.