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
  a real `terraform plan` against the destination account (`cosmicai`
  profile, account 881908115028): clean, 10 to add, 0 errors.
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

```bash
aws ecr get-login-password --profile cosmicai --region us-east-1 | \
  docker login --username AWS --password-stdin 881908115028.dkr.ecr.us-east-1.amazonaws.com

aws ecr create-repository --profile cosmicai --region us-east-1 --repository-name cosmic-ai-executor

docker tag cosmic-ai-executor:latest 881908115028.dkr.ecr.us-east-1.amazonaws.com/cosmic-ai-executor:latest
docker push 881908115028.dkr.ecr.us-east-1.amazonaws.com/cosmic-ai-executor:latest
```

Then, from `target/aws/scripts/terraform-cosmic-ai/`:

```bash
cp terraform.tfvars.example terraform.tfvars
# terraform.tfvars already has the right image URI if you used the tag above —
# edit it if you tagged/pushed differently.

terraform init
terraform plan    # re-verify before apply; last verified plan was clean
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