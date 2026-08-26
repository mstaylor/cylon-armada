# direct-redis Fargate Deployment

Steps to get the `direct-redis` FMI channel running on real AWS Fargate, after the
implementation and review work in `cylon` and `cylon-armada`. This is a deployment
runbook, not an experiment design doc — see `EXPERIMENT_PLAYBOOK.md` for the general
experiment pipeline this slots into.

**Context:** `direct-redis` replaces TCPunch NAT hole-punching with plain TCP
listen/connect between ranks, using Redis only to exchange `host:port` addresses.
It requires ranks that can bind and accept inbound connections — Fargate/ECS, not
Lambda (no inbound networking there). See `cylon`'s `RedisDirectPair.{hpp,cpp}` and
`cylon-armada`'s `target/shared/scripts/communicator/fmi_bridge.py` for the
implementation; the Rust port (`cylon`'s `rust/src/net/fmi/`) exists for parity but
Fargate experiments run through the Python/pycylon path.

---

## Prerequisites

- [ ] `cylon` repo: direct-redis work (C++/Cython + Rust) committed and pushed to
      `main` on `github.com/mstaylor/cylon` — the Docker builds clone this fresh,
      not your local checkout.
- [ ] `cylon-armada` repo: `fmi_bridge.py`, Terraform (`variables.tf`/`main.tf`), and
      the five `docker/Dockerfile.*` branch-pin fixes committed.

Nothing past this point works until both are true — the image build pulls `cylon`
from GitHub, and the port/security-group plumbing needs the Terraform changes applied.

---

## 1. Rebuild the image

The FMI-channel Lambda/ECS image is built from `Dockerfile.fmi.python`
(`python_image_tag` in Terraform defaults to `cylon-armada-fmi-python`, matching this
file):

```bash
docker build --platform=linux/amd64 \
    -t cylon-armada-fmi-python \
    -f docker/Dockerfile.fmi.python .
```

This clones `cylon`'s `main` branch fresh (per the Dockerfile fix) and builds pycylon
with `-DCYLON_USE_REDIS=1 -DCYLON_FMI=1` — both already set, no flag changes needed.

## 2. Push to ECR

```bash
aws ecr get-login-password --region <aws_region> \
    | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<aws_region>.amazonaws.com

docker tag cylon-armada-fmi-python:latest \
    <account_id>.dkr.ecr.<aws_region>.amazonaws.com/<ecr_repository_name>:cylon-armada-fmi-python

docker push \
    <account_id>.dkr.ecr.<aws_region>.amazonaws.com/<ecr_repository_name>:cylon-armada-fmi-python
```

Fill in `<account_id>`, `<aws_region>`, `<ecr_repository_name>` from your
`terraform.tfvars` (these have no defaults in `variables.tf` — they're required inputs).

## 3. Apply Terraform

```bash
cd target/aws/scripts/terraform
terraform init      # if providers aren't already cached locally
terraform plan       # confirm only the direct-redis additions show up as diffs:
                      #   + var.fmi_direct_redis_port
                      #   + aws_security_group_rule.fmi_direct_redis_ingress
                      #   ~ FMI_LISTEN_PORT added to the ECS task def env block
terraform apply
```

This is a real infrastructure change (creates a security group rule, updates the ECS
task definition). Run it yourself, or ask Claude to — either way, review the plan
output first per this project's standing rule on infrastructure changes.

## 4. Sync scripts to S3

The hot-reload path pulls `target/shared/scripts/` from S3 at worker startup — a
rebuilt image alone isn't enough if the deployed workers hot-reload past what's baked
into the container:

```bash
aws s3 sync target/shared/scripts/ s3://staylor.dev2/cylon-armada/scripts-v3/
```

Confirm this matches the `S3_SCRIPTS_PREFIX` the Lambda/ECS task is actually
configured with — prefix drift here has silently served stale code before (see
CLAUDE.md's "Debugging Distributed Runs" table).

## 5. Smoke test at world_size=2

Before any real sweep, confirm two Fargate tasks can actually find each other over
direct-redis — this is the first real exercise of the ECS metadata auto-discovery path
(`ECS_CONTAINER_METADATA_URI_V4`), which had zero live-Fargate coverage until now:

```bash
python target/aws/scripts/experiment/cloud_sweep.py \
    --arch ecs-fargate --scenario <domain> \
    --world-sizes 2 --runs 1 \
    --fmi-channel direct-redis \
    --dry-run   # then drop --dry-run once the plan looks right
```

Watch CloudWatch logs for both ranks. Look specifically for:
- `direct-redis: published rank N address ...` (confirms address publish succeeded)
- the non-`localhost` advertised address (confirms ECS metadata discovery worked,
  not a fallback to a wrong default)
- a successful connect between rank 0 and rank 1 (confirms the security group rule
  is actually allowing rank-to-rank traffic on `FMI_LISTEN_PORT`)

If this hangs or fails, check in this order: security group rule applied correctly,
`FMI_LISTEN_PORT` env var present in the running task definition, and whether
`ECS_CONTAINER_METADATA_URI_V4` is actually set in the container's environment
(Fargate sets this automatically — an EC2-backed ECS task might not).

## 6. Full experiment sweep

Once the smoke test round-trips cleanly, scale up per the world-size sweep convention
(always include 1 and 2 as baselines, per CLAUDE.md) and decide what direct-redis is
being compared against — likely the existing TCPunch (`direct`) and Redis-storage
(`redis`) channels, to show what plain-TCP-within-VPC buys over hole-punching or
storage-relay on Fargate specifically. Dry-run first per CLAUDE.md's AWS cost
discipline — sweeps run 4x per config across domains, and this is real spend.

---

## What changed to enable this

Quick pointer back to the actual work, for traceability:

| Change | Where |
|---|---|
| `direct-redis` channel (C++/Cython) | `cylon`: `RedisDirectPair.{hpp,cpp}`, `Direct.{hpp,cpp}`, `fmi_communicator.cpp` |
| `direct-redis` channel (Rust port) | `cylon`: `rust/src/net/fmi/redis_direct_pair.rs`, `direct.rs`, `cylon_communicator.rs` |
| Channel-type validation + `FMI_LISTEN_PORT` port resolution | `cylon-armada`: `target/shared/scripts/communicator/fmi_bridge.py` |
| `fmi_direct_redis_port` var, SG ingress rule, task-def env wiring | `cylon-armada`: `target/aws/scripts/terraform/{variables,main}.tf` |
| Docker builds pinned to `cylon`'s `main` branch | `cylon-armada`: `docker/Dockerfile.{python,fmi.python,uccucx.python,gpu,nodejs}` |

---

## Next: Experiment E

Once direct-redis is validated on Fargate, the next phase is Experiment E — the
headline end-to-end speedup measurement `S(N)` for the agentic framework, primary
baseline Ray/Matrix (HTTP/LangChain is a diagnostic, not the headline comparison) —
per `docs/EXPERIMENT_STATUS.md` and the proposal's Track A scope. This is a distinct,
larger design phase (agentic framework implementation, not just a channel deployment)
and gets its own scoping conversation rather than being folded into this runbook.