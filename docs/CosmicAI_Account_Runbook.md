# Cosmic AI in the cosmicai account: host runbook

Steps to deploy the Cosmic AI Lambda arms to the CloudBank cosmicai account (881908115028) and run
the first experiments, from the host OS. Prepared 6 October 2026.

Design: `docs/superpowers/specs/2026-10-06-cosmicai-account-portable-deployment-design.md`.
Experiments: `docs/superpowers/specs/2026-10-06-cosmicai-agentic-campaign-design.md`.

## Quick start: cylon account, zero config (current default)

Since 7 October the module, the campaign driver and the aggregator all default to the cylon account
(448324707516, AWS profile `cylon`, bucket `cosmicai-data-cylon`). The cosmicai account is an
override, for when AWS lifts its 3,008 MB Lambda memory cap (sections 1 to 7 below).

```bash
cd target/aws/scripts/terraform-cosmic-ai
terraform init
terraform plan        # executor 10240 MB, image 448324707516...cylon-armada@sha256:8b8961b0...
terraform apply
```

Pilot and summary, from the repository root:

```bash
export PYTHONPATH=$PWD/target/shared/scripts:$PYTHONPATH
python target/aws/scripts/experiment/cosmic_campaign_run.py --dry-run --series baseline --experiment pilot
python target/aws/scripts/experiment/cosmic_campaign_run.py --live --series baseline --experiment pilot --manifest pilot_cylon.jsonl
python -m results.cosmic_lambda_results --manifest pilot_cylon.jsonl --out pilot_cylon.csv
```

Before the first plan on a host that already deployed cosmicai in the default workspace, run
`terraform destroy` there with the previous (cosmicai) defaults, so the default workspace holds only
the cylon deployment.

## Status before you start

| Item | State |
|---|---|
| Admin access | `qad5gv` role has AdministratorAccess; its trust policy lets the IAM user `qad5gv` assume it |
| Terraform | `target/aws/scripts/terraform-cosmic-ai/`. Zero config: `terraform workspace select -or-create cosmicai`, then a plain `terraform plan` and `terraform apply`; the workspace chooses the account |
| Data | Bucket `cosmicai` has the 10, 25, 50, 75 and 100 MB partitions and the `Anomaly Detection/` folder |
| Executor image | **Not current** (checked 12 October). ECR `cosmic-ai-executor:latest` is the 6 October push (`sha256:51341c34...`), before the pairing, abort and warm-container fixes. The current image is the cylon account's `cylon-armada:cosmic-ai-executor`, `sha256:8b8961b0eb25b2644226517afb8bf50302dfaab9449baf60dcacf038acff2ba0`, pushed 11 October 19:00 with the TCPunch pairing fix |
| Lambda memory | Was capped at 3,008 MB per function; AWS reports the cap lifted (12 October). There is no API that reports the cap, so the apply is the check: the executor is created at 10,240 MB only if the cap is gone (section 5) |
| Lambda concurrency | **11,000** (checked 12 October); the campaign needs at most 517 |
| Lambda functions and state machines | None deployed (checked 12 October) |
| Rendezvous server | Arm B uses the public server `cylon-rendezvous.aws-cylondata.com:10000`, the same one as the cylon account; nothing to deploy in this account |

## 1. Set up the admin profile on the host

The host needs the `cosmicai` profile (the IAM user's access keys) and a profile that assumes the
admin role. Add to the host's `~/.aws/config`:

```ini
[profile cosmicai-admin]
role_arn = arn:aws:iam::881908115028:role/qad5gv
source_profile = cosmicai
region = us-east-1
```

Check it:

```bash
aws sts get-caller-identity --profile cosmicai-admin --query Arn --output text
```

Expected: `arn:aws:sts::881908115028:assumed-role/qad5gv/...`

## 2. Push the current executor image

Use the image the cylon account runs now, so both accounts execute the same code. Pull it by digest
from the cylon account's ECR:

```bash
aws ecr get-login-password --region us-east-1 --profile cylon | \
  docker login --username AWS --password-stdin 448324707516.dkr.ecr.us-east-1.amazonaws.com
docker pull --platform=linux/amd64 \
  448324707516.dkr.ecr.us-east-1.amazonaws.com/cylon-armada@sha256:8b8961b0eb25b2644226517afb8bf50302dfaab9449baf60dcacf038acff2ba0
docker tag \
  448324707516.dkr.ecr.us-east-1.amazonaws.com/cylon-armada@sha256:8b8961b0eb25b2644226517afb8bf50302dfaab9449baf60dcacf038acff2ba0 \
  cosmic-ai-executor:latest
```

Push to the cosmicai account:

```bash
aws ecr get-login-password --region us-east-1 --profile cosmicai-admin | \
  docker login --username AWS --password-stdin 881908115028.dkr.ecr.us-east-1.amazonaws.com
docker tag cosmic-ai-executor:latest 881908115028.dkr.ecr.us-east-1.amazonaws.com/cosmic-ai-executor:latest
docker push 881908115028.dkr.ecr.us-east-1.amazonaws.com/cosmic-ai-executor:latest
```

Verify the digest:

```bash
aws ecr describe-images --repository-name cosmic-ai-executor --image-ids imageTag=latest \
  --profile cosmicai-admin --region us-east-1 --query 'imageDetails[0].imageDigest' --output text
```

Expected: `sha256:8b8961b0eb25b2644226517afb8bf50302dfaab9449baf60dcacf038acff2ba0`, the same digest
as the cylon account. If it is still `sha256:51341c34...`, the push did not land. If it is a third
digest, the image was rebuilt rather than copied; pull by digest again.

## 3. Optional: keep a copy of the files the apply replaces

The apply overwrites two objects in the bucket: `Anomaly Detection/Inference/inference.py` (an old
version) and `Anomaly Detection/Inference/inference_FMI.py` (the original 2024 version, also in
AI-for-Astronomy git history at `bd0c147`).

```bash
aws s3 cp "s3://cosmicai/Anomaly Detection/Inference/inference_FMI.py" ./inference_FMI.2024.py --profile cosmicai-admin
aws s3 cp "s3://cosmicai/Anomaly Detection/Inference/inference.py" ./inference.2024.py --profile cosmicai-admin
```

## 4. Deploy

From the cylon-armada repository root on the host, with the current working tree:

```bash
cd target/aws/scripts/terraform-cosmic-ai
terraform init
terraform workspace select -or-create cosmicai
terraform plan
```

Check the plan before applying:

- only additions, nothing to change or destroy (the account has no Cosmic AI resources yet)
- the executor `image_uri` ends in `@sha256:8b8961b0...`
- the FMI initializer's environment has `FMI_HOLEPUNCH_CONNECT_TIMEOUT_MS = 10000`,
  `FMI_ESTABLISH_PARALLELISM = 64` and `RENDEZVOUS_HOST = cylon-rendezvous.aws-cylondata.com`
- no error from `terraform_data.account_guard` (it fails the plan when the credentials belong to
  another account)

Then:

```bash
terraform apply
```

No variables or var-file: the selected workspace chooses the account. The `accounts` variable in
`variables.tf` holds each account's profile, account id, bucket and executor image, keyed by workspace
(`cylon`, `cosmicai`; `default` means cylon). The workspaces also keep the two accounts' state apart.

## 5. Check the deployment

```bash
aws stepfunctions list-state-machines --profile cosmicai-admin --region us-east-1 \
  --query 'stateMachines[?contains(name,`cosmic`)].name' --output text
aws lambda get-function --function-name cylon-armada-cosmic-ai-executor --profile cosmicai-admin \
  --region us-east-1 --query '[Configuration.MemorySize,Code.ResolvedImageUri]' --output text
```

Expected: `cylon-armada-cosmic-ai-workflow` and `cylon-armada-cosmic-ai-fmi-workflow`; executor at
10240 MB on `@sha256:8b8961b0...`. If the apply fails creating the executor with a memory error, the
3,008 MB cap is still in place.

## 6. First runs: 1 and 2 workers

The campaign driver and aggregator run with the cylon-armada Python code. From the repository root:

```bash
export PYTHONPATH=$PWD/target/shared/scripts:$PYTHONPATH
cd target/aws/scripts/experiment
```

Dry run first; it launches nothing:

```bash
python cosmic_campaign_run.py --dry-run --series baseline --experiment pilot \
  --profile cosmicai-admin --bucket cosmicai --data-bucket cosmicai
```

### Arm A, 1 and 2 workers

```bash
python cosmic_campaign_run.py --live --series baseline --experiment pilot \
  --profile cosmicai-admin --bucket cosmicai --data-bucket cosmicai --manifest pilot_cosmicai.jsonl
```

That is 8 executions (a cold-start run and 3 measured runs at 1 and 2 workers each).

### Arm B, 1 and 2 workers

Arm B uses the same public rendezvous server as the cylon account. Before the 2-worker run, check
it answers:

```bash
getent hosts cylon-rendezvous.aws-cylondata.com
timeout 5 bash -c "</dev/tcp/cylon-rendezvous.aws-cylondata.com/10000" && echo open
```

Then:

```bash
python cosmic_campaign_run.py --live --arm B --series baseline --experiment pilot \
  --profile cosmicai-admin --bucket cosmicai --data-bucket cosmicai --manifest pilot_cosmicai.jsonl
```

### Summarize

```bash
cd ../../../..
python -m results.cosmic_lambda_results --manifest target/aws/scripts/experiment/pilot_cosmicai.jsonl \
  --profile cosmicai-admin --bucket cosmicai --out pilot_cosmicai.csv
```

Expected: every row `complete`, with Arm A and Arm B rows separate and a `cold_start` row for each
configuration.

## 7. Ramp, then the full campaign

The concurrency limit (11,000) covers the largest configuration (517 workers). After the pilot rows
are complete, ramp to 41 workers, then run the full campaign. Dry run before each to see the
execution count and cost estimate:

```bash
python cosmic_campaign_run.py --dry-run --series scaling --max-workers 41 --profile cosmicai-admin --bucket cosmicai --data-bucket cosmicai
python cosmic_campaign_run.py --live --series scaling --max-workers 41 --experiment pilot \
  --profile cosmicai-admin --bucket cosmicai --data-bucket cosmicai --manifest pilot_cosmicai.jsonl
python cosmic_campaign_run.py --dry-run --profile cosmicai-admin --bucket cosmicai --data-bucket cosmicai
python cosmic_campaign_run.py --live --experiment exp1 --profile cosmicai-admin --bucket cosmicai --data-bucket cosmicai \
  --manifest exp1_cosmicai.jsonl
```

Check the limit at any time:

```bash
aws lambda get-account-settings --profile cosmicai-admin --region us-east-1 \
  --query AccountLimit.ConcurrentExecutions
```

## Reminders

- One campaign driver at a time per account: Arm A executions share `payload.json` at the bucket root.
- CloudBank: do not grant access to anyone outside the ACCESS allocation process.
- Nothing in this runbook stages or commits code.
