# Cosmic AI in the cosmicai account: host runbook

Steps to deploy the Cosmic AI Lambda arms to the CloudBank cosmicai account (881908115028) and run
the first experiments, from the host OS. Prepared 6 October 2026.

Design: `docs/superpowers/specs/2026-10-06-cosmicai-account-portable-deployment-design.md`.
Experiments: `docs/superpowers/specs/2026-10-06-cosmicai-agentic-campaign-design.md`.

## Status before you start

| Item | State |
|---|---|
| Admin access | `qad5gv` role has AdministratorAccess; its trust policy lets the IAM user `qad5gv` assume it |
| Terraform | `target/aws/scripts/terraform-cosmic-ai/`, workspace `cosmicai`, `cosmicai.tfvars`. Last plan: 18 to add, 0 to change, 0 to destroy |
| Data | Bucket `cosmicai` has the 10, 25, 50, 75 and 100 MB partitions and the `Anomaly Detection/` folder |
| Executor image | **Not current.** ECR `cosmic-ai-executor:latest` is the 1 October image (`sha256:5d477d77...`). The current image is `sha256:51341c34...` |
| Lambda concurrency | **10.** Increase to 11,000 requested 6 October (request id `8b00390eb57d42b3838f184a01897de9E8DIHMhH`), pending. Until approved, run nothing above 10 workers |

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

The image was built on the host on 4 October as `cosmic-ai-executor:latest`. Check it is the right
one first:

```bash
docker image inspect cosmic-ai-executor:latest --format '{{.Id}}'
```

If the host no longer has it, pull it from the cylon account's ECR:

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 448324707516.dkr.ecr.us-east-1.amazonaws.com
docker pull --platform=linux/amd64 448324707516.dkr.ecr.us-east-1.amazonaws.com/cylon-armada:cosmic-ai-executor
docker tag 448324707516.dkr.ecr.us-east-1.amazonaws.com/cylon-armada:cosmic-ai-executor cosmic-ai-executor:latest
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

Expected: `sha256:51341c349abfae4f5dd5f3110165d82714331dfe54067171fb24ec3fc08d49e0`. If it is still
`sha256:5d477d77...`, the push did not land.

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
terraform workspace select cosmicai || terraform workspace new cosmicai
terraform plan -var-file=cosmicai.tfvars
```

Check the plan before applying:

- `Plan: 18 to add, 0 to change, 0 to destroy`
- the executor `image_uri` ends in `@sha256:51341c34...`
- no error from `terraform_data.account_guard` (it fails the plan when the credentials belong to
  another account)

Then:

```bash
terraform apply -var-file=cosmicai.tfvars
```

Do not use `cylon.tfvars` yet: in the cylon account, the main `terraform/` module still defines
Cosmic AI resources with the same names.

## 5. Check the deployment

```bash
aws stepfunctions list-state-machines --profile cosmicai-admin --region us-east-1 \
  --query 'stateMachines[?contains(name,`cosmic`)].name' --output text
aws lambda get-function --function-name cylon-armada-cosmic-ai-executor --profile cosmicai-admin \
  --region us-east-1 --query '[Configuration.MemorySize,Code.ResolvedImageUri]' --output text
```

Expected: `cylon-armada-cosmic-ai-workflow` and `cylon-armada-cosmic-ai-fmi-workflow`; executor at
10240 MB on `@sha256:51341c34...`.

## 6. First runs within the 10-worker limit

The campaign driver and aggregator run with the cylon-armada Python code. From the repository root:

```bash
export AWS_PROFILE=cosmicai-admin
export PYTHONPATH=$PWD/target/shared/scripts:$PYTHONPATH
cd target/aws/scripts/experiment
```

Dry run first; it launches nothing:

```bash
python cosmic_campaign_run.py --dry-run --series baseline --experiment pilot \
  --bucket cosmicai --data-bucket cosmicai
```

### Arm A, 1 and 2 workers

```bash
python cosmic_campaign_run.py --live --series baseline --experiment pilot \
  --bucket cosmicai --data-bucket cosmicai --manifest pilot_cosmicai.jsonl
```

That is 8 executions (a cold-start run and 3 measured runs at 1 and 2 workers each).

### Arm B, 1 and 2 workers

Before the 2-worker run, start the rendezvous server with your script and check its DNS name points
at the running task and the port answers:

```bash
getent hosts cylon-rendezvous.aws-cylondata.com
timeout 5 bash -c "</dev/tcp/cylon-rendezvous.aws-cylondata.com/10000" && echo open
```

Then:

```bash
python cosmic_campaign_run.py --live --arm B --series baseline --experiment pilot \
  --bucket cosmicai --data-bucket cosmicai --manifest pilot_cosmicai.jsonl
```

### Summarize

```bash
cd ../../../..
python -m results.cosmic_lambda_results --manifest target/aws/scripts/experiment/pilot_cosmicai.jsonl \
  --bucket cosmicai --measured-runs 3 --batch-measured-runs 4 --cold-init-threshold-s 10 \
  --out pilot_cosmicai.csv
```

Expected: every row `complete`, with Arm A and Arm B rows separate and a `cold_start` row for each
configuration.

## 7. Check the concurrency increase

```bash
aws service-quotas list-requested-service-quota-change-history-by-quota --service-code lambda \
  --quota-code L-B99A9384 --profile cosmicai-admin --region us-east-1 \
  --query 'RequestedQuotas[].[DesiredValue,Status]' --output text
aws lambda get-account-settings --profile cosmicai-admin --region us-east-1 \
  --query AccountLimit.ConcurrentExecutions
```

When it shows `APPROVED` and a limit of 600 or more, the next steps are the ramp
(`--series scaling --max-workers 41`) and then the full campaign. Run a dry run before each to see
the execution count and cost estimate.

## Reminders

- One campaign driver at a time per account: Arm A executions share `payload.json` at the bucket root.
- CloudBank: do not grant access to anyone outside the ACCESS allocation process.
- Nothing in this runbook stages or commits code.