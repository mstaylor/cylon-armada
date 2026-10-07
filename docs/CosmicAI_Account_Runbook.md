# Cosmic AI in the cosmicai account: host runbook

Steps to deploy the Cosmic AI Lambda arms to the CloudBank cosmicai account (881908115028) and run
the first experiments, from the host OS. Prepared 6 October 2026.

Design: `docs/superpowers/specs/2026-10-06-cosmicai-account-portable-deployment-design.md`.
Experiments: `docs/superpowers/specs/2026-10-06-cosmicai-agentic-campaign-design.md`.

## Status before you start

| Item | State |
|---|---|
| Admin access | `qad5gv` role has AdministratorAccess; its trust policy lets the IAM user `qad5gv` assume it |
| Terraform | `target/aws/scripts/terraform-cosmic-ai/`. Zero config: the cosmicai account is the default, so a plain `terraform plan` needs no inputs. Last plan: 18 to add, 0 to change, 0 to destroy |
| Data | Bucket `cosmicai` has the 10, 25, 50, 75 and 100 MB partitions and the `Anomaly Detection/` folder |
| Executor image | **Not current.** ECR `cosmic-ai-executor:latest` is the 1 October image (`sha256:5d477d77...`). The current image is `sha256:51341c34...` |
| Lambda memory | **Capped at 3,008 MB** per function (new-account limit). The executor needs 10,240 MB, the original campaign's setting (it uses up to 7 GB on 100 MB partitions). Lifting it needs an AWS Support case, possibly through CloudBank. Until then the apply creates everything except the executor |
| Lambda concurrency | **1,000** (raised 7 October; the campaign needs at most 517). A separate request for 11,000 is still open as a support case and is not needed |

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
terraform plan
```

Check the plan before applying:

- `Plan: 18 to add, 0 to change, 0 to destroy`
- the executor `image_uri` ends in `@sha256:51341c34...`
- no error from `terraform_data.account_guard` (it fails the plan when the credentials belong to
  another account)

Then:

```bash
terraform apply
```

The cylon account is an override (`terraform workspace new cylon`, then `-var-file=cylon.tfvars`),
kept in its own workspace so its state never mixes with cosmicai's. Do not use it yet: in the cylon
account, the main `terraform/` module still defines Cosmic AI resources with the same names.

## 5. Check the deployment

```bash
aws stepfunctions list-state-machines --profile cosmicai-admin --region us-east-1 \
  --query 'stateMachines[?contains(name,`cosmic`)].name' --output text
aws lambda get-function --function-name cylon-armada-cosmic-ai-executor --profile cosmicai-admin \
  --region us-east-1 --query '[Configuration.MemorySize,Code.ResolvedImageUri]' --output text
```

Expected: `cylon-armada-cosmic-ai-workflow` and `cylon-armada-cosmic-ai-fmi-workflow`; executor at
10240 MB on `@sha256:51341c34...`.

## 6. First runs: 1 and 2 workers

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

## 7. Ramp, then the full campaign

The concurrency limit (1,000) covers the largest configuration (517 workers). After the pilot rows
are complete, ramp to 41 workers, then run the full campaign. Dry run before each to see the
execution count and cost estimate:

```bash
python cosmic_campaign_run.py --dry-run --series scaling --max-workers 41 --bucket cosmicai --data-bucket cosmicai
python cosmic_campaign_run.py --live --series scaling --max-workers 41 --experiment pilot \
  --bucket cosmicai --data-bucket cosmicai --manifest pilot_cosmicai.jsonl
python cosmic_campaign_run.py --dry-run --bucket cosmicai --data-bucket cosmicai
python cosmic_campaign_run.py --live --experiment exp1 --bucket cosmicai --data-bucket cosmicai \
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
## Alternative: run in the cylon account while the memory cap stands

The cosmicai account rejects Lambda functions above 3,008 MB until AWS lifts the new-account limit.
The cylon account (448324707516) has no such cap, a concurrency limit of 11,000, the current image
(`cylon-armada:cosmic-ai-executor`, `sha256:51341c34...`) and the data in `cosmicai-data-cylon`.
A read-only plan there on 7 October showed 18 to add, 0 to change, 0 to destroy.

Deploy the same module with the cylon override, in its own workspace (the default workspace holds
the cosmicai deployment):

```bash
cd target/aws/scripts/terraform-cosmic-ai
terraform workspace new cylon        # later: terraform workspace select cylon
terraform plan  -var-file=cylon.tfvars
terraform apply -var-file=cylon.tfvars
```

Do not apply the main `terraform/` module at the same time: it defines Cosmic AI resources with the
same names.

The campaign driver and aggregator defaults are the cylon account, so no bucket flags or
`AWS_PROFILE` are needed:

```bash
export PYTHONPATH=$PWD/target/shared/scripts:$PYTHONPATH
cd target/aws/scripts/experiment
python cosmic_campaign_run.py --dry-run --series baseline --experiment pilot
python cosmic_campaign_run.py --live --series baseline --experiment pilot --manifest pilot_cylon.jsonl
```

Summarize with `--bucket cosmicai-data-cylon` in the step 6 aggregator command.
