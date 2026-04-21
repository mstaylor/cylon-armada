terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = "gpu-experiments"
    ManagedBy   = "terraform"
  }

  # Static env vars baked into the GPU task definition.
  # Dynamic per-run config (WORKFLOW_ID, TASKS_JSON, SCALING, etc.) is
  # injected at execution time via Step Functions ContainerOverrides.
  ecs_gpu_env = [
    { name = "BEDROCK_LLM_MODEL_ID",         value = var.bedrock_llm_model_id },
    { name = "BEDROCK_EMBEDDING_MODEL_ID",    value = var.bedrock_embedding_model_id },
    { name = "BEDROCK_EMBEDDING_DIMENSIONS",  value = tostring(var.bedrock_embedding_dimensions) },
    { name = "SIMILARITY_THRESHOLD",          value = tostring(var.similarity_threshold) },
    { name = "CONTEXT_BACKEND",               value = var.context_backend },
    { name = "REDIS_HOST",                    value = var.redis_host },
    { name = "REDIS_PORT",                    value = tostring(var.redis_port) },
    { name = "DYNAMO_TABLE_NAME",             value = var.dynamo_table_name },
    { name = "RESULTS_BUCKET",                value = var.results_bucket_name },
    { name = "AWS_DEFAULT_REGION",            value = var.aws_region },
    { name = "SIMD_BACKEND",                  value = "pycylon" },
    { name = "COMPUTE_PLATFORM",              value = "ecs-ec2-gpu" },
  ]

  # Template variables for the GPU Step Functions ASL
  gpu_asl_vars = {
    ECS_EC2_CLUSTER    = data.aws_ecs_cluster.ec2.arn
    GPU_TASK_DEF       = aws_ecs_task_definition.gpu_armada.arn
    CONTAINER_NAME     = var.ecs_container_name
    SUBNET_IDS         = jsonencode(var.ecs_task_subnet_ids)
    SECURITY_GROUP_IDS = jsonencode(var.ecs_security_group_ids)
  }
}

# ---------------------------------------------------------------------------
# ECR — reference existing repository
# ---------------------------------------------------------------------------

data "aws_ecr_repository" "main" {
  name = var.ecr_repository_name
}

# ---------------------------------------------------------------------------
# ECS — reference existing EC2 cluster with GPU instances
# ---------------------------------------------------------------------------

data "aws_ecs_cluster" "ec2" {
  cluster_name = var.ecs_ec2_cluster_name
}

# ---------------------------------------------------------------------------
# S3 — reference existing buckets
# ---------------------------------------------------------------------------

data "aws_s3_bucket" "results" {
  bucket = var.results_bucket_name
}

data "aws_s3_bucket" "scripts" {
  bucket = var.scripts_bucket_name
}

# ---------------------------------------------------------------------------
# CloudWatch Log Group — GPU ECS tasks
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ecs_gpu" {
  name              = "/ecs/${var.project_name}-gpu"
  retention_in_days = var.ecs_log_retention_days
  tags              = local.common_tags
}

# ---------------------------------------------------------------------------
# IAM Role — ECS task execution (ECR pull + CloudWatch logs)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-gpu-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---------------------------------------------------------------------------
# IAM Role — ECS task role (what the container can do at runtime)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-gpu-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${var.project_name}-gpu-ecs-task-policy"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/*",
          "arn:aws:bedrock:*:${var.account_id}:inference-profile/*"
        ]
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          data.aws_s3_bucket.results.arn,
          "${data.aws_s3_bucket.results.arn}/*",
          data.aws_s3_bucket.scripts.arn,
          "${data.aws_s3_bucket.scripts.arn}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem",
          "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan",
        ]
        Resource = [
          "arn:aws:dynamodb:${var.aws_region}:${var.account_id}:table/${var.dynamo_table_name}",
          "arn:aws:dynamodb:${var.aws_region}:${var.account_id}:table/${var.dynamo_table_name}/index/*",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# IAM Role — Step Functions execution
# ---------------------------------------------------------------------------

resource "aws_iam_role" "step_functions_execution" {
  name = "${var.project_name}-gpu-sfn-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "states.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "step_functions_policy" {
  name = "${var.project_name}-gpu-sfn-policy"
  role = aws_iam_role.step_functions_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["ecs:RunTask", "ecs:StopTask", "ecs:DescribeTasks"]
        Resource = ["*"]
      },
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [
          aws_iam_role.ecs_task.arn,
          aws_iam_role.ecs_execution.arn,
        ]
      },
      # EventBridge callback for ecs:runTask.sync
      {
        Effect = "Allow"
        Action = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
        Resource = ["arn:aws:events:${var.aws_region}:${var.account_id}:rule/StepFunctionsGetEventsForECSTaskRule"]
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery",
                    "logs:DeleteLogDelivery", "logs:ListLogDeliveries", "logs:PutResourcePolicy",
                    "logs:DescribeResourcePolicies", "logs:DescribeLogGroups"]
        Resource = ["*"]
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# ECS Task Definition — GPU armada runner
#
# EC2 launch type only (GPU placement). GPU allocation via resourceRequirements.
# network_mode = "awsvpc" is required for EC2 + awsvpc.
# The container runs armada_ecs_runner.py with SIMD_BACKEND=pycylon (gcylon).
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "gpu_armada" {
  family                   = "${var.project_name}-gpu"
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.gpu_cpu)
  memory                   = tostring(var.gpu_memory_mb)
  task_role_arn            = aws_iam_role.ecs_task.arn
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name      = var.ecs_container_name
    image     = "${data.aws_ecr_repository.main.repository_url}:${var.gpu_image_tag}"
    essential = true

    entryPoint = ["/cylon/target/aws/scripts/lambda/runCyloninLambda.sh"]
    command     = ["python", "armada_ecs_runner.py"]

    environment = local.ecs_gpu_env

    # Request GPU(s) from the EC2 instance via ECS resource requirements
    resourceRequirements = [
      {
        type  = "GPU"
        value = tostring(var.gpu_count)
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_gpu.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs-gpu"
      }
    }
  }])

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Step Functions State Machine — ECS EC2 GPU workflow
# ---------------------------------------------------------------------------

resource "aws_sfn_state_machine" "ecs_ec2_gpu_workflow" {
  name     = "${var.project_name}-ecs-ec2-gpu-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "STANDARD"

  definition = templatefile(
    "${path.module}/../step_functions/workflow_ecs_ec2_gpu.asl.json",
    local.gpu_asl_vars
  )

  tags = local.common_tags
}