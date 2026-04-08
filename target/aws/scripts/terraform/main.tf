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
    Environment = "phase1"
    ManagedBy   = "terraform"
  }

  redis_endpoint = (
    var.create_ecs_redis    ? var.redis_hostname :
    var.create_elasticache  ? aws_elasticache_cluster.redis[0].cache_nodes[0].address :
    var.redis_host
  )

  # Env vars shared by all Lambda functions
  # Note: AWS_DEFAULT_REGION is a reserved Lambda key and cannot be set here;
  # Lambda functions read the region via the execution environment automatically.
  lambda_env = {
    BEDROCK_LLM_MODEL_ID         = var.bedrock_llm_model_id
    BEDROCK_EMBEDDING_MODEL_ID   = var.bedrock_embedding_model_id
    BEDROCK_EMBEDDING_DIMENSIONS = tostring(var.bedrock_embedding_dimensions)
    SIMILARITY_THRESHOLD         = tostring(var.similarity_threshold)
    CONTEXT_BACKEND              = var.context_backend
    REDIS_HOST                   = local.redis_endpoint
    REDIS_PORT                   = tostring(var.redis_port)
    DYNAMO_TABLE_NAME            = aws_dynamodb_table.context_store.name
  }

  # Env vars shared by all ECS tasks (static; dynamic fields injected per-run
  # via Step Functions ContainerOverrides)
  ecs_env = [
    { name = "BEDROCK_LLM_MODEL_ID",         value = var.bedrock_llm_model_id },
    { name = "BEDROCK_EMBEDDING_MODEL_ID",    value = var.bedrock_embedding_model_id },
    { name = "BEDROCK_EMBEDDING_DIMENSIONS",  value = tostring(var.bedrock_embedding_dimensions) },
    { name = "SIMILARITY_THRESHOLD",          value = tostring(var.similarity_threshold) },
    { name = "CONTEXT_BACKEND",               value = var.context_backend },
    { name = "REDIS_HOST",                    value = local.redis_endpoint },
    { name = "REDIS_PORT",                    value = tostring(var.redis_port) },
    { name = "DYNAMO_TABLE_NAME",             value = aws_dynamodb_table.context_store.name },
    { name = "RESULTS_BUCKET",                value = var.results_bucket_name },
    { name = "AWS_DEFAULT_REGION",            value = var.aws_region },
  ]

  # Template variables for Lambda Step Functions ASL files
  asl_vars = {
    AWS_REGION = var.aws_region
    ACCOUNT_ID = var.account_id
  }

  # Template variables for ECS Step Functions ASL files
  ecs_asl_vars = {
    AWS_REGION            = var.aws_region
    ACCOUNT_ID            = var.account_id
    ECS_FARGATE_CLUSTER   = data.aws_ecs_cluster.fargate.arn
    ECS_EC2_CLUSTER       = data.aws_ecs_cluster.ec2.arn
    PYTHON_TASK_DEF       = aws_ecs_task_definition.python_armada.arn
    CONTAINER_NAME        = var.ecs_container_name
    SUBNET_IDS            = jsonencode(var.ecs_task_subnet_ids)
    SECURITY_GROUP_IDS    = jsonencode(var.ecs_security_group_ids)
    ASSIGN_PUBLIC_IP      = var.ecs_assign_public_ip
    RESULTS_PREFIX_FARGATE = var.results_prefix_ecs_fargate
    RESULTS_PREFIX_EC2    = var.results_prefix_ecs_ec2
  }
}

# ---------------------------------------------------------------------------
# ECR — reference existing repository (not created by Terraform)
# ---------------------------------------------------------------------------

data "aws_ecr_repository" "main" {
  name = var.ecr_repository_name
}

# ---------------------------------------------------------------------------
# ECS — reference existing clusters (not created by Terraform)
# ---------------------------------------------------------------------------

data "aws_ecs_cluster" "fargate" {
  cluster_name = var.ecs_fargate_cluster_name
}

data "aws_ecs_cluster" "ec2" {
  cluster_name = var.ecs_ec2_cluster_name
}

# ---------------------------------------------------------------------------
# S3 — reference existing results bucket; create scripts bucket
# ---------------------------------------------------------------------------

data "aws_s3_bucket" "results" {
  bucket = var.results_bucket_name
}

data "aws_s3_bucket" "scripts" {
  bucket = var.scripts_bucket_name
}

# ---------------------------------------------------------------------------
# DynamoDB Table
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "context_store" {
  name         = "${var.project_name}-context-store"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "context_id"
  range_key    = "workflow_id"

  attribute {
    name = "context_id"
    type = "S"
  }

  attribute {
    name = "workflow_id"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "S"
  }

  global_secondary_index {
    name            = "workflow_id-created_at-index"
    hash_key        = "workflow_id"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# ElastiCache Redis (optional)
# ---------------------------------------------------------------------------

resource "aws_elasticache_cluster" "redis" {
  count = var.create_elasticache ? 1 : 0

  cluster_id           = "${var.project_name}-redis"
  engine               = "redis"
  node_type            = var.elasticache_node_type
  num_cache_nodes      = 1
  port                 = var.redis_port
  parameter_group_name = "default.redis7"

  subnet_group_name  = length(var.subnet_ids) > 0 ? aws_elasticache_subnet_group.redis[0].name : null
  security_group_ids = var.security_group_ids

  tags = local.common_tags
}

resource "aws_elasticache_subnet_group" "redis" {
  count = var.create_elasticache && length(var.subnet_ids) > 0 ? 1 : 0

  name       = "${var.project_name}-redis-subnet"
  subnet_ids = var.subnet_ids
}

# ---------------------------------------------------------------------------
# IAM Role — Lambda execution
# ---------------------------------------------------------------------------

resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem",
          "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan",
        ]
        Resource = [
          aws_dynamodb_table.context_store.arn,
          "${aws_dynamodb_table.context_store.arn}/index/*",
        ]
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket", "s3:PutObject"]
        Resource = [
          data.aws_s3_bucket.scripts.arn,
          "${data.aws_s3_bucket.scripts.arn}/*",
        ]
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_vpc" {
  count      = length(var.subnet_ids) > 0 ? 1 : 0
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# ---------------------------------------------------------------------------
# IAM Role — ECS task execution (ECR pull + CloudWatch logs)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-ecs-execution-role"

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

# public.ecr.aws requires these two permissions in addition to the managed policy
resource "aws_iam_role_policy" "ecs_execution_public_ecr" {
  count = var.create_ecs_redis ? 1 : 0
  name  = "${var.project_name}-ecs-execution-public-ecr"
  role  = aws_iam_role.ecs_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ecr-public:GetAuthorizationToken", "sts:GetServiceBearerToken"]
      Resource = "*"
    }]
  })
}

# ---------------------------------------------------------------------------
# IAM Role — ECS task role (what the container can do at runtime)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ecs-task-role"

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
  name = "${var.project_name}-ecs-task-policy"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/*"
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          # Results bucket (existing)
          data.aws_s3_bucket.results.arn,
          "${data.aws_s3_bucket.results.arn}/*",
          # Scripts bucket (hot-reload)
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
          aws_dynamodb_table.context_store.arn,
          "${aws_dynamodb_table.context_store.arn}/index/*",
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
# CloudWatch Log Group — ECS tasks
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ecs_python" {
  name              = "/ecs/${var.project_name}-python"
  retention_in_days = var.ecs_log_retention_days
  tags              = local.common_tags
}

# ---------------------------------------------------------------------------
# ECS Task Definition — Python armada runner
#
# Supports both FARGATE and EC2 launch types (awsvpc network mode).
# Static env vars baked in; per-run config injected via Step Functions
# ContainerOverrides (WORKFLOW_ID, TASKS_JSON, SCALING, WORLD_SIZE, etc.)
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "python_armada" {
  family                   = "${var.project_name}-python"
  requires_compatibilities = ["FARGATE", "EC2"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.ecs_python_cpu)
  memory                   = tostring(var.ecs_python_memory_mb)
  task_role_arn            = aws_iam_role.ecs_task.arn
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name      = var.ecs_container_name
    image     = "${data.aws_ecr_repository.main.repository_url}:${var.ecs_image_tag}"
    essential = true

    # Entry point for ECS experiments — reads per-run config from env vars
    command = ["python", "armada_ecs_runner.py"]

    environment = local.ecs_env

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_python.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Lambda Functions — Python (init / executor / aggregate)
#
# All three use the same image with different CMD overrides,
# following the cylon paper's dedicated-per-function pattern.
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "python_init" {
  function_name = "${var.project_name}-init"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.main.repository_url}:${var.python_image_tag}"
  memory_size   = var.python_memory_mb
  timeout       = var.lambda_timeout

  image_config {
    command = ["armada_init.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_function" "python_executor" {
  function_name = "${var.project_name}-executor"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.main.repository_url}:${var.python_image_tag}"
  memory_size   = var.python_memory_mb
  timeout       = var.lambda_timeout

  image_config {
    command = ["armada_executor.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_function" "python_aggregate" {
  function_name = "${var.project_name}-aggregate"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.main.repository_url}:${var.python_image_tag}"
  memory_size   = var.python_memory_mb
  timeout       = var.lambda_timeout

  image_config {
    command = ["armada_aggregate.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Lambda Functions — Node.js (init / executor / aggregate)
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "nodejs_init" {
  function_name = "${var.project_name}-init-node"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.main.repository_url}:${var.nodejs_image_tag}"
  memory_size   = var.nodejs_memory_mb
  timeout       = var.lambda_timeout

  image_config {
    command = ["armada_init.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_function" "nodejs_executor" {
  function_name = "${var.project_name}-executor-node"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.main.repository_url}:${var.nodejs_image_tag}"
  memory_size   = var.nodejs_memory_mb
  timeout       = var.lambda_timeout

  image_config {
    command = ["armada_executor.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_function" "nodejs_aggregate" {
  function_name = "${var.project_name}-aggregate-node"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.main.repository_url}:${var.nodejs_image_tag}"
  memory_size   = var.nodejs_memory_mb
  timeout       = var.lambda_timeout

  image_config {
    command = ["armada_aggregate.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# IAM Role — Step Functions execution
# ---------------------------------------------------------------------------

resource "aws_iam_role" "step_functions_execution" {
  name = "${var.project_name}-sfn-role"

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
  name = "${var.project_name}-sfn-policy"
  role = aws_iam_role.step_functions_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Invoke Lambda functions (all six)
      {
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = [
          aws_lambda_function.python_init.arn,
          aws_lambda_function.python_executor.arn,
          aws_lambda_function.python_aggregate.arn,
          aws_lambda_function.nodejs_init.arn,
          aws_lambda_function.nodejs_executor.arn,
          aws_lambda_function.nodejs_aggregate.arn,
        ]
      },
      # Run and monitor ECS tasks (used by ecs:runTask.sync)
      {
        Effect = "Allow"
        Action = [
          "ecs:RunTask",
          "ecs:StopTask",
          "ecs:DescribeTasks",
        ]
        Resource = ["*"]
      },
      # Pass IAM roles to ECS (required for runTask)
      {
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          aws_iam_role.ecs_task.arn,
          aws_iam_role.ecs_execution.arn,
        ]
      },
      # EventBridge integration for ecs:runTask.sync callback
      {
        Effect = "Allow"
        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule",
        ]
        Resource = ["arn:aws:events:${var.aws_region}:${var.account_id}:rule/StepFunctionsGetEventsForECSTaskRule"]
      },
      # CloudWatch logs for Express workflows
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
# Step Functions State Machines — Lambda workflows
# ---------------------------------------------------------------------------

resource "aws_sfn_state_machine" "python_workflow" {
  name     = "${var.project_name}-python-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "EXPRESS"

  definition = templatefile("${path.module}/../step_functions/workflow.asl.json", local.asl_vars)

  tags = local.common_tags
}

resource "aws_sfn_state_machine" "nodejs_workflow" {
  name     = "${var.project_name}-nodejs-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "EXPRESS"

  definition = templatefile("${path.module}/../step_functions/workflow_nodejs.asl.json", local.asl_vars)

  tags = local.common_tags
}

resource "aws_sfn_state_machine" "model_parallel_workflow" {
  name     = "${var.project_name}-model-parallel-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "EXPRESS"

  definition = templatefile("${path.module}/../step_functions/workflow_model_parallel.asl.json", local.asl_vars)

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Step Functions State Machines — ECS workflows
#
# Each workflow triggers a single ECS task (ecs:runTask.sync) that runs
# the full experiment — init, execute, aggregate — inside the container.
# Per-run config (workflow_id, tasks, scaling, world_size, etc.) is
# injected via ContainerOverrides at execution time.
# ---------------------------------------------------------------------------

resource "aws_sfn_state_machine" "ecs_fargate_workflow" {
  name     = "${var.project_name}-ecs-fargate-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/../step_functions/workflow_ecs_fargate.asl.json", local.ecs_asl_vars)

  tags = local.common_tags
}

resource "aws_sfn_state_machine" "ecs_ec2_workflow" {
  name     = "${var.project_name}-ecs-ec2-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/../step_functions/workflow_ecs_ec2.asl.json", local.ecs_asl_vars)

  tags = local.common_tags
}
# ---------------------------------------------------------------------------
# Redis — ECS Fargate service from ECR image (create_ecs_redis = true)
#
# Provides a Redis service backed by your own ECR image rather than
# ElastiCache. Reachable at redis.<project_name>.local:6379 within the VPC
# via Cloud Map private DNS.
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ecs_redis" {
  count             = var.create_ecs_redis ? 1 : 0
  name              = "/ecs/${var.project_name}-redis"
  retention_in_days = var.redis_log_retention_days
  tags              = local.common_tags
}

resource "aws_security_group" "redis" {
  count       = var.create_ecs_redis ? 1 : 0
  name        = "${var.project_name}-redis-sg"
  description = "Allow inbound Redis traffic from Lambda and ECS tasks"
  vpc_id      = var.vpc_id

  ingress {
    description = "Redis - open to internet (public IP Fargate task)"
    from_port   = var.redis_container_port
    to_port     = var.redis_container_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_ecs_task_definition" "redis" {
  count = var.create_ecs_redis ? 1 : 0

  family                   = "${var.project_name}-redis"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.redis_cpu
  memory                   = var.redis_memory_mb
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([{
    name      = "redis"
    image     = var.redis_image_uri
    essential = true

    portMappings = [{
      containerPort = var.redis_container_port
      protocol      = "tcp"
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_redis[0].name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "redis"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "redis-cli ping || exit 1"]
      interval    = 10
      timeout     = 5
      retries     = 3
      startPeriod = 15
    }
  }])

  tags = local.common_tags
}

resource "aws_ecs_service" "redis" {
  count = var.create_ecs_redis ? 1 : 0

  name            = "${var.project_name}-redis"
  cluster         = data.aws_ecs_cluster.fargate.id
  task_definition = aws_ecs_task_definition.redis[0].arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.ecs_task_subnet_ids
    security_groups  = [aws_security_group.redis[0].id]
    assign_public_ip = true
  }

  # Allow the container health check time to pass before ECS considers
  # the task unhealthy and replaces it (Redis needs ~15s to start).
  health_check_grace_period_seconds = 60

  # Prevent Terraform from restarting the service on every plan
  lifecycle {
    ignore_changes = [desired_count]
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Rendezvous test Lambda
#
# Validates FMI rendezvous server connectivity via a TCP connect check.
# Returns success/failure and round-trip latency in milliseconds.
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "rendezvous_test" {
  function_name = "${var.project_name}-rendezvous-test"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.main.repository_url}:${var.python_image_tag}"
  memory_size   = 256
  timeout       = 60

  image_config {
    command = ["rendezvous_test.handler"]
  }

  environment {
    variables = merge(local.lambda_env, {
      RENDEZVOUS_HOST = var.rendezvous_host
      RENDEZVOUS_PORT = tostring(var.rendezvous_port)
    })
  }

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Redis DNS updater — Lambda + EventBridge
#
# When ECS replaces an unhealthy Redis task the new task gets a new public IP.
# An EventBridge rule fires on every ECS Task State Change → RUNNING for the
# Redis service. The Lambda reads the ENI's public IP and upserts the Route 53
# A record so all clients resolve to the live task automatically.
#
# Only deployed when create_ecs_redis = true AND route53_zone_id is set.
# ---------------------------------------------------------------------------

locals {
  deploy_redis_dns = var.create_ecs_redis && var.route53_zone_id != "" && var.redis_hostname != ""
}

data "archive_file" "redis_dns_updater" {
  count = local.deploy_redis_dns ? 1 : 0

  type        = "zip"
  source_file = "${path.module}/../lambda/python/redis_dns_updater.py"
  output_path = "${path.module}/../lambda/python/redis_dns_updater.zip"
}

resource "aws_iam_role" "redis_dns_updater" {
  count = local.deploy_redis_dns ? 1 : 0
  name  = "${var.project_name}-redis-dns-updater-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "redis_dns_updater" {
  count = local.deploy_redis_dns ? 1 : 0
  name  = "${var.project_name}-redis-dns-updater-policy"
  role  = aws_iam_role.redis_dns_updater[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect   = "Allow"
        Action   = ["ec2:DescribeNetworkInterfaces"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["route53:ChangeResourceRecordSets"]
        Resource = "arn:aws:route53:::hostedzone/${var.route53_zone_id}"
      },
    ]
  })
}

resource "aws_lambda_function" "redis_dns_updater" {
  count = local.deploy_redis_dns ? 1 : 0

  function_name    = "${var.project_name}-redis-dns-updater"
  role             = aws_iam_role.redis_dns_updater[0].arn
  runtime          = "python3.12"
  handler          = "redis_dns_updater.handler"
  filename         = data.archive_file.redis_dns_updater[0].output_path
  source_code_hash = data.archive_file.redis_dns_updater[0].output_base64sha256
  memory_size      = 128
  timeout          = 30

  environment {
    variables = {
      ROUTE53_ZONE_ID = var.route53_zone_id
      REDIS_HOSTNAME  = var.redis_hostname
      REDIS_DNS_TTL   = tostring(var.redis_dns_ttl)
    }
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_event_rule" "redis_task_running" {
  count = local.deploy_redis_dns ? 1 : 0

  name        = "${var.project_name}-redis-task-running"
  description = "Fires when a Redis ECS task reaches RUNNING — triggers DNS update"

  event_pattern = jsonencode({
    source        = ["aws.ecs"]
    "detail-type" = ["ECS Task State Change"]
    detail = {
      lastStatus = ["RUNNING"]
      group      = ["service:${var.project_name}-redis"]
    }
  })

  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "redis_dns_updater" {
  count = local.deploy_redis_dns ? 1 : 0

  rule      = aws_cloudwatch_event_rule.redis_task_running[0].name
  target_id = "redis-dns-updater"
  arn       = aws_lambda_function.redis_dns_updater[0].arn
}

resource "aws_lambda_permission" "redis_dns_updater_eventbridge" {
  count = local.deploy_redis_dns ? 1 : 0

  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.redis_dns_updater[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.redis_task_running[0].arn
}
