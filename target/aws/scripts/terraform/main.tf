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

  lambda_env = {
    BEDROCK_LLM_MODEL_ID       = var.bedrock_llm_model_id
    BEDROCK_EMBEDDING_MODEL_ID = var.bedrock_embedding_model_id
    BEDROCK_EMBEDDING_DIMENSIONS = tostring(var.bedrock_embedding_dimensions)
    SIMILARITY_THRESHOLD       = tostring(var.similarity_threshold)
    CONTEXT_BACKEND            = var.context_backend
    REDIS_HOST                 = var.create_elasticache ? aws_elasticache_cluster.redis[0].cache_nodes[0].address : var.redis_host
    REDIS_PORT                 = tostring(var.redis_port)
    DYNAMO_TABLE_NAME          = aws_dynamodb_table.context_store.name
    AWS_DEFAULT_REGION         = var.aws_region
  }
}

# ---------------------------------------------------------------------------
# ECR Repositories
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "python" {
  name                 = "${var.project_name}-python"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = local.common_tags
}

resource "aws_ecr_repository" "nodejs" {
  name                 = "${var.project_name}-nodejs"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# S3 Bucket (Lambda scripts)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "scripts" {
  bucket        = var.scripts_bucket_name
  force_destroy = true

  tags = local.common_tags
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
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
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
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan",
        ]
        Resource = [
          aws_dynamodb_table.context_store.arn,
          "${aws_dynamodb_table.context_store.arn}/index/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject",
        ]
        Resource = [
          aws_s3_bucket.scripts.arn,
          "${aws_s3_bucket.scripts.arn}/*",
        ]
      },
    ]
  })
}

# VPC access (if configured)
resource "aws_iam_role_policy_attachment" "lambda_vpc" {
  count      = length(var.subnet_ids) > 0 ? 1 : 0
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# ---------------------------------------------------------------------------
# Lambda Functions
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "python_worker" {
  function_name = "${var.project_name}-worker"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = var.python_image_uri
  memory_size   = var.python_memory_mb
  timeout       = var.lambda_timeout

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

resource "aws_lambda_function" "nodejs_worker" {
  function_name = "${var.project_name}-worker-node"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = var.nodejs_image_uri
  memory_size   = var.nodejs_memory_mb
  timeout       = var.lambda_timeout

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
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "states.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "step_functions_policy" {
  name = "${var.project_name}-sfn-policy"
  role = aws_iam_role.step_functions_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "lambda:InvokeFunction",
      ]
      Resource = [
        aws_lambda_function.python_worker.arn,
        aws_lambda_function.nodejs_worker.arn,
      ]
    }]
  })
}

# ---------------------------------------------------------------------------
# Step Functions State Machines
# ---------------------------------------------------------------------------

resource "aws_sfn_state_machine" "python_workflow" {
  name     = "${var.project_name}-python-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "EXPRESS"

  definition = templatefile("${path.module}/../step_functions/workflow.asl.json", {
    ACCOUNT_ID = var.account_id
  })

  tags = local.common_tags
}

resource "aws_sfn_state_machine" "nodejs_workflow" {
  name     = "${var.project_name}-nodejs-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "EXPRESS"

  definition = templatefile("${path.module}/../step_functions/workflow_nodejs.asl.json", {
    ACCOUNT_ID = var.account_id
  })

  tags = local.common_tags
}

resource "aws_sfn_state_machine" "model_parallel_workflow" {
  name     = "${var.project_name}-model-parallel-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "EXPRESS"

  definition = templatefile("${path.module}/../step_functions/workflow_model_parallel.asl.json", {
    ACCOUNT_ID = var.account_id
  })

  tags = local.common_tags
}
