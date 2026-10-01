terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = "track1-lambda-dataplane"
    ManagedBy   = "terraform"
  }

  init_function_name      = "${var.project_name}-init"
  executor_function_name  = "${var.project_name}-executor"
  summarize_function_name = "${var.project_name}-summarize"

  asl_vars = {
    AWS_REGION              = var.aws_region
    ACCOUNT_ID              = var.account_id
    INIT_FUNCTION_NAME      = local.init_function_name
    EXECUTOR_FUNCTION_NAME  = local.executor_function_name
    SUMMARIZE_FUNCTION_NAME = local.summarize_function_name
    MAX_CONCURRENCY         = var.max_concurrency
    TIMEOUT_SECONDS         = var.state_machine_timeout_seconds
  }
}

data "aws_s3_bucket" "data" {
  bucket = var.data_bucket_name
}

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
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          data.aws_s3_bucket.data.arn,
          "${data.aws_s3_bucket.data.arn}/*",
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

data "archive_file" "init_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/initializer.py"
  output_path = "${path.module}/.build/initializer.zip"
}

data "archive_file" "summarize_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/summarizer.py"
  output_path = "${path.module}/.build/summarizer.zip"
}

resource "aws_lambda_function" "init" {
  function_name    = local.init_function_name
  role             = aws_iam_role.lambda_execution.arn
  package_type     = "Zip"
  filename         = data.archive_file.init_zip.output_path
  source_code_hash = data.archive_file.init_zip.output_base64sha256
  handler          = "initializer.lambda_handler"
  runtime          = var.python_runtime
  memory_size      = var.init_memory_mb
  timeout          = var.lambda_timeout

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_function" "executor" {
  function_name = local.executor_function_name
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = var.executor_image_uri
  memory_size   = var.executor_memory_mb
  timeout       = var.lambda_timeout

  image_config {
    command = ["lambda_entry3.handler"]
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

resource "aws_lambda_function" "summarize" {
  function_name    = local.summarize_function_name
  role             = aws_iam_role.lambda_execution.arn
  package_type     = "Zip"
  filename         = data.archive_file.summarize_zip.output_path
  source_code_hash = data.archive_file.summarize_zip.output_base64sha256
  handler          = "summarizer.lambda_handler"
  runtime          = var.python_runtime
  memory_size      = var.summarize_memory_mb
  timeout          = var.lambda_timeout

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

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
      {
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = [
          aws_lambda_function.init.arn,
          aws_lambda_function.executor.arn,
          aws_lambda_function.summarize.arn,
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${data.aws_s3_bucket.data.arn}/*"]
      },
      {
        Effect = "Allow"
        Action = ["states:StartExecution"]
        Resource = [
          "arn:aws:states:${var.aws_region}:${var.account_id}:stateMachine:${var.project_name}-workflow"
        ]
      },
      {
        Effect = "Allow"
        Action = ["states:DescribeExecution", "states:StopExecution"]
        Resource = [
          "arn:aws:states:${var.aws_region}:${var.account_id}:execution:${var.project_name}-workflow:*"
        ]
      },
      {
        Effect = "Allow"
        Action = ["logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery", "logs:ListLogDeliveries", "logs:PutResourcePolicy",
        "logs:DescribeResourcePolicies", "logs:DescribeLogGroups"]
        Resource = ["*"]
      },
    ]
  })
}

resource "aws_cloudwatch_log_group" "sfn_workflow" {
  name              = "/aws/vendedlogs/states/${var.project_name}-workflow"
  retention_in_days = var.log_retention_days
  tags              = local.common_tags
}

resource "aws_cloudwatch_log_resource_policy" "sfn_workflow" {
  policy_name = "${var.project_name}-sfn-log-policy"
  policy_document = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "delivery.logs.amazonaws.com" }
        Action    = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource  = "${aws_cloudwatch_log_group.sfn_workflow.arn}:*"
      },
      {
        Effect    = "Allow"
        Principal = { Service = "states.amazonaws.com" }
        Action = ["logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery", "logs:ListLogDeliveries",
        "logs:PutResourcePolicy", "logs:DescribeResourcePolicies", "logs:DescribeLogGroups"]
        Resource = "*"
      },
    ]
  })
}

resource "aws_sfn_state_machine" "workflow" {
  name     = "${var.project_name}-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/../step_functions/workflow_cosmic_ai.asl.json", local.asl_vars)

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.sfn_workflow.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tags = local.common_tags
}