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
    Environment = "ec2-experiments"
    ManagedBy   = "terraform"
  }

  ecs_cpu_env = [
    { name = "BEDROCK_LLM_MODEL_ID",        value = var.bedrock_llm_model_id },
    { name = "BEDROCK_EMBEDDING_MODEL_ID",   value = var.bedrock_embedding_model_id },
    { name = "BEDROCK_EMBEDDING_DIMENSIONS", value = tostring(var.bedrock_embedding_dimensions) },
    { name = "SIMILARITY_THRESHOLD",         value = tostring(var.similarity_threshold) },
    { name = "CONTEXT_BACKEND",              value = var.context_backend },
    { name = "REDIS_HOST",                   value = var.redis_host },
    { name = "REDIS_PORT",                   value = tostring(var.redis_port) },
    { name = "DYNAMO_TABLE_NAME",            value = var.dynamo_table_name },
    { name = "RESULTS_BUCKET",               value = var.results_bucket_name },
    { name = "AWS_DEFAULT_REGION",           value = var.aws_region },
    { name = "SIMD_BACKEND",                 value = "numpy" },
    { name = "COMPUTE_PLATFORM",             value = "ecs-ec2" },
  ]

  asl_vars = {
    ECS_EC2_CLUSTER        = data.aws_ecs_cluster.ec2.arn
    PYTHON_TASK_DEF        = aws_ecs_task_definition.cpu_armada.arn
    CONTAINER_NAME         = var.ecs_container_name
    SUBNET_IDS             = jsonencode(var.ecs_task_subnet_ids)
    SECURITY_GROUP_IDS     = jsonencode(var.ecs_security_group_ids)
    CAPACITY_PROVIDER_NAME = aws_ecs_capacity_provider.cpu.name
  }
}

# ---------------------------------------------------------------------------
# Data sources — existing resources
# ---------------------------------------------------------------------------

data "aws_ecr_repository" "main" {
  name = var.ecr_repository_name
}

data "aws_ecs_cluster" "ec2" {
  cluster_name = var.ecs_ec2_cluster_name
}

data "aws_s3_bucket" "results" {
  bucket = var.results_bucket_name
}

data "aws_s3_bucket" "scripts" {
  bucket = var.scripts_bucket_name
}

# Latest ECS-optimized Amazon Linux 2 AMI
data "aws_ssm_parameter" "ecs_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id"
}

# ---------------------------------------------------------------------------
# CloudWatch Log Group — CPU ECS tasks
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ecs_cpu" {
  name              = "/ecs/${var.project_name}-ec2"
  retention_in_days = var.ecs_log_retention_days
  tags              = local.common_tags
}

# ---------------------------------------------------------------------------
# IAM — EC2 instance profile (registers instance to ECS cluster)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ec2_instance" {
  name = "${var.project_name}-ec2-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ec2_ecs_managed" {
  role       = aws_iam_role.ec2_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy_attachment" "ec2_ssm_managed" {
  role       = aws_iam_role.ec2_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_instance" {
  name = "${var.project_name}-ec2-instance-profile"
  role = aws_iam_role.ec2_instance.name
  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# IAM — ECS task execution role (ECR pull + CloudWatch logs)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-ec2-ecs-execution-role"

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
# IAM — ECS task role (what the container can do at runtime)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ec2-ecs-task-role"

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
  name = "${var.project_name}-ec2-ecs-task-policy"
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
      {
        Effect   = "Allow"
        Action   = ["pricing:GetProducts"]
        Resource = "*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# IAM — Step Functions execution role
# ---------------------------------------------------------------------------

resource "aws_iam_role" "step_functions_execution" {
  name = "${var.project_name}-ec2-sfn-role"

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
  name = "${var.project_name}-ec2-sfn-policy"
  role = aws_iam_role.step_functions_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:RunTask", "ecs:StopTask", "ecs:DescribeTasks"]
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
      {
        Effect = "Allow"
        Action = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
        Resource = ["arn:aws:events:${var.aws_region}:${var.account_id}:rule/StepFunctionsGetEventsForECSTaskRule"]
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

# ---------------------------------------------------------------------------
# EC2 Launch Template — ECS-optimized AMI, registers to cluster via user_data
# ---------------------------------------------------------------------------

resource "aws_launch_template" "ecs_cpu" {
  name_prefix   = "${var.project_name}-ec2-"
  image_id      = data.aws_ssm_parameter.ecs_ami.value
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.ec2_instance.name
  }

  key_name = var.key_pair_name != "" ? var.key_pair_name : null

  vpc_security_group_ids = length(var.ec2_security_group_ids) > 0 ? var.ec2_security_group_ids : null

  user_data = base64encode(<<-EOF
    #!/bin/bash
    echo ECS_CLUSTER=${var.ecs_ec2_cluster_name} >> /etc/ecs/ecs.config
    echo ECS_ENABLE_AWSLOGS_EXECUTIONROLE_OVERRIDE=true >> /etc/ecs/ecs.config
  EOF
  )

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    http_endpoint               = "enabled"
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name = "${var.project_name}-ecs-ec2-instance"
    })
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Auto Scaling Group — manages the pool of ECS container instances
# ---------------------------------------------------------------------------

resource "aws_autoscaling_group" "ecs_cpu" {
  name                = "${var.project_name}-ecs-ec2-asg"
  min_size            = var.asg_min_size
  max_size            = var.asg_max_size
  desired_capacity    = var.asg_desired_capacity
  vpc_zone_identifier = var.ec2_subnet_ids

  launch_template {
    id      = aws_launch_template.ecs_cpu.id
    version = "$Latest"
  }

  # Let capacity provider manage scale-in protection
  protect_from_scale_in = true

  tag {
    key                 = "AmazonECSManaged"
    value               = true
    propagate_at_launch = true
  }

  dynamic "tag" {
    for_each = local.common_tags
    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }
}

# ---------------------------------------------------------------------------
# ECS Capacity Provider — wires ASG to the cluster for managed scaling
# ---------------------------------------------------------------------------

resource "aws_ecs_capacity_provider" "cpu" {
  name = "${var.project_name}-ec2-capacity-provider"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.ecs_cpu.arn
    managed_termination_protection = "ENABLED"

    managed_scaling {
      status                    = "ENABLED"
      target_capacity           = 100
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 4
    }
  }

  tags = local.common_tags
}

resource "aws_ecs_cluster_capacity_providers" "cpu" {
  cluster_name       = data.aws_ecs_cluster.ec2.cluster_name
  capacity_providers = [aws_ecs_capacity_provider.cpu.name]

  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.cpu.name
    weight            = 1
    base              = 0
  }
}

# ---------------------------------------------------------------------------
# ECS Task Definition — CPU armada runner
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "cpu_armada" {
  family                   = "${var.project_name}-ec2"
  requires_compatibilities = ["EC2"]
  network_mode             = "host"
  cpu                      = tostring(var.cpu_task_cpu)
  memory                   = tostring(var.cpu_task_memory_mb)
  task_role_arn            = aws_iam_role.ecs_task.arn
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name      = var.ecs_container_name
    image     = "${data.aws_ecr_repository.main.repository_url}:${var.cpu_image_tag}"
    essential = true

    entryPoint = ["/cylon/target/aws/scripts/lambda/runCyloninLambda.sh"]
    command     = ["python", "armada_ecs_runner.py"]

    environment = local.ecs_cpu_env

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_cpu.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs-ec2"
      }
    }
  }])

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Step Functions State Machine — ECS EC2 CPU workflow
# ---------------------------------------------------------------------------

resource "aws_sfn_state_machine" "ecs_ec2_workflow" {
  name     = "${var.project_name}-ecs-ec2-cpu-workflow"
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "STANDARD"

  definition = templatefile(
    "${path.module}/../step_functions/workflow_ecs_ec2.asl.json",
    local.asl_vars
  )

  tags = local.common_tags
}