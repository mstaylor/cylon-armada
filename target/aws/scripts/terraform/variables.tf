variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "cylon-armada"
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
}

# ---------------------------------------------------------------------------
# ECR (existing repository — not created by Terraform)
# ---------------------------------------------------------------------------

variable "ecr_repository_name" {
  description = "Name of the existing ECR repository for cylon-armada images"
  type        = string
}

# ---------------------------------------------------------------------------
# Lambda configuration
# ---------------------------------------------------------------------------

variable "python_image_tag" {
  description = "ECR image tag for the Python Lambda image"
  type        = string
  default     = "python-latest"
}

variable "nodejs_image_tag" {
  description = "ECR image tag for the Node.js Lambda image"
  type        = string
  default     = "nodejs-latest"
}

variable "python_memory_mb" {
  description = "Memory for Python Lambda (MB)"
  type        = number
  default     = 1024
}

variable "nodejs_memory_mb" {
  description = "Memory for Node.js Lambda (MB)"
  type        = number
  default     = 512
}

variable "lambda_timeout" {
  description = "Lambda timeout (seconds)"
  type        = number
  default     = 300
}

# ---------------------------------------------------------------------------
# ECS — existing clusters (data sources, not created by Terraform)
# ---------------------------------------------------------------------------

variable "ecs_ec2_cluster_name" {
  description = "Name of the existing ECS EC2 cluster"
  type        = string
  default     = "CylonEC2Experiments"
}

variable "ecs_fargate_cluster_name" {
  description = "Name of the existing ECS Fargate cluster"
  type        = string
  default     = "CylonFargateExperiments"
}

# ---------------------------------------------------------------------------
# ECS task definition
# ---------------------------------------------------------------------------

variable "ecs_container_name" {
  description = "Container name inside the ECS task definition"
  type        = string
  default     = "cylon-armada"
}

variable "ecs_python_cpu" {
  description = "CPU units for the Python ECS task (1024 = 1 vCPU)"
  type        = number
  default     = 4096
}

variable "ecs_python_memory_mb" {
  description = "Memory for the Python ECS task (MB)"
  type        = number
  default     = 8192
}

variable "ecs_image_tag" {
  description = "ECR image tag for the ECS Python runner image"
  type        = string
  default     = "python-latest"
}

variable "ecs_log_retention_days" {
  description = "CloudWatch log retention for ECS tasks (days)"
  type        = number
  default     = 7
}

# ---------------------------------------------------------------------------
# ECS networking (awsvpc mode — required for Fargate and EC2+awsvpc)
# ---------------------------------------------------------------------------

variable "ecs_task_subnet_ids" {
  description = "Subnet IDs for ECS tasks (awsvpc network mode)"
  type        = list(string)
  default     = []
}

variable "ecs_security_group_ids" {
  description = "Security group IDs for ECS tasks"
  type        = list(string)
  default     = []
}

variable "ecs_assign_public_ip" {
  description = "Assign public IP to Fargate tasks (ENABLED or DISABLED)"
  type        = string
  default     = "ENABLED"
}

# ---------------------------------------------------------------------------
# Redis / ElastiCache
# ---------------------------------------------------------------------------

variable "redis_host" {
  description = "Redis host (ElastiCache endpoint or existing instance)"
  type        = string
  default     = ""
}

variable "redis_port" {
  description = "Redis port"
  type        = number
  default     = 6379
}

variable "create_elasticache" {
  description = "Create a new ElastiCache Redis cluster"
  type        = bool
  default     = false
}

variable "elasticache_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.micro"
}

# ---------------------------------------------------------------------------
# Networking (Lambda VPC / ElastiCache)
# ---------------------------------------------------------------------------

variable "vpc_id" {
  description = "VPC ID for Lambda and ElastiCache"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for Lambda VPC config and ElastiCache"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Security group IDs for Lambda"
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

variable "scripts_bucket_name" {
  description = "S3 bucket for Lambda/ECS hot-reload scripts"
  type        = string
  default     = "cylon-armada-scripts"
}

variable "results_bucket_name" {
  description = "Existing S3 bucket for experiment results"
  type        = string
}

variable "results_prefix_lambda" {
  description = "S3 key prefix for Lambda experiment results"
  type        = string
  default     = "results/lambda/"
}

variable "results_prefix_ecs_fargate" {
  description = "S3 key prefix for ECS Fargate experiment results"
  type        = string
  default     = "results/ecs-fargate/"
}

variable "results_prefix_ecs_ec2" {
  description = "S3 key prefix for ECS EC2 experiment results"
  type        = string
  default     = "results/ecs-ec2/"
}

# ---------------------------------------------------------------------------
# Step Functions
# ---------------------------------------------------------------------------

variable "step_functions_max_concurrency" {
  description = "Max concurrent Lambda workers in the Map state"
  type        = number
  default     = 10
}

# ---------------------------------------------------------------------------
# Bedrock defaults (injected as env vars into Lambda and ECS tasks)
# ---------------------------------------------------------------------------

variable "bedrock_llm_model_id" {
  description = "Default Bedrock LLM model ID"
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

variable "bedrock_embedding_model_id" {
  description = "Default Bedrock embedding model ID"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "bedrock_embedding_dimensions" {
  description = "Default embedding dimensions"
  type        = number
  default     = 1024
}

variable "similarity_threshold" {
  description = "Default cosine similarity threshold"
  type        = number
  default     = 0.85
}

variable "context_backend" {
  description = "Context store backend: cylon or redis"
  type        = string
  default     = "cylon"
}

# ---------------------------------------------------------------------------
# Rendezvous server (FMI direct channel)
# ---------------------------------------------------------------------------

variable "rendezvous_host" {
  description = "Rendezvous server hostname for FMI direct (TCPunch) channel"
  type        = string
  default     = ""
}

variable "rendezvous_port" {
  description = "Rendezvous server port"
  type        = number
  default     = 10000
}

# ---------------------------------------------------------------------------
# Redis — ECS Fargate service from ECR image
# ---------------------------------------------------------------------------

variable "create_ecs_redis" {
  description = "Deploy Redis as an ECS Fargate service using an ECR image"
  type        = bool
  default     = false
}

variable "redis_ecr_repository_name" {
  description = "Name of the ECR repository containing the Redis image"
  type        = string
  default     = ""
}

variable "redis_image_tag" {
  description = "ECR image tag for the Redis container"
  type        = string
  default     = "redis-latest"
}

variable "redis_container_port" {
  description = "Port Redis listens on inside the container"
  type        = number
  default     = 6379
}

variable "redis_cpu" {
  description = "CPU units for the Redis ECS task (256 = 0.25 vCPU)"
  type        = number
  default     = 256
}

variable "redis_memory_mb" {
  description = "Memory for the Redis ECS task (MB)"
  type        = number
  default     = 512
}

variable "redis_log_retention_days" {
  description = "CloudWatch log retention for the Redis ECS task (days)"
  type        = number
  default     = 7
}