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

variable "gpu_image_tag" {
  description = "ECR image tag for the GPU ECS image"
  type        = string
  default     = "cylon-armada-gpu"
}

# ---------------------------------------------------------------------------
# ECS — existing EC2 cluster (data source, not created by Terraform)
# ---------------------------------------------------------------------------

variable "ecs_ec2_cluster_name" {
  description = "Name of the existing ECS EC2 cluster with GPU instances"
  type        = string
  default     = "CylonEC2Experiments"
}

# ---------------------------------------------------------------------------
# ECS task definition
# ---------------------------------------------------------------------------

variable "ecs_container_name" {
  description = "Container name inside the ECS task definition"
  type        = string
  default     = "cylon-armada"
}

variable "gpu_cpu" {
  description = "CPU units for the GPU ECS task (g4dn.xlarge has 4 vCPUs = 4096 units)"
  type        = number
  default     = 4096
}

variable "gpu_memory_mb" {
  description = "Memory for the GPU ECS task in MB (g4dn.xlarge has 16 GB)"
  type        = number
  default     = 14336
}

variable "gpu_count" {
  description = "Number of GPUs to allocate per task"
  type        = number
  default     = 1
}

variable "ecs_log_retention_days" {
  description = "CloudWatch log retention for ECS GPU tasks (days)"
  type        = number
  default     = 14
}

# ---------------------------------------------------------------------------
# ECS networking (awsvpc mode)
# ---------------------------------------------------------------------------

variable "ecs_task_subnet_ids" {
  description = "Subnet IDs for ECS GPU tasks (awsvpc network mode)"
  type        = list(string)
  default     = []
}

variable "ecs_security_group_ids" {
  description = "Security group IDs for ECS GPU tasks"
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

variable "scripts_bucket_name" {
  description = "S3 bucket for ECS hot-reload scripts"
  type        = string
  default     = "cylon-armada-scripts"
}

variable "s3_scripts_prefix" {
  description = "S3 key prefix for the scripts folder"
  type        = string
  default     = "scripts/"
}

variable "results_bucket_name" {
  description = "Existing S3 bucket for experiment results"
  type        = string
}

variable "results_prefix_ecs_ec2_gpu" {
  description = "S3 key prefix for ECS EC2 GPU experiment results"
  type        = string
  default     = "results/ecs-ec2-gpu/"
}

# ---------------------------------------------------------------------------
# Bedrock defaults
# ---------------------------------------------------------------------------

variable "bedrock_llm_model_id" {
  description = "Default Bedrock LLM model ID"
  type        = string
  default     = "amazon.nova-lite-v1:0"
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
# Redis
# ---------------------------------------------------------------------------

variable "redis_host" {
  description = "Redis host (ElastiCache endpoint or external)"
  type        = string
  default     = ""
}

variable "redis_port" {
  description = "Redis port"
  type        = number
  default     = 6379
}

# ---------------------------------------------------------------------------
# DynamoDB (existing table — shared with main Terraform state)
# ---------------------------------------------------------------------------

variable "dynamo_table_name" {
  description = "Name of the existing DynamoDB context store table"
  type        = string
  default     = "cylon-armada-context-store"
}