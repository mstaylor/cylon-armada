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

variable "cpu_image_tag" {
  description = "ECR image tag for the CPU ECS image"
  type        = string
  default     = "cylon-armada-python"
}

# ---------------------------------------------------------------------------
# ECS — existing EC2 cluster (data source, not created by Terraform)
# ---------------------------------------------------------------------------

variable "ecs_ec2_cluster_name" {
  description = "Name of the existing ECS EC2 cluster"
  type        = string
  default     = "CylonEC2Experiments"
}

variable "ecs_container_name" {
  description = "Container name inside the ECS task definition"
  type        = string
  default     = "cylon-armada"
}

# ---------------------------------------------------------------------------
# EC2 instance configuration
# ---------------------------------------------------------------------------

variable "instance_type" {
  description = "EC2 instance type for ECS container instances (CPU)"
  type        = string
  default     = "m3.xlarge"
}

variable "asg_min_size" {
  description = "Minimum number of EC2 instances in the Auto Scaling Group"
  type        = number
  default     = 0
}

variable "asg_max_size" {
  description = "Maximum number of EC2 instances in the Auto Scaling Group"
  type        = number
  default     = 4
}

variable "asg_desired_capacity" {
  description = "Desired number of EC2 instances at apply time"
  type        = number
  default     = 1
}

variable "key_pair_name" {
  description = "EC2 key pair name for SSH access (leave empty to disable)"
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# ECS task definition — CPU sizing
# ---------------------------------------------------------------------------

variable "cpu_task_cpu" {
  description = "CPU units for the CPU ECS task (1 vCPU = 1024 units)"
  type        = number
  default     = 4096
}

variable "cpu_task_memory_mb" {
  description = "Memory for the CPU ECS task in MB"
  type        = number
  default     = 14336
}

variable "ecs_log_retention_days" {
  description = "CloudWatch log retention for ECS CPU tasks (days)"
  type        = number
  default     = 14
}

# ---------------------------------------------------------------------------
# ECS networking (awsvpc mode)
# ---------------------------------------------------------------------------

variable "ecs_task_subnet_ids" {
  description = "Subnet IDs for ECS CPU tasks (awsvpc network mode)"
  type        = list(string)
  default     = []
}

variable "ecs_security_group_ids" {
  description = "Security group IDs for ECS CPU tasks"
  type        = list(string)
  default     = []
}

# Subnets for the EC2 container instances themselves (ASG)
variable "ec2_subnet_ids" {
  description = "Subnet IDs for EC2 container instances (ASG)"
  type        = list(string)
  default     = []
}

variable "ec2_security_group_ids" {
  description = "Security group IDs for EC2 container instances"
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

variable "scripts_bucket_name" {
  description = "S3 bucket for ECS hot-reload scripts"
  type        = string
  default     = "staylor.dev2"
}

variable "s3_scripts_prefix" {
  description = "S3 key prefix for the scripts folder"
  type        = string
  default     = "cylon-armada/scripts/"
}

variable "results_bucket_name" {
  description = "Existing S3 bucket for experiment results"
  type        = string
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