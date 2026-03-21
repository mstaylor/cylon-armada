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

# Lambda configuration
variable "python_image_uri" {
  description = "ECR image URI for Python Lambda (Path A1/A2)"
  type        = string
}

variable "nodejs_image_uri" {
  description = "ECR image URI for Node.js Lambda (Path B)"
  type        = string
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

# Redis / ElastiCache
variable "redis_host" {
  description = "Redis host (ElastiCache endpoint or existing)"
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

# Networking
variable "vpc_id" {
  description = "VPC ID for Lambda and ElastiCache"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for Lambda and ElastiCache"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Security group IDs for Lambda"
  type        = list(string)
  default     = []
}

# S3
variable "scripts_bucket_name" {
  description = "S3 bucket for Lambda scripts"
  type        = string
  default     = "cylon-armada-scripts"
}

# Step Functions
variable "step_functions_max_concurrency" {
  description = "Max concurrent Lambda workers in Map state"
  type        = number
  default     = 10
}

# Bedrock defaults (passed as Lambda env vars)
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