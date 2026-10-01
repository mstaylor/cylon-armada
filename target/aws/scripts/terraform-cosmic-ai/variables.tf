variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile for the destination account"
  type        = string
  default     = "cosmicai"
}

variable "account_id" {
  description = "Destination AWS account ID (the cosmicai profile's account)"
  type        = string
  default     = "881908115028"
}

variable "project_name" {
  description = "Name prefix for all resources created by this module"
  type        = string
  default     = "cosmic-ai"
}

variable "data_bucket_name" {
  description = "Existing S3 bucket (in this account) holding SDSS partitions and results"
  type        = string
  default     = "cosmicai"
}

variable "python_runtime" {
  description = "AWS Lambda Python runtime for the zip-packaged init/summarize functions"
  type        = string
  default     = "python3.12"
}

variable "executor_image_uri" {
  description = "ECR image URI built from docker/cosmic-ai-lambda/Dockerfile (lambda_entry3.handler entry point) — no default, must be built and pushed before apply"
  type        = string
}

variable "init_memory_mb" {
  description = "Memory for the initializer Lambda"
  type        = number
  default     = 512
}

variable "executor_memory_mb" {
  description = "Memory for the inference (AstroMAE model) Lambda"
  type        = number
  default     = 3008
}

variable "summarize_memory_mb" {
  description = "Memory for the summarizer Lambda"
  type        = number
  default     = 1024
}

variable "lambda_timeout" {
  description = "Timeout (seconds) for all three Lambdas"
  type        = number
  default     = 300
}

variable "subnet_ids" {
  description = "Optional VPC subnet IDs for the Lambdas (empty = no VPC)"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Optional VPC security group IDs for the Lambdas"
  type        = list(string)
  default     = []
}

variable "max_concurrency" {
  description = "Distributed Map MaxConcurrency (published state machine used 3000)"
  type        = number
  default     = 3000
}

variable "state_machine_timeout_seconds" {
  description = "Overall state machine timeout"
  type        = number
  default     = 3600
}

variable "log_retention_days" {
  description = "CloudWatch log retention for the state machine's Express logs"
  type        = number
  default     = 14
}