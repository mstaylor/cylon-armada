variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile; the cylon account by default, cosmicai.tfvars overrides it"
  type        = string
  default     = "cylon"
}

variable "account_id" {
  description = "Account the plan must run against; the plan fails if the credentials belong to another"
  type        = string
  default     = "448324707516"
}

variable "project_name" {
  description = "Prefix for every resource; matches the cylon account names so the campaign driver works in both"
  type        = string
  default     = "cylon-armada-cosmic-ai"
}

variable "data_bucket_name" {
  description = "Bucket holding the partitions, the Anomaly Detection folder, payloads and results"
  type        = string
  default     = "cosmicai-data-cylon"
}

variable "ecr_repository_name" {
  description = "ECR repository holding the executor image"
  type        = string
  default     = "cylon-armada"
}

variable "executor_image_tag" {
  description = "Executor image tag; resolved to a digest at plan time so a push and an apply redeploy the executor"
  type        = string
  default     = "cosmic-ai-executor"
}

variable "model_key" {
  description = "Key of the AstroMAE weights in the data bucket, inside the Anomaly Detection folder the executor downloads"
  type        = string
  default     = "Anomaly Detection/Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt"
}

variable "script_prefix" {
  description = "Prefix in the data bucket where the executor finds inference.py, inference_FMI.py and the communicator package"
  type        = string
  default     = "Anomaly Detection/Inference"
}

variable "python_runtime" {
  description = "Python runtime for the zip-packaged init and summarize Lambdas"
  type        = string
  default     = "python3.12"
}

variable "init_memory_mb" {
  description = "Memory for the init Lambdas"
  type        = number
  default     = 128
}

variable "summarize_memory_mb" {
  description = "Memory for the summarize Lambdas"
  type        = number
  default     = 128
}

variable "init_timeout" {
  description = "Timeout in seconds for the init and summarize Lambdas"
  type        = number
  default     = 150
}

variable "executor_memory_mb" {
  description = "Executor memory; 10240 matches the measured runs in the cylon account"
  type        = number
  default     = 10240
}

variable "executor_timeout" {
  description = "Executor timeout in seconds"
  type        = number
  default     = 900
}

variable "subnet_ids" {
  description = "Subnets for the Lambdas; empty runs them outside a VPC, which Arm B needs to reach the rendezvous server"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Security groups for the Lambdas when subnet_ids is set"
  type        = list(string)
  default     = []
}

variable "max_concurrency" {
  description = "Distributed Map MaxConcurrency"
  type        = number
  default     = 3000
}

variable "state_machine_timeout_seconds" {
  description = "Overall state machine timeout"
  type        = number
  default     = 3600
}

variable "executor_retry_errors" {
  description = "Errors Arm A's Distributed Map retries an executor invocation on; Arm A ranks are independent, so re-running one is safe"
  type        = list(string)
  default = [
    "Lambda.ServiceException",
    "Lambda.AWSLambdaException",
    "Lambda.SdkClientException",
    "Lambda.TooManyRequestsException",
  ]

  validation {
    condition     = length(var.executor_retry_errors) > 0
    error_message = "executor_retry_errors must not be empty; Step Functions rejects an empty ErrorEquals."
  }
}

variable "fmi_executor_retry_errors" {
  description = "Errors Arm B's Distributed Map retries an executor invocation on; only throttling, since re-invoking a rank that already joined the FMI communicator deadlocks or duplicates the rank"
  type        = list(string)
  default     = ["Lambda.TooManyRequestsException"]

  validation {
    condition     = length(var.fmi_executor_retry_errors) > 0
    error_message = "fmi_executor_retry_errors must not be empty; Step Functions rejects an empty ErrorEquals."
  }
}

variable "fmi_channel_type" {
  description = "FMI channel Arm B ranks pair over; Lambda accepts no inbound connections, so only the TCPunch direct channel applies"
  type        = string
  default     = "direct"
}

variable "fmi_options" {
  description = "FMI communicator mode for Arm B ranks"
  type        = string
  default     = "nonblocking"

  validation {
    condition     = contains(["nonblocking", "blocking"], var.fmi_options)
    error_message = "fmi_options must be nonblocking or blocking, matching inference_FMI.py's --fmi_options choices."
  }
}

variable "fmi_max_timeout" {
  description = "FMI communicator max timeout in milliseconds"
  type        = number
  default     = 300000
}

variable "rendezvous_host" {
  description = "TCPunch rendezvous server host for Arm B (the user's own deployment, reached over the public internet)"
  type        = string
  default     = "cylon-rendezvous.aws-cylondata.com"
}

variable "rendezvous_port" {
  description = "TCPunch rendezvous server port"
  type        = number
  default     = 10000
}

variable "sfn_log_level" {
  description = "Step Functions log level for both workflows; ERROR records failed runs without logging every state transition"
  type        = string
  default     = "ERROR"

  validation {
    condition     = contains(["ALL", "ERROR", "FATAL", "OFF"], var.sfn_log_level)
    error_message = "sfn_log_level must be ALL, ERROR, FATAL or OFF."
  }
}

variable "log_retention_days" {
  description = "CloudWatch log retention for the workflow log group"
  type        = number
  default     = 14
}