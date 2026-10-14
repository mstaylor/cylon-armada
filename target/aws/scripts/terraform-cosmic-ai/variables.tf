variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "accounts" {
  description = "Per-account settings keyed by Terraform workspace; the selected workspace picks the account (default means cylon). The plan fails if the credentials belong to another account"
  type = map(object({
    aws_profile         = string
    account_id          = string
    data_bucket_name    = string
    ecr_repository_name = string
    executor_image_tag  = string
  }))
  default = {
    cylon = {
      aws_profile         = "cylon"
      account_id          = "448324707516"
      data_bucket_name    = "cosmicai-data-cylon"
      ecr_repository_name = "cylon-armada"
      executor_image_tag  = "cosmic-ai-executor"
    }
    cosmicai = {
      aws_profile         = "cosmicai-admin"
      account_id          = "881908115028"
      data_bucket_name    = "cosmicai"
      ecr_repository_name = "cosmic-ai-executor"
      executor_image_tag  = "latest"
    }
  }
}

variable "default_workspace_account" {
  description = "Account the default workspace deploys to"
  type        = string
  default     = "cylon"
}

variable "project_name" {
  description = "Prefix for every resource; matches the cylon account names so the campaign driver works in both"
  type        = string
  default     = "cylon-armada-cosmic-ai"
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

variable "abort_poll_s" {
  description = "Seconds between each Arm B worker's checks for its run's S3 abort marker; a worker that sees it kills its inference process"
  type        = number
  default     = 5

  validation {
    condition     = var.abort_poll_s > 0
    error_message = "abort_poll_s must be positive."
  }
}

variable "fmi_holepunch_connect_timeout_ms" {
  description = "Milliseconds cylon's direct channel waits for one TCPunch hole punch, during connection setup and on first use, before retrying it"
  type        = number
  default     = 10000
}

variable "fmi_tcpunch_max_tries" {
  description = "Hole-punch attempts per peer before cylon's direct channel raises a timeout"
  type        = number
  default     = 6
}

variable "fmi_establish_parallelism" {
  description = "Connections cylon's direct channel establishes at once per rank; rank 0 pairs with every rank for the gather, so this bounds its setup time"
  type        = number
  default     = 64

  validation {
    condition     = var.fmi_establish_parallelism >= 1
    error_message = "fmi_establish_parallelism must be at least 1."
  }
}

variable "fmi_gather_topology" {
  description = "Arm B gatherv payload algorithm: linear (every rank sends to rank 0) or binomial (tree, rank 0 receives O(log N) messages). binomial needs an executor image whose cylon has the binomial gatherv"
  type        = string
  default     = "linear"

  validation {
    condition     = contains(["linear", "binomial"], var.fmi_gather_topology)
    error_message = "fmi_gather_topology must be linear or binomial."
  }
}

variable "arm_b_sync_mode" {
  description = "Arm B rank flow: barrier (connect, barrier, partition map broadcast, infer, barrier, gather) or early_exit (read the partition from the S3 payload as Arm A does, infer, then connect and gather, no barriers)"
  type        = string
  default     = "barrier"

  validation {
    condition     = contains(["barrier", "early_exit"], var.arm_b_sync_mode)
    error_message = "arm_b_sync_mode must be barrier or early_exit."
  }
}

variable "fmi_finalize_linger_s" {
  description = "Seconds each Arm B socket close waits for the peer's TCP acknowledgement of all sent data (SO_LINGER); 0 closes immediately. Positive values need an executor image whose cylon supports FMI_FINALIZE_LINGER_S"
  type        = number
  default     = 0

  validation {
    condition     = var.fmi_finalize_linger_s >= 0
    error_message = "fmi_finalize_linger_s must not be negative."
  }
}

variable "cylon_log_level" {
  description = "CYLON_LOG_LEVEL for the Arm B ranks (glog: 0 INFO, 1 WARNING, 2 ERROR); pycylon disables glog when unset"
  type        = number
  default     = 1
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