# ---------------------------------------------------------------------------
# ECR
# ---------------------------------------------------------------------------

output "ecr_repository_url" {
  description = "ECR repository URL (existing, referenced by Terraform)"
  value       = data.aws_ecr_repository.main.repository_url
}

# ---------------------------------------------------------------------------
# Lambda ARNs — Python
# ---------------------------------------------------------------------------

output "python_init_arn" {
  description = "ARN of the Python armada_init Lambda"
  value       = aws_lambda_function.python_init.arn
}

output "python_executor_arn" {
  description = "ARN of the Python armada_executor Lambda"
  value       = aws_lambda_function.python_executor.arn
}

output "python_aggregate_arn" {
  description = "ARN of the Python armada_aggregate Lambda"
  value       = aws_lambda_function.python_aggregate.arn
}

# ---------------------------------------------------------------------------
# Lambda ARNs — Node.js
# ---------------------------------------------------------------------------

output "nodejs_init_arn" {
  description = "ARN of the Node.js armada_init Lambda"
  value       = aws_lambda_function.nodejs_init.arn
}

output "nodejs_executor_arn" {
  description = "ARN of the Node.js armada_executor Lambda"
  value       = aws_lambda_function.nodejs_executor.arn
}

output "nodejs_aggregate_arn" {
  description = "ARN of the Node.js armada_aggregate Lambda"
  value       = aws_lambda_function.nodejs_aggregate.arn
}

# ---------------------------------------------------------------------------
# ECS
# ---------------------------------------------------------------------------

output "ecs_task_definition_arn" {
  description = "ARN of the cylon-armada Python ECS task definition"
  value       = aws_ecs_task_definition.python_armada.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task IAM role"
  value       = aws_iam_role.ecs_task.arn
}

output "ecs_execution_role_arn" {
  description = "ARN of the ECS task execution IAM role"
  value       = aws_iam_role.ecs_execution.arn
}

output "ecs_log_group" {
  description = "CloudWatch log group name for ECS tasks"
  value       = aws_cloudwatch_log_group.ecs_python.name
}

# ---------------------------------------------------------------------------
# Step Functions — Lambda workflows
# ---------------------------------------------------------------------------

output "python_workflow_arn" {
  description = "ARN of the Python Lambda Step Functions state machine"
  value       = aws_sfn_state_machine.python_workflow.arn
}

output "nodejs_workflow_arn" {
  description = "ARN of the Node.js Lambda Step Functions state machine"
  value       = aws_sfn_state_machine.nodejs_workflow.arn
}

output "model_parallel_workflow_arn" {
  description = "ARN of the model parallel Step Functions state machine"
  value       = aws_sfn_state_machine.model_parallel_workflow.arn
}

# ---------------------------------------------------------------------------
# Step Functions — ECS workflows
# ---------------------------------------------------------------------------

output "ecs_fargate_workflow_arn" {
  description = "ARN of the ECS Fargate Step Functions state machine"
  value       = aws_sfn_state_machine.ecs_fargate_workflow.arn
}

output "ecs_ec2_workflow_arn" {
  description = "ARN of the ECS EC2 Step Functions state machine"
  value       = aws_sfn_state_machine.ecs_ec2_workflow.arn
}

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

output "dynamodb_table_name" {
  description = "Name of the DynamoDB context store table"
  value       = aws_dynamodb_table.context_store.name
}

output "dynamodb_table_arn" {
  description = "ARN of the DynamoDB context store table"
  value       = aws_dynamodb_table.context_store.arn
}

output "scripts_bucket" {
  description = "S3 bucket for Lambda/ECS hot-reload scripts"
  value       = data.aws_s3_bucket.scripts.bucket
}

output "results_bucket" {
  description = "S3 bucket for experiment results (existing)"
  value       = data.aws_s3_bucket.results.bucket
}

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

output "redis_endpoint" {
  description = "Redis endpoint (ElastiCache if created, otherwise var.redis_host)"
  value       = local.redis_endpoint
}

output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution IAM role"
  value       = aws_iam_role.lambda_execution.arn
}

# ---------------------------------------------------------------------------
# Rendezvous test Lambda
# ---------------------------------------------------------------------------

output "rendezvous_test_arn" {
  description = "ARN of the rendezvous test Lambda"
  value       = aws_lambda_function.rendezvous_test.arn
}

output "rendezvous_host" {
  description = "Rendezvous server host configured for FMI direct channel"
  value       = var.rendezvous_host
}

output "rendezvous_port" {
  description = "Rendezvous server port"
  value       = var.rendezvous_port
}

# ---------------------------------------------------------------------------
# Redis ECS service (when create_ecs_redis = true)
# ---------------------------------------------------------------------------

output "redis_service_dns" {
  description = "Cloud Map DNS name for the ECS Redis service (VPC-internal)"
  value       = var.create_ecs_redis ? "redis.${var.project_name}.local" : null
}

output "redis_image_uri" {
  description = "Container image used for the Redis ECS service"
  value       = var.create_ecs_redis ? var.redis_image_uri : null
}

output "redis_hostname" {
  description = "Route 53 hostname for the Redis ECS service (auto-updated on task replacement)"
  value       = var.redis_hostname != "" ? var.redis_hostname : null
}

output "redis_dns_updater_arn" {
  description = "ARN of the Lambda that updates Route 53 when Redis restarts"
  value       = local.deploy_redis_dns ? aws_lambda_function.redis_dns_updater[0].arn : null
}