output "python_lambda_arn" {
  description = "ARN of the Python Lambda function"
  value       = aws_lambda_function.python_worker.arn
}

output "nodejs_lambda_arn" {
  description = "ARN of the Node.js Lambda function"
  value       = aws_lambda_function.nodejs_worker.arn
}

output "python_workflow_arn" {
  description = "ARN of the Python Step Functions state machine"
  value       = aws_sfn_state_machine.python_workflow.arn
}

output "nodejs_workflow_arn" {
  description = "ARN of the Node.js Step Functions state machine"
  value       = aws_sfn_state_machine.nodejs_workflow.arn
}

output "model_parallel_workflow_arn" {
  description = "ARN of the model parallel Step Functions state machine"
  value       = aws_sfn_state_machine.model_parallel_workflow.arn
}

output "dynamodb_table_name" {
  description = "Name of the DynamoDB context store table"
  value       = aws_dynamodb_table.context_store.name
}

output "dynamodb_table_arn" {
  description = "ARN of the DynamoDB context store table"
  value       = aws_dynamodb_table.context_store.arn
}

output "scripts_bucket" {
  description = "S3 bucket for Lambda scripts"
  value       = aws_s3_bucket.scripts.bucket
}

output "ecr_python_url" {
  description = "ECR repository URL for Python image"
  value       = aws_ecr_repository.python.repository_url
}

output "ecr_nodejs_url" {
  description = "ECR repository URL for Node.js image"
  value       = aws_ecr_repository.nodejs.repository_url
}

output "redis_endpoint" {
  description = "Redis endpoint (if ElastiCache was created)"
  value       = var.create_elasticache ? aws_elasticache_cluster.redis[0].cache_nodes[0].address : var.redis_host
}

output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution IAM role"
  value       = aws_iam_role.lambda_execution.arn
}