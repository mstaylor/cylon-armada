# ---------------------------------------------------------------------------
# ECR
# ---------------------------------------------------------------------------

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = data.aws_ecr_repository.main.repository_url
}

output "gpu_image_uri" {
  description = "Full URI of the GPU ECS image"
  value       = "${data.aws_ecr_repository.main.repository_url}:${var.gpu_image_tag}"
}

# ---------------------------------------------------------------------------
# ECS
# ---------------------------------------------------------------------------

output "gpu_task_definition_arn" {
  description = "ARN of the GPU ECS task definition"
  value       = aws_ecs_task_definition.gpu_armada.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the GPU ECS task IAM role"
  value       = aws_iam_role.ecs_task.arn
}

output "ecs_execution_role_arn" {
  description = "ARN of the GPU ECS task execution IAM role"
  value       = aws_iam_role.ecs_execution.arn
}

output "ecs_log_group" {
  description = "CloudWatch log group for GPU ECS tasks"
  value       = aws_cloudwatch_log_group.ecs_gpu.name
}

# ---------------------------------------------------------------------------
# Step Functions
# ---------------------------------------------------------------------------

output "gpu_workflow_arn" {
  description = "ARN of the ECS EC2 GPU Step Functions state machine"
  value       = aws_sfn_state_machine.ecs_ec2_gpu_workflow.arn
}

output "gpu_workflow_name" {
  description = "Name of the ECS EC2 GPU Step Functions state machine"
  value       = aws_sfn_state_machine.ecs_ec2_gpu_workflow.name
}