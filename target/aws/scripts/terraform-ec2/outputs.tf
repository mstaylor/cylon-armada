output "cpu_task_definition_arn" {
  description = "ARN of the CPU ECS task definition"
  value       = aws_ecs_task_definition.cpu_armada.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the CPU ECS task IAM role"
  value       = aws_iam_role.ecs_task.arn
}

output "ecs_execution_role_arn" {
  description = "ARN of the CPU ECS task execution IAM role"
  value       = aws_iam_role.ecs_execution.arn
}

output "ec2_instance_profile_arn" {
  description = "ARN of the EC2 instance profile (attached to ASG instances)"
  value       = aws_iam_instance_profile.ec2_instance.arn
}

output "asg_name" {
  description = "Name of the EC2 Auto Scaling Group"
  value       = aws_autoscaling_group.ecs_cpu.name
}

output "capacity_provider_name" {
  description = "Name of the ECS capacity provider"
  value       = aws_ecs_capacity_provider.cpu.name
}

output "launch_template_id" {
  description = "ID of the EC2 launch template"
  value       = aws_launch_template.ecs_cpu.id
}

output "ecs_log_group" {
  description = "CloudWatch log group for CPU ECS tasks"
  value       = aws_cloudwatch_log_group.ecs_cpu.name
}

output "ec2_workflow_arn" {
  description = "ARN of the ECS EC2 CPU Step Functions state machine"
  value       = aws_sfn_state_machine.ecs_ec2_workflow.arn
}

output "ec2_workflow_name" {
  description = "Name of the ECS EC2 CPU Step Functions state machine"
  value       = aws_sfn_state_machine.ecs_ec2_workflow.name
}