output "account_id" {
  description = "Account this deployment targets"
  value       = data.aws_caller_identity.current.account_id
}

output "executor_image_uri" {
  description = "Executor image, by digest"
  value       = local.executor_image_uri
}

output "state_machine_arn" {
  description = "Arm A (S3) workflow"
  value       = aws_sfn_state_machine.workflow.arn
}

output "fmi_state_machine_arn" {
  description = "Arm B (Cylon FMI) workflow"
  value       = aws_sfn_state_machine.fmi_workflow.arn
}

output "executor_function_arn" {
  description = "Shared executor Lambda"
  value       = aws_lambda_function.executor.arn
}

output "data_bucket" {
  description = "Data bucket the campaign driver passes as --bucket and --data-bucket"
  value       = data.aws_s3_bucket.data.id
}