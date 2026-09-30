output "init_function_arn" {
  description = "ARN of the initializer Lambda"
  value       = aws_lambda_function.init.arn
}

output "executor_function_arn" {
  description = "ARN of the inference (Model Inference) Lambda"
  value       = aws_lambda_function.executor.arn
}

output "summarize_function_arn" {
  description = "ARN of the summarizer Lambda"
  value       = aws_lambda_function.summarize.arn
}

output "state_machine_arn" {
  description = "ARN of the replicated Cosmic AI state machine"
  value       = aws_sfn_state_machine.workflow.arn
}

output "data_bucket_arn" {
  description = "ARN of the existing data/results bucket this module reads and writes"
  value       = data.aws_s3_bucket.data.arn
}