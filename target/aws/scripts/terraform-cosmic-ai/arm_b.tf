locals {
  fmi_init_function_name      = "${var.project_name}-fmi-init"
  fmi_summarize_function_name = "${var.project_name}-fmi-summarize"
  fmi_workflow_name           = "${var.project_name}-fmi-workflow"

  fmi_asl_vars = merge(local.asl_vars, {
    INIT_FUNCTION_NAME      = local.fmi_init_function_name
    SUMMARIZE_FUNCTION_NAME = local.fmi_summarize_function_name
    EXECUTOR_RETRY_ERRORS   = jsonencode(var.fmi_executor_retry_errors)
  })
}

data "archive_file" "fmi_init_zip" {
  type        = "zip"
  output_path = "${path.module}/.build/initializer_fmi.zip"

  source {
    content  = file("${path.module}/lambda_src/initializer_FMI.py")
    filename = "initializer_FMI.py"
  }

  source {
    content  = file("${path.module}/lambda_src/initializer.py")
    filename = "initializer.py"
  }
}

data "archive_file" "fmi_summarize_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/summarizer_FMI.py"
  output_path = "${path.module}/.build/summarizer_fmi.zip"
}

resource "aws_lambda_function" "fmi_init" {
  function_name    = local.fmi_init_function_name
  role             = aws_iam_role.lambda_execution.arn
  package_type     = "Zip"
  filename         = data.archive_file.fmi_init_zip.output_path
  source_code_hash = data.archive_file.fmi_init_zip.output_base64sha256
  handler          = "initializer_FMI.lambda_handler"
  runtime          = var.python_runtime
  memory_size      = var.init_memory_mb
  timeout          = var.init_timeout

  environment {
    variables = {
      FMI_CHANNEL_TYPE = var.fmi_channel_type
      FMI_OPTIONS      = var.fmi_options
      FMI_MAX_TIMEOUT  = tostring(var.fmi_max_timeout)
      RENDEZVOUS_HOST  = var.rendezvous_host
      RENDEZVOUS_PORT  = tostring(var.rendezvous_port)
    }
  }

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags       = local.common_tags
  depends_on = [terraform_data.account_guard]
}

resource "aws_lambda_function" "fmi_summarize" {
  function_name    = local.fmi_summarize_function_name
  role             = aws_iam_role.lambda_execution.arn
  package_type     = "Zip"
  filename         = data.archive_file.fmi_summarize_zip.output_path
  source_code_hash = data.archive_file.fmi_summarize_zip.output_base64sha256
  handler          = "summarizer_FMI.lambda_handler"
  runtime          = var.python_runtime
  memory_size      = var.summarize_memory_mb
  timeout          = var.init_timeout

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags       = local.common_tags
  depends_on = [terraform_data.account_guard]
}

resource "aws_sfn_state_machine" "fmi_workflow" {
  name     = local.fmi_workflow_name
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/../step_functions/workflow_cosmic_ai.asl.json", local.fmi_asl_vars)

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.sfn_workflow.arn}:*"
    include_execution_data = false
    level                  = var.sfn_log_level
  }

  tags       = local.common_tags
  depends_on = [aws_iam_role_policy.step_functions_policy, aws_cloudwatch_log_resource_policy.sfn_workflow]
}