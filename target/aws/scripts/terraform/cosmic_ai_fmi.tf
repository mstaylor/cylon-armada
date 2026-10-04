# ---------------------------------------------------------------------------
# Cosmic AI Track 1, Arm B (FMI direct): broadcast and gather over FMI.
#
# Shares the Arm A executor Lambda and the Arm A ASL file, so orchestration
# (Initialize, Distributed Map, Summarize) is identical between the arms and
# only the init/summarize functions and the executed script differ.
# ---------------------------------------------------------------------------

locals {
  cosmic_ai_fmi_init_function_name      = "${var.project_name}-cosmic-ai-fmi-init"
  cosmic_ai_fmi_summarize_function_name = "${var.project_name}-cosmic-ai-fmi-summarize"
  cosmic_ai_fmi_workflow_name           = "${var.project_name}-cosmic-ai-fmi-workflow"

  cosmic_ai_fmi_asl_vars = merge(local.cosmic_ai_asl_vars, {
    INIT_FUNCTION_NAME      = local.cosmic_ai_fmi_init_function_name
    SUMMARIZE_FUNCTION_NAME = local.cosmic_ai_fmi_summarize_function_name
  })

  cosmic_ai_fmi_scripts = {
    inference_fmi = {
      source = "${local.cosmic_ai_lambda_src_dir}/inference_FMI.py"
      key    = var.cosmic_ai_fmi_inference_key
    }
    communicator_package = {
      source = "${path.module}/../../../shared/scripts/communicator/__init__.py"
      key    = "${var.cosmic_ai_fmi_communicator_prefix}/__init__.py"
    }
    fmi_bridge = {
      source = "${path.module}/../../../shared/scripts/communicator/fmi_bridge.py"
      key    = "${var.cosmic_ai_fmi_communicator_prefix}/fmi_bridge.py"
    }
  }
}

data "archive_file" "cosmic_ai_fmi_init_zip" {
  type        = "zip"
  output_path = "${path.module}/.build/cosmic_ai_fmi_initializer.zip"

  source {
    content  = file("${local.cosmic_ai_lambda_src_dir}/initializer_FMI.py")
    filename = "initializer_FMI.py"
  }

  source {
    content  = file("${local.cosmic_ai_lambda_src_dir}/initializer.py")
    filename = "initializer.py"
  }
}

data "archive_file" "cosmic_ai_fmi_summarize_zip" {
  type        = "zip"
  source_file = "${local.cosmic_ai_lambda_src_dir}/summarizer_FMI.py"
  output_path = "${path.module}/.build/cosmic_ai_fmi_summarizer.zip"
}

resource "aws_lambda_function" "cosmic_ai_fmi_init" {
  function_name    = local.cosmic_ai_fmi_init_function_name
  role             = aws_iam_role.lambda_execution.arn
  package_type     = "Zip"
  filename         = data.archive_file.cosmic_ai_fmi_init_zip.output_path
  source_code_hash = data.archive_file.cosmic_ai_fmi_init_zip.output_base64sha256
  handler          = "initializer_FMI.lambda_handler"
  runtime          = var.cosmic_ai_python_runtime
  memory_size      = var.cosmic_ai_init_memory_mb
  timeout          = var.cosmic_ai_init_timeout

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_function" "cosmic_ai_fmi_summarize" {
  function_name    = local.cosmic_ai_fmi_summarize_function_name
  role             = aws_iam_role.lambda_execution.arn
  package_type     = "Zip"
  filename         = data.archive_file.cosmic_ai_fmi_summarize_zip.output_path
  source_code_hash = data.archive_file.cosmic_ai_fmi_summarize_zip.output_base64sha256
  handler          = "summarizer_FMI.lambda_handler"
  runtime          = var.cosmic_ai_python_runtime
  memory_size      = var.cosmic_ai_summarize_memory_mb
  timeout          = var.cosmic_ai_init_timeout

  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }

  tags = local.common_tags
}

resource "aws_sfn_state_machine" "cosmic_ai_fmi_workflow" {
  name     = local.cosmic_ai_fmi_workflow_name
  role_arn = aws_iam_role.step_functions_execution.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/../step_functions/workflow_cosmic_ai.asl.json", local.cosmic_ai_fmi_asl_vars)

  tags = local.common_tags
}

resource "aws_s3_object" "cosmic_ai_fmi_scripts" {
  for_each = local.cosmic_ai_fmi_scripts

  bucket = data.aws_s3_bucket.cosmic_ai_lambda_data.id
  key    = each.value.key
  source = each.value.source
  etag   = try(filemd5(each.value.source), null)

  lifecycle {
    precondition {
      condition     = try(fileexists(each.value.source), false)
      error_message = "Arm B script source ${each.value.source} does not exist."
    }
  }

  tags = local.common_tags
}
