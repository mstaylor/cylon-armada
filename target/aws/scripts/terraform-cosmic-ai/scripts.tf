locals {
  scripts = {
    inference = {
      source = "${path.module}/lambda_src/inference.py"
      key    = "${var.script_prefix}/inference.py"
    }
    inference_fmi = {
      source = "${path.module}/lambda_src/inference_FMI.py"
      key    = "${var.script_prefix}/inference_FMI.py"
    }
    communicator_package = {
      source = "${path.module}/../../../shared/scripts/communicator/__init__.py"
      key    = "${var.script_prefix}/communicator/__init__.py"
    }
    fmi_bridge = {
      source = "${path.module}/../../../shared/scripts/communicator/fmi_bridge.py"
      key    = "${var.script_prefix}/communicator/fmi_bridge.py"
    }
  }
}

resource "aws_s3_object" "scripts" {
  for_each = local.scripts

  bucket = data.aws_s3_bucket.data.id
  key    = each.value.key
  source = each.value.source
  etag   = try(filemd5(each.value.source), null)

  lifecycle {
    precondition {
      condition     = try(fileexists(each.value.source), false)
      error_message = "Script source ${each.value.source} does not exist."
    }
  }

  tags       = local.common_tags
  depends_on = [terraform_data.account_guard]
}