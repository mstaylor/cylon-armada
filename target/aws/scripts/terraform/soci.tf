locals {
  deploy_soci = var.enable_soci_indexing
}

data "archive_file" "soci_index_generator" {
  count = local.deploy_soci ? 1 : 0

  type        = "zip"
  source_file = "${path.module}/../lambda/soci/bootstrap"
  output_path = "${path.module}/../lambda/soci/bootstrap.zip"
}

resource "aws_iam_role" "soci_index_generator" {
  count = local.deploy_soci ? 1 : 0

  name = "${var.project_name}-soci-index-generator"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "soci_index_generator" {
  count = local.deploy_soci ? 1 : 0

  name = "${var.project_name}-soci-index-generator"
  role = aws_iam_role.soci_index_generator[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
        ]
        Resource = data.aws_ecr_repository.main.arn
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      },
    ]
  })
}

resource "aws_lambda_function" "soci_index_generator" {
  count = local.deploy_soci ? 1 : 0

  function_name    = "${var.project_name}-soci-index-generator"
  role             = aws_iam_role.soci_index_generator[0].arn
  runtime          = "provided.al2023"
  handler          = "main"
  architectures    = ["x86_64"]
  filename         = data.archive_file.soci_index_generator[0].output_path
  source_code_hash = data.archive_file.soci_index_generator[0].output_base64sha256
  memory_size      = 1024
  timeout          = 900

  ephemeral_storage {
    size = 10240
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_event_rule" "soci_image_pushed" {
  count = local.deploy_soci ? 1 : 0

  name        = "${var.project_name}-soci-image-pushed"
  description = "Fires when the Fargate-pulled image is pushed to ECR, so it can be SOCI indexed"

  event_pattern = jsonencode({
    source        = ["aws.ecr"]
    "detail-type" = ["ECR Image Action"]
    detail = {
      "action-type"     = ["PUSH"]
      result            = ["SUCCESS"]
      "repository-name" = [var.ecr_repository_name]
      "image-tag"       = [var.soci_indexed_tag]
    }
  })

  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "soci_index_generator" {
  count = local.deploy_soci ? 1 : 0

  rule      = aws_cloudwatch_event_rule.soci_image_pushed[0].name
  target_id = "soci-index-generator"
  arn       = aws_lambda_function.soci_index_generator[0].arn
}

resource "aws_lambda_permission" "soci_index_generator" {
  count = local.deploy_soci ? 1 : 0

  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.soci_index_generator[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.soci_image_pushed[0].arn
}