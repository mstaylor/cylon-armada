locals {
  deploy_ecr_endpoints = var.enable_ecr_vpc_endpoints && var.vpc_id != "" && length(var.ecs_task_subnet_ids) > 0
}

data "aws_vpc" "tasks" {
  count = local.deploy_ecr_endpoints ? 1 : 0
  id    = var.vpc_id
}

data "aws_route_table" "tasks_main" {
  count  = local.deploy_ecr_endpoints ? 1 : 0
  vpc_id = var.vpc_id

  filter {
    name   = "association.main"
    values = ["true"]
  }
}

resource "aws_security_group" "ecr_endpoints" {
  count = local.deploy_ecr_endpoints ? 1 : 0

  name        = "${var.project_name}-ecr-endpoints"
  description = "HTTPS from the VPC to the ECR interface endpoints"
  vpc_id      = var.vpc_id

  ingress {
    description = "Image pulls from tasks in the VPC"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.tasks[0].cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_vpc_endpoint" "ecr" {
  for_each = local.deploy_ecr_endpoints ? toset(["ecr.api", "ecr.dkr"]) : toset([])

  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.aws_region}.${each.key}"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = var.ecs_task_subnet_ids
  security_group_ids  = [aws_security_group.ecr_endpoints[0].id]
  private_dns_enabled = true

  tags = merge(local.common_tags, { Name = "${var.project_name}-${each.key}" })
}

resource "aws_vpc_endpoint" "s3_gateway" {
  count = local.deploy_ecr_endpoints ? 1 : 0

  vpc_id            = var.vpc_id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [data.aws_route_table.tasks_main[0].id]

  tags = merge(local.common_tags, { Name = "${var.project_name}-s3" })
}