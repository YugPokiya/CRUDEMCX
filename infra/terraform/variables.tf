variable "aws_region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

variable "project" {
  type        = string
  description = "Project/service prefix"
  default     = "mcx-trade"
}

variable "environment" {
  type        = string
  description = "Environment name"
  default     = "prod"
}

variable "database_name" {
  type    = string
  default = "mcx_trade"
}

variable "database_username" {
  type    = string
  default = "mcx_app"
}

variable "database_password" {
  type      = string
  sensitive = true
}

variable "backend_image" {
  type        = string
  description = "Backend ECR image URI with tag"
}

variable "ml_image" {
  type        = string
  description = "ML ECR image URI with tag"
}

variable "frontend_bucket_name" {
  type        = string
  description = "Unique S3 bucket name for frontend"
}
