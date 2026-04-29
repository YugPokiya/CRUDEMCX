# AWS deployment (Terraform)

This Terraform stack deploys the full MCX platform:
- VPC + subnets + security groups
- RDS PostgreSQL
- ECS Fargate backend service behind ALB
- Scheduled ECS Fargate ML task (hourly via EventBridge)
- S3 + CloudFront for frontend hosting

## Prerequisites
- Terraform >= 1.5
- AWS CLI configured (`aws configure`)
- Backend + ML images pushed to ECR
- Frontend `dist/` built (`npm run build`) and uploaded to S3 bucket output by Terraform

## Steps
1. Copy variables file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```
2. Edit `terraform.tfvars` values.
3. Deploy:
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```
4. Apply DB schema to RDS endpoint output:
   ```bash
   psql "postgres://<user>:<pass>@<rds-endpoint>:5432/mcx_trade" -f ../../db/schema.sql
   ```
5. Upload frontend build to S3:
   ```bash
   aws s3 sync ../../frontend/dist s3://<frontend_bucket_name> --delete
   ```
