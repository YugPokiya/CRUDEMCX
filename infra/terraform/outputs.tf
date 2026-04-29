output "backend_alb_url" {
  value = "http://${aws_lb.api.dns_name}"
}

output "frontend_cloudfront_domain" {
  value = aws_cloudfront_distribution.frontend.domain_name
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.address
}
