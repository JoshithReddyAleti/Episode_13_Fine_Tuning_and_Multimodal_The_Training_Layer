variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "adapter_bucket_name" {
  description = "S3 bucket name for LoRA adapter storage"
  type        = string
}

variable "training_bucket_name" {
  description = "S3 bucket name for training datasets"
  type        = string
}
