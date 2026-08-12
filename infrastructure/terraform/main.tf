terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" { default = "us-west-2" }
variable "cluster_name" { default = "ep13-training" }

# EKS cluster with GPU nodes for training
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"

  vpc_id     = var.vpc_id
  subnet_ids = var.subnet_ids

  eks_managed_node_groups = {
    gpu_nodes = {
      min_size     = 0
      max_size     = 8
      desired_size = 2
      instance_types = ["p4d.24xlarge"]  # 8x A100 40GB
      capacity_type  = "ON_DEMAND"
      labels = {
        gpu = "true"
        episode = "13"
      }
      taints = [
        { key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }
      ]
    }
  }
}

variable "vpc_id" { type = string }
variable "subnet_ids" { type = list(string) }
