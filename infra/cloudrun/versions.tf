# Applied by CD (GitHub Actions) once per environment, with its own state:
#   terraform init -backend-config="bucket=<project>-tfstate" -backend-config="prefix=slack-agent/cloudrun/<env>"
terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.0"
    }
  }

  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}
