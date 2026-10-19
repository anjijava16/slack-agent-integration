# Platform stack: everything that exists once per project and that CD must not
# be able to change (database, identities, secrets, CI trust). Applied by a
# human with Owner on the project, through scripts/gcp-setup.sh.
terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6"
    }
  }

  # Partial config, completed by gcp-setup.sh:
  #   terraform init -backend-config="bucket=<project>-tfstate"
  backend "gcs" {
    prefix = "slack-agent/platform"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
