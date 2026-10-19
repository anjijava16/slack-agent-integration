variable "project_id" {
  description = "GCP project to deploy into."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Artifact Registry and Cloud SQL."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Base name. Cloud Run services become <service_name>-<env>."
  type        = string
  default     = "slack-agent"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,20}$", var.service_name))
    error_message = "service_name: 3-21 chars, lowercase letters, digits and hyphens, starting with a letter."
  }
}

variable "environments" {
  description = "Cloud environments. Each gets its own Cloud Run service, database, DB user and secrets."
  type        = list(string)
  default     = ["test", "prod"]

  validation {
    condition     = alltrue([for e in var.environments : contains(["test", "prod"], e)])
    error_message = "environments may contain \"test\" and/or \"prod\" (local runs on SQLite and needs nothing here)."
  }
}

variable "github_repository" {
  description = "GitHub repo allowed to deploy, owner/name, exactly as on GitHub."
  type        = string
}

variable "github_branch" {
  description = "Only workflows running on this branch can impersonate the deployer."
  type        = string
  default     = "develop"
}

variable "state_bucket" {
  description = "Existing Terraform state bucket (created by gcp-setup.sh; shared with other stacks)."
  type        = string
}

variable "wif_pool_id" {
  description = "Existing Workload Identity pool id (gcp-setup.sh creates it if missing; shared with other repos)."
  type        = string
  default     = "github-pool"
}

variable "openrouter_secret_id" {
  description = "Existing Secret Manager secret holding the OpenRouter key (shared with other apps in the project)."
  type        = string
  default     = "openrouter-api-key"
}

variable "artifact_repo_id" {
  type    = string
  default = "slack-agent"
}

# ---------------------------------------------------------------- Cloud SQL
variable "db_tier" {
  description = "db-f1-micro is the cheapest shared-core tier (fine for a Slack bot). db-g1-small or db-custom-1-3840 for more headroom."
  type        = string
  default     = "db-f1-micro"
}

variable "db_version" {
  type    = string
  default = "POSTGRES_16"
}

variable "db_deletion_protection" {
  description = "Protects the instance from `terraform destroy` AND from deletion in the console."
  type        = bool
  default     = true
}
