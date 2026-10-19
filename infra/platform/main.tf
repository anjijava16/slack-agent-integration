locals {
  apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "serviceusage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "cloudtrace.googleapis.com",
  ]

  envs = toset(var.environments)

  # Per-environment secrets whose VALUES you add with gcloud (scripts/gcp-setup.sh
  # reads them from ~/.zprofile.sh), so they never sit in Terraform state.
  #   slack-bot-token       xoxb-...   (OAuth & Permissions)
  #   slack-signing-secret  32 hex     (Basic Information -> App Credentials)
  #   slack-app-token       xapp-...   (only used when SLACK_MODE=socket)
  #   api-key               protects /api/* (X-API-Key header)
  #   qdrant-api-key        Qdrant Cloud API key (course FAQ RAG)
  #   cohere-api-key        optional, only with rerank_provider = "cohere"
  user_secret_kinds = ["slack-bot-token", "slack-signing-secret", "slack-app-token", "api-key", "qdrant-api-key", "cohere-api-key"]
  user_secrets = {
    for pair in setproduct(var.environments, local.user_secret_kinds) :
    "${pair[0]}/${pair[1]}" => { env = pair[0], kind = pair[1] }
  }

  runtime_sa_id  = "${var.service_name}-run"
  deployer_sa_id = "${var.service_name}-deployer"
}

data "google_project" "this" {
  project_id = var.project_id
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ================================================================ Artifact Registry
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = var.artifact_repo_id
  description   = "Images for ${var.service_name}"
  format        = "DOCKER"

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 15
    }
  }
  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      older_than = "2592000s" # 30 days
    }
  }

  depends_on = [google_project_service.apis]
}

# ================================================================ Cloud SQL (PostgreSQL)
# One instance, one database + user per environment. Cloud Run connects through
# its built-in Cloud SQL connector (a unix socket under /cloudsql), which is
# authorised by IAM (roles/cloudsql.client) — the public IP has no authorised
# networks, so nothing else can reach it.
resource "google_sql_database_instance" "pg" {
  name                = "${var.service_name}-pg"
  database_version    = var.db_version
  region              = var.region
  deletion_protection = var.db_deletion_protection

  settings {
    # PostgreSQL 16 defaults to Enterprise Plus, which has no shared-core tiers.
    edition           = "ENTERPRISE"
    tier              = var.db_tier
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_size         = 10
    disk_autoresize   = true

    deletion_protection_enabled = var.db_deletion_protection

    ip_configuration {
      ipv4_enabled = true # reached only via the Cloud SQL connector; no authorised networks
      ssl_mode     = "ENCRYPTED_ONLY"
    }

    backup_configuration {
      enabled                        = true
      start_time                     = "07:00" # UTC
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
      }
    }

    maintenance_window {
      day  = 7 # Sunday
      hour = 8 # UTC
    }

    insights_config {
      query_insights_enabled = true
    }

    database_flags {
      name  = "max_connections"
      value = "50"
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "env" {
  for_each = local.envs
  name     = "slack_agent_${each.key}"
  instance = google_sql_database_instance.pg.name
}

resource "random_password" "db" {
  for_each = local.envs
  length   = 32
  special  = false # keeps the DSN URL-safe without escaping
}

resource "google_sql_user" "env" {
  for_each = local.envs
  name     = "slack_agent_${each.key}"
  instance = google_sql_database_instance.pg.name
  password = random_password.db[each.key].result
  # Postgres won't drop a user that owns tables; leave it behind on destroy.
  deletion_policy = "ABANDON"
}

# The full DSN for each environment, in the shape both drivers understand
# (asyncpg for votes/runs, psycopg for LangGraph memory). host= points at the
# unix socket Cloud Run mounts at /cloudsql/<connection name>.
#
# Note: Terraform generated this password, so it is in the state file (private,
# versioned GCS bucket). Rotate with: terraform apply -replace='random_password.db["prod"]'
resource "google_secret_manager_secret" "db_url" {
  for_each  = local.envs
  secret_id = "${var.service_name}-${each.key}-db-url"
  labels    = { app = var.service_name, env = each.key }
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "db_url" {
  for_each    = local.envs
  secret      = google_secret_manager_secret.db_url[each.key].id
  secret_data = "postgresql://${google_sql_user.env[each.key].name}:${random_password.db[each.key].result}@/${google_sql_database.env[each.key].name}?host=/cloudsql/${google_sql_database_instance.pg.connection_name}"
}

# ================================================================ App secrets (containers only)
resource "google_secret_manager_secret" "user" {
  for_each  = local.user_secrets
  secret_id = "${var.service_name}-${each.value.env}-${each.value.kind}"
  labels    = { app = var.service_name, env = each.value.env }
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

# ================================================================ Runtime identity (what the containers run as)
resource "google_service_account" "runtime" {
  account_id   = local.runtime_sa_id
  display_name = "Cloud Run runtime for ${var.service_name}"
}

resource "google_project_iam_member" "runtime_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent",
    "roles/cloudsql.client", # connect through the Cloud SQL connector
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "runtime_db_url" {
  for_each  = google_secret_manager_secret.db_url
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "runtime_user_secrets" {
  for_each  = google_secret_manager_secret.user
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# The OpenRouter secret is shared with your other apps, so it is referenced, not owned.
resource "google_secret_manager_secret_iam_member" "runtime_openrouter" {
  secret_id = "projects/${var.project_id}/secrets/${var.openrouter_secret_id}"
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# ================================================================ Deployer identity (what GitHub Actions becomes)
resource "google_service_account" "deployer" {
  account_id   = local.deployer_sa_id
  display_name = "GitHub Actions deployer for ${var.service_name}"
}

resource "google_project_iam_member" "deployer_roles" {
  for_each = toset([
    "roles/run.admin",            # create/update Cloud Run services and their IAM (public invoker)
    "roles/secretmanager.viewer", # read secret metadata during terraform plan
    "roles/cloudsql.viewer",      # resolve the Cloud SQL instance attached to the service
    "roles/serviceusage.serviceUsageConsumer",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

# Push images to THIS repository only.
resource "google_artifact_registry_repository_iam_member" "deployer_push" {
  location   = google_artifact_registry_repository.docker.location
  repository = google_artifact_registry_repository.docker.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

# The scheduled ingestion workflow (.github/workflows/ingest.yml) calls the deployed
# service's /api/ingest endpoints, so it may read each environment's API key — and nothing else.
resource "google_secret_manager_secret_iam_member" "deployer_reads_api_key" {
  for_each  = local.envs
  secret_id = google_secret_manager_secret.user["${each.key}/api-key"].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.deployer.email}"
}

# Attach the runtime SA to the services it deploys.
resource "google_service_account_iam_member" "deployer_actas_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

# Read/write the cloudrun stack's Terraform state.
resource "google_storage_bucket_iam_member" "deployer_state" {
  bucket = var.state_bucket
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.deployer.email}"
}

# ================================================================ Workload Identity Federation (keyless GitHub -> GCP)
# A provider of our own inside the shared pool, locked to this repo + branch.
# Your other repos keep their own providers in the same pool untouched.
resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = var.wif_pool_id
  workload_identity_pool_provider_id = "${var.service_name}-gh"
  display_name                       = "GitHub ${var.service_name}"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
    "attribute.actor"      = "assertion.actor"
  }

  # Hard gate: tokens from any other repo or branch are rejected before IAM is checked.
  attribute_condition = "assertion.repository == \"${var.github_repository}\" && assertion.ref == \"refs/heads/${var.github_branch}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  depends_on = [google_project_service.apis]
}

resource "google_service_account_iam_member" "wif_binding" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/projects/${data.google_project.this.number}/locations/global/workloadIdentityPools/${var.wif_pool_id}/attribute.repository/${var.github_repository}"
}
