locals {
  service       = "${var.service_name}-${var.env}"
  secret_prefix = "${var.service_name}-${var.env}"
  # Slash commands are global per workspace: test gets /agent-test, prod gets /agent.
  # scripts/slack-manifest.sh writes the same names into the Slack app.
  cmd_suffix = var.env == "prod" ? "" : "-${var.env}"

  # name in the container -> Secret Manager secret id. Always injected.
  secret_env = {
    DATABASE_URL         = "${local.secret_prefix}-db-url"
    LLM_API_KEY          = var.openrouter_secret_id
    SLACK_BOT_TOKEN      = "${local.secret_prefix}-slack-bot-token"
    SLACK_SIGNING_SECRET = "${local.secret_prefix}-slack-signing-secret"
  }
  # Only when they're needed (an empty secret would fail the deploy).
  optional_secret_env = merge(
    var.slack_mode == "socket" ? { SLACK_APP_TOKEN = "${local.secret_prefix}-slack-app-token" } : {},
    var.require_api_key ? { API_KEY = "${local.secret_prefix}-api-key" } : {},
    var.rag_enabled ? { QDRANT_API_KEY = "${local.secret_prefix}-qdrant-api-key" } : {},
    var.rag_enabled && var.rerank_provider == "cohere" ? { COHERE_API_KEY = "${local.secret_prefix}-cohere-api-key" } : {},
  )

  # One Qdrant Cloud cluster can serve every environment: collections are prefixed
  # per env (test_faq_llm, prod_faq_llm, ...).
  rag_env = var.rag_enabled ? {
    RAG_ENABLED              = "true"
    QDRANT_MODE              = "cloud"
    QDRANT_URL               = var.qdrant_url
    QDRANT_COLLECTION_PREFIX = "${var.env}_faq_"
    RERANK_PROVIDER          = var.rerank_provider
    SLACK_CMD_FAQ            = "/faq${local.cmd_suffix}"
  } : { RAG_ENABLED = "false" }

  plain_env = {
    APP_ENV          = var.env
    SLACK_MODE       = var.slack_mode
    SLACK_CHANNEL_ID = var.slack_channel_id
    LLM_MODEL        = var.llm_model
    OPENAI_BASE_URL  = var.llm_base_url
    LOG_LEVEL        = var.log_level
    DB_POOL_MAX      = tostring(var.db_pool_max)
    APP_NAME         = local.service
    SLACK_CMD_AGENT  = "/agent${local.cmd_suffix}"
    SLACK_CMD_ADD    = "/agent-add${local.cmd_suffix}"
  }
}

resource "google_cloud_run_v2_service" "app" {
  name                = local.service
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  labels = {
    app = var.service_name
    env = var.env
  }

  template {
    service_account                  = var.runtime_service_account
    timeout                          = "300s"
    max_instance_request_concurrency = var.concurrency

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    labels = {
      app = var.service_name
      env = var.env
    }

    # Cloud Run's built-in Cloud SQL connector: a unix socket at
    # /cloudsql/<project>:<region>:<instance>, authorised by the runtime SA's
    # roles/cloudsql.client. DATABASE_URL's ?host= points here.
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [var.cloudsql_connection_name]
      }
    }

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
        # IMPORTANT for Slack: Bolt acks the HTTP request first and runs the
        # agent afterwards. With CPU only allocated during requests (cpu_idle =
        # true), that after-the-response work is throttled to almost nothing and
        # answers arrive minutes late or never. Keep CPU allocated.
        cpu_idle          = false
        startup_cpu_boost = true
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      dynamic "env" {
        for_each = merge(local.plain_env, local.rag_env)
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = merge(local.secret_env, local.optional_secret_env)
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 2
        period_seconds        = 3
        timeout_seconds       = 2
        failure_threshold     = 20
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds    = 30
        timeout_seconds   = 3
        failure_threshold = 3
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  lifecycle {
    precondition {
      condition     = !var.rag_enabled || var.qdrant_url != ""
      error_message = "rag_enabled needs qdrant_url (set the QDRANT_URL repository variable)."
    }
    precondition {
      condition     = var.slack_mode == "http" || var.min_instances >= 1
      error_message = "SLACK_MODE=socket keeps a WebSocket open, so it needs min_instances >= 1 (scale-to-zero would disconnect Slack). Use slack_mode = \"http\" or raise min_instances."
    }
  }
}

# Slack calls /slack/events from the internet without Google credentials.
# Requests are still authenticated: Bolt checks every X-Slack-Signature with the
# signing secret, and /api/* requires X-API-Key.
resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_unauthenticated ? 1 : 0
  name     = google_cloud_run_v2_service.app.name
  location = google_cloud_run_v2_service.app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
