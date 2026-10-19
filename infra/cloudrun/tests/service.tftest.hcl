# `terraform test` (or `tofu test`) — plans with a mocked Google provider, no cloud calls.
mock_provider "google" {}

variables {
  project_id               = "demo-project"
  region                   = "us-central1"
  env                      = "test"
  image                    = "us-central1-docker.pkg.dev/demo-project/slack-agent/slack-agent:abc123"
  runtime_service_account  = "slack-agent-run@demo-project.iam.gserviceaccount.com"
  cloudsql_connection_name = "demo-project:us-central1:slack-agent-pg"
  slack_channel_id         = "C0123456789"
}

run "http_mode_defaults" {
  command = plan

  assert {
    condition     = google_cloud_run_v2_service.app.name == "slack-agent-test"
    error_message = "service should be named <service>-<env>"
  }
  assert {
    condition     = google_cloud_run_v2_service.app.template[0].containers[0].resources[0].cpu_idle == false
    error_message = "CPU must stay allocated so Slack work after the ack isn't throttled"
  }
  assert {
    condition     = google_cloud_run_v2_service.app.template[0].volumes[0].cloud_sql_instance[0].instances == toset(["demo-project:us-central1:slack-agent-pg"])
    error_message = "Cloud SQL socket must be mounted"
  }
  assert {
    condition = toset([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.name]) == toset([
      "APP_ENV", "SLACK_MODE", "SLACK_CHANNEL_ID", "LLM_MODEL", "OPENAI_BASE_URL", "LOG_LEVEL", "DB_POOL_MAX", "APP_NAME",
      "SLACK_CMD_AGENT", "SLACK_CMD_ADD", "RAG_ENABLED",
      "DATABASE_URL", "LLM_API_KEY", "SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "API_KEY",
    ])
    error_message = "unexpected env vars (http mode must not inject SLACK_APP_TOKEN)"
  }
  assert {
    condition     = one([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.value_source[0].secret_key_ref[0].secret if e.name == "DATABASE_URL"]) == "slack-agent-test-db-url"
    error_message = "DATABASE_URL must come from the env's db-url secret"
  }
  assert {
    condition     = one([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.value if e.name == "SLACK_CMD_AGENT"]) == "/agent-test"
    error_message = "test env gets suffixed slash commands"
  }
  assert {
    condition     = length(google_cloud_run_v2_service_iam_member.public) == 1
    error_message = "Slack needs a public endpoint"
  }
}

run "rag_on_qdrant_cloud" {
  command = plan
  variables {
    rag_enabled     = true
    qdrant_url      = "https://xyz.us-east4-0.gcp.cloud.qdrant.io:6333"
    rerank_provider = "cohere"
  }
  assert {
    condition     = one([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.value if e.name == "QDRANT_COLLECTION_PREFIX"]) == "test_faq_"
    error_message = "collections are prefixed per environment"
  }
  assert {
    condition     = one([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.value_source[0].secret_key_ref[0].secret if e.name == "QDRANT_API_KEY"]) == "slack-agent-test-qdrant-api-key"
    error_message = "Qdrant key comes from Secret Manager"
  }
  assert {
    condition     = contains([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.name], "COHERE_API_KEY")
    error_message = "cohere reranker gets its key"
  }
  assert {
    condition     = one([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.value if e.name == "SLACK_CMD_FAQ"]) == "/faq-test"
    error_message = "test env gets /faq-test"
  }
}

run "rag_needs_a_url" {
  command = plan
  variables {
    rag_enabled = true
  }
  expect_failures = [google_cloud_run_v2_service.app]
}

run "prod_socket_mode_without_api_key" {
  command = plan
  variables {
    env             = "prod"
    slack_mode      = "socket"
    min_instances   = 1
    require_api_key = false
  }
  assert {
    condition     = contains([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.name], "SLACK_APP_TOKEN")
    error_message = "socket mode needs SLACK_APP_TOKEN"
  }
  assert {
    condition     = one([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.value if e.name == "SLACK_CMD_AGENT"]) == "/agent"
    error_message = "prod keeps the plain /agent command"
  }
  assert {
    condition     = !contains([for e in google_cloud_run_v2_service.app.template[0].containers[0].env : e.name], "API_KEY")
    error_message = "API_KEY should be absent when require_api_key = false"
  }
}

run "socket_mode_cannot_scale_to_zero" {
  command = plan
  variables {
    slack_mode    = "socket"
    min_instances = 0
  }
  expect_failures = [google_cloud_run_v2_service.app]
}

run "rejects_unknown_env" {
  command = plan
  variables {
    env = "staging"
  }
  expect_failures = [var.env]
}
