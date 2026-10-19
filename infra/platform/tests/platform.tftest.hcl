# `terraform test` (or `tofu test`) — plans with mocked providers, no cloud calls.
mock_provider "google" {
  mock_data "google_project" {
    defaults = { number = "123456789012" }
  }
  mock_resource "google_sql_database_instance" {
    defaults = { connection_name = "demo-project:us-central1:slack-agent-pg" }
  }
  # Mocks invent random strings; service-account fields are validated, so give real shapes.
  mock_resource "google_service_account" {
    defaults = {
      name  = "projects/demo-project/serviceAccounts/mock-runtime@demo-project.iam.gserviceaccount.com"
      email = "mock-runtime@demo-project.iam.gserviceaccount.com"
    }
  }
}
mock_provider "random" {}

variables {
  project_id        = "demo-project"
  github_repository = "anjijava16/slack-fastapi-langgraph-agent"
  state_bucket      = "demo-project-tfstate"
}

run "two_environments" {
  command = apply # apply against mocks, so computed values (connection_name) resolve

  assert {
    condition     = keys(google_sql_database.env) == ["prod", "test"]
    error_message = "one database per environment"
  }
  assert {
    condition     = google_sql_database_instance.pg.settings[0].edition == "ENTERPRISE"
    error_message = "shared-core tiers need the ENTERPRISE edition"
  }
  assert {
    condition     = length(google_secret_manager_secret.user) == 12
    error_message = "6 user-filled secrets x 2 environments"
  }
  assert {
    condition     = contains([for s in google_secret_manager_secret.user : s.secret_id], "slack-agent-test-qdrant-api-key")
    error_message = "Qdrant API key secret per environment"
  }
  assert {
    condition     = length(google_secret_manager_secret_iam_member.deployer_reads_api_key) == 2
    error_message = "deployer may read each env's API key (scheduled ingestion)"
  }
  assert {
    condition     = contains([for s in google_secret_manager_secret.user : s.secret_id], "slack-agent-prod-slack-signing-secret")
    error_message = "secret ids are <service>-<env>-<kind>"
  }
  assert {
    condition     = startswith(google_secret_manager_secret_version.db_url["test"].secret_data, "postgresql://slack_agent_test:")
    error_message = "DSN user"
  }
  assert {
    condition     = endswith(google_secret_manager_secret_version.db_url["test"].secret_data, "@/slack_agent_test?host=/cloudsql/demo-project:us-central1:slack-agent-pg")
    error_message = "DSN must use the Cloud Run unix socket"
  }
  assert {
    condition     = google_iam_workload_identity_pool_provider.github.attribute_condition == "assertion.repository == \"anjijava16/slack-fastapi-langgraph-agent\" && assertion.ref == \"refs/heads/develop\""
    error_message = "WIF must trust only this repo's develop branch"
  }
  assert {
    condition     = google_service_account_iam_member.wif_binding.member == "principalSet://iam.googleapis.com/projects/123456789012/locations/global/workloadIdentityPools/github-pool/attribute.repository/anjijava16/slack-fastapi-langgraph-agent"
    error_message = "WIF binding member"
  }
}

run "test_only" {
  command = plan
  variables {
    environments = ["test"]
  }
  assert {
    condition     = keys(google_sql_database.env) == ["test"]
    error_message = "only the test database"
  }
}
