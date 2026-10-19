# scripts/gcp-setup.sh copies these into .gcp-setup.env; github-setup.sh turns them
# into GitHub repository variables.
output "GCP_PROJECT_ID" {
  value = var.project_id
}

output "GCP_REGION" {
  value = var.region
}

output "GCP_WIF_PROVIDER" {
  value = google_iam_workload_identity_pool_provider.github.name
}

output "GCP_DEPLOYER_SA" {
  value = google_service_account.deployer.email
}

output "GCP_RUNTIME_SA" {
  value = google_service_account.runtime.email
}

output "GAR_REPOSITORY" {
  value = google_artifact_registry_repository.docker.repository_id
}

output "CLOUDSQL_CONNECTION_NAME" {
  value = google_sql_database_instance.pg.connection_name
}

output "SERVICE_NAME" {
  value = var.service_name
}

output "secrets_to_fill" {
  description = "Secrets whose values you add with gcloud (gcp-setup.sh does it from ~/.zprofile.sh)."
  value       = sort([for s in google_secret_manager_secret.user : s.secret_id])
}

output "databases" {
  value = { for e, db in google_sql_database.env : e => db.name }
}
