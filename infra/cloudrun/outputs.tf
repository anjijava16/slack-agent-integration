output "service_name" {
  value = google_cloud_run_v2_service.app.name
}

output "service_url" {
  value = google_cloud_run_v2_service.app.uri
}

output "slack_request_url" {
  description = "Paste into the Slack app: Event Subscriptions, Interactivity and each slash command."
  value       = "${google_cloud_run_v2_service.app.uri}/slack/events"
}

output "latest_revision" {
  value = google_cloud_run_v2_service.app.latest_ready_revision
}

output "image" {
  value = var.image
}
