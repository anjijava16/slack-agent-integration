variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "env" {
  description = "test or prod. Picks the service name, database and secrets."
  type        = string

  validation {
    condition     = contains(["test", "prod"], var.env)
    error_message = "env must be \"test\" or \"prod\"."
  }
}

variable "service_name" {
  description = "Base name from infra/platform. The Cloud Run service is <service_name>-<env>."
  type        = string
  default     = "slack-agent"
}

variable "image" {
  description = "Full image reference, set by CD (<region>-docker.pkg.dev/<project>/<repo>/<service>:<sha>)."
  type        = string
}

variable "runtime_service_account" {
  description = "Email of the runtime SA from infra/platform."
  type        = string
}

variable "cloudsql_connection_name" {
  description = "<project>:<region>:<instance> from infra/platform."
  type        = string
}

# ---------------------------------------------------------------- app settings
variable "llm_model" {
  description = "OpenRouter model id with tool calling, e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4.5, google/gemini-2.5-flash."
  type        = string
  default     = "openai/gpt-4o-mini"
}

variable "llm_base_url" {
  type    = string
  default = "https://openrouter.ai/api/v1"
}

variable "openrouter_secret_id" {
  type    = string
  default = "openrouter-api-key"
}

variable "slack_mode" {
  description = "http (recommended on Cloud Run) or socket (needs min_instances >= 1)."
  type        = string
  default     = "http"

  validation {
    condition     = contains(["http", "socket"], var.slack_mode)
    error_message = "slack_mode must be \"http\" or \"socket\"."
  }
}

variable "slack_channel_id" {
  description = "Default channel for POST /api/slack/notify. Not a secret."
  type        = string
  default     = ""
}

variable "require_api_key" {
  description = "Require X-API-Key on /api/* (value in secret <service>-<env>-api-key)."
  type        = bool
  default     = true
}

variable "log_level" {
  type    = string
  default = "INFO"
}

# ---------------------------------------------------------------- capacity
variable "min_instances" {
  description = "0 = scale to zero (HTTP mode). Socket Mode needs >= 1 to keep the WebSocket open."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Keep (max_instances x 2 x db_pool_max) x environments under Cloud SQL max_connections (50)."
  type        = number
  default     = 3
}

variable "concurrency" {
  description = "Concurrent requests per instance. Slack events are I/O bound."
  type        = number
  default     = 40
}

variable "db_pool_max" {
  description = "Max connections per pool per instance (there are two pools: votes/runs and LangGraph memory)."
  type        = number
  default     = 3
}

variable "cpu" {
  type    = string
  default = "1"
}

variable "memory" {
  description = "2Gi leaves room for the embedding model + cross-encoder (~300 MB) and ingestion batches."
  type        = string
  default     = "2Gi"
}

# ---------------------------------------------------------------- RAG (course FAQ bot on Qdrant Cloud)
variable "rag_enabled" {
  description = "Turn on the course FAQ RAG. Needs qdrant_url and the <service>-<env>-qdrant-api-key secret."
  type        = bool
  default     = false
}

variable "qdrant_url" {
  description = "Qdrant Cloud cluster URL, e.g. https://xyz.us-east4-0.gcp.cloud.qdrant.io:6333"
  type        = string
  default     = ""
}

variable "rerank_provider" {
  description = "fastembed (local cross-encoder baked into the image), cohere (needs the cohere-api-key secret) or none."
  type        = string
  default     = "fastembed"

  validation {
    condition     = contains(["fastembed", "cohere", "none"], var.rerank_provider)
    error_message = "rerank_provider must be fastembed, cohere or none."
  }
}

variable "allow_unauthenticated" {
  description = "Slack must be able to POST to /slack/events, so the service is public; Slack requests are verified with the signing secret, /api/* with X-API-Key."
  type        = bool
  default     = true
}
