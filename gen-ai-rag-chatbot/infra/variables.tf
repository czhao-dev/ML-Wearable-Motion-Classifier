variable "project_id" {
  description = "GCP project ID hosting the deployment."
  type        = string
}

variable "region" {
  description = "GCP region for Artifact Registry and Cloud Run."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name shared by the Cloud Run service and Artifact Registry repository."
  type        = string
  default     = "rag-pdf-chatbot"
}

variable "image_tag" {
  description = "Fully-qualified image reference to deploy, e.g. \"us-central1-docker.pkg.dev/PROJECT_ID/rag-pdf-chatbot/rag-pdf-chatbot:latest\". Built and pushed separately (docker buildx build --push) -- Terraform does not own the image build."
  type        = string
}

variable "vertex_llm_model_id" {
  description = "Vertex AI Gemini model ID used for answer generation."
  type        = string
  default     = "gemini-2.5-flash"
}

variable "vertex_embedding_model_id" {
  description = "Vertex AI embedding model ID used for the RAG retriever."
  type        = string
  default     = "text-embedding-004"
}
