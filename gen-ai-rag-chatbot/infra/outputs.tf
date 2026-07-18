output "service_url" {
  description = "Public URL of the deployed Cloud Run service."
  value       = google_cloud_run_v2_service.rag_pdf_chatbot.uri
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository path for pushing images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.rag_pdf_chatbot.repository_id}"
}
