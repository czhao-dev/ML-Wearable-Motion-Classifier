data "google_project" "this" {
  project_id = var.project_id
}

locals {
  apis = [
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
  ]

  # Default Compute Engine service account -- the Cloud Run runtime identity
  # used here, matching the previous manual `gcloud run deploy` setup (no
  # dedicated service account was created).
  runtime_service_account = "${data.google_project.this.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "rag_pdf_chatbot" {
  project       = var.project_id
  location      = var.region
  repository_id = var.service_name
  format        = "DOCKER"
  description   = "Container images for the gen-ai-rag-chatbot Cloud Run service."

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service" "rag_pdf_chatbot" {
  project  = var.project_id
  name     = var.service_name
  location = var.region

  template {
    containers {
      image = var.image_tag

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_LOCATION"
        value = var.region
      }
      env {
        name  = "VERTEX_LLM_MODEL_ID"
        value = var.vertex_llm_model_id
      }
      env {
        name  = "VERTEX_EMBEDDING_MODEL_ID"
        value = var.vertex_embedding_model_id
      }
    }

    timeout = "300s"

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    service_account = local.runtime_service_account
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.rag_pdf_chatbot,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.rag_pdf_chatbot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${local.runtime_service_account}"
}
