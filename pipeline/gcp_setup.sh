#!/usr/bin/env bash
# Setup GCP sekali saja untuk pipeline continuous training, jalankan di Cloud Shell
# Hasilnya: service account untuk GitHub Actions + Workload Identity Federation (login tanpa key JSON)
set -euo pipefail

PROJECT_ID="project-f9262c16-d41b-407d-9e6"
REPO="lostmys3lf/final-project-building-a-machine-learning-system"
SA_NAME="github-mlops"

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
POOL_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github"

gcloud services enable run.googleapis.com iamcredentials.googleapis.com sts.googleapis.com compute.googleapis.com

gcloud iam service-accounts create "$SA_NAME" --display-name="GitHub Actions MLOps pipeline"

for ROLE in roles/compute.instanceAdmin.v1 roles/run.admin; do           # Start/stop VM runner dan deploy Cloud Run
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${SA_EMAIL}" --role="$ROLE" --condition=None
done
gcloud iam service-accounts add-iam-policy-binding "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/iam.serviceAccountUser"   # Cloud Run berjalan sebagai default compute service account

gcloud iam workload-identity-pools create github --location=global --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc github-oidc --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${REPO}'"                # Hanya repo ini yang boleh login

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${REPO}"

echo ""
echo "Isi dua repository variable di GitHub (Settings > Secrets and variables > Actions > Variables):"
echo "GCP_WIF_PROVIDER=${POOL_ID}/providers/github-oidc"
echo "GCP_SERVICE_ACCOUNT=${SA_EMAIL}"
