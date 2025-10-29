# Building on GCP with Private Credentials and Pushing to Docker Hub

## 1. GCP Service Account Setup
- Create a new service account in your GCP project.
- Grant it the following roles:
  - Cloud Build Service Account
  - Secret Manager Secret Accessor
  - Storage Admin (if using GCR or Artifact Registry)
- Download the service account key (JSON) and configure your local gcloud to use it:
  ```
  gcloud auth activate-service-account --key-file=YOUR_KEY.json
  gcloud config set project YOUR_PROJECT_ID
  ```

## 2. Store Docker Hub Credentials in GCP Secret Manager
- Add your Docker Hub username and password as secrets:
  ```
  echo -n 'your-docker-username' | gcloud secrets create docker-username --data-file=-
  echo -n 'your-docker-password' | gcloud secrets create docker-password --data-file=-
  ```
- Or update existing secrets:
  ```
  echo -n 'your-docker-username' | gcloud secrets versions add docker-username --data-file=-
  echo -n 'your-docker-password' | gcloud secrets versions add docker-password --data-file=-
  ```

## 3. Update cloudbuild.yaml
- In `wan_animate/cloudbuild.yaml`, set:
  - `serviceAccount` to your new service account email.
  - `availableSecrets` to reference your secrets (docker-username, docker-password).
  - Ensure the Docker Hub login step uses these secrets.

## 4. Trigger the Build
- Run the build from your local machine:
  ```
  gcloud builds submit wan_animate --config wan_animate/cloudbuild.yaml --substitutions=_TAG_NAME=wan-animate-v1
  ```
  - Replace `_TAG_NAME` with your desired tag.

## 5. Verify
- Check GCP Cloud Build logs for success.
- Verify the image is pushed to your Docker Hub repository.

## Security Notes
- Never commit your credentials or service account key to source control.
- Use GCP Secret Manager for all sensitive data.
- Restrict service account permissions to the minimum required.
