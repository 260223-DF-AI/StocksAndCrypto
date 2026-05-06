#!/usr/bin/env bash
# =============================================================================
# ResearchFlow — Deployment Script
# =============================================================================
# Builds and deploys the Lambda function using AWS SAM.
#
# Prerequisites:
#   - AWS CLI configured with valid credentials
#   - AWS SAM CLI installed (pip install aws-sam-cli)
#   - Docker installed (for sam build)
#
# Usage:
#   cd deployment && bash deploy.sh
# =============================================================================

#!/usr/bin/env bash
set -euo pipefail

# Read sensitive params from .env or environment.
: "${PINECONE_API_KEY:?PINECONE_API_KEY env var must be set}"

echo "Building SAM application (Docker image)..."
sam build --template-file template.yaml

echo "Deploying to AWS..."
sam deploy \
  --guided \
  --stack-name researchflow \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      PineconeApiKey="${PINECONE_API_KEY}" \
      PineconeIndexName="${PINECONE_INDEX_NAME:-researchflow}" \
      BedrockModelId="${BEDROCK_MODEL_ID:-anthropic.claude-opus-4-5-20251101-v1:0}" \
      EmbeddingModelId="${EMBEDDING_MODEL_ID:-amazon.titan-embed-text-v2:0}"

echo "Deployment complete. Check the Outputs above for your API endpoint."