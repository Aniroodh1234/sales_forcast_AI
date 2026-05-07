#!/bin/bash
# Script to sync training artifacts from S3 to EC2
# Usage: ./s3_sync.sh s3://your-bucket-name/salescast/artifacts/

S3_BUCKET=$1

if [ -z "$S3_BUCKET" ]; then
    echo "Usage: ./s3_sync.sh s3://your-bucket-name/salescast/artifacts/"
    exit 1
fi

echo "Installing AWS CLI if not present..."
sudo apt install awscli -y || true

echo "Syncing artifacts from $S3_BUCKET to local artifacts/ directory..."
aws s3 sync "$S3_BUCKET" ../artifacts/ --exclude "*" \
    --include "preprocessor/*" \
    --include "models/*" \
    --include "metrics/*" \
    --include "data/*"

echo "Sync complete!"
