#!/bin/bash
set -e

echo "Setting up DVC with MinIO..."

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    git add .dvc .dvcignore
    git commit -m "Initialize DVC"
fi

# Configure MinIO as remote storage
echo "Configuring MinIO remote..."
dvc remote add -d minio s3://dvc-storage -f
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin

# Commit DVC config
git add .dvc/config
git commit -m "Configure DVC remote storage (MinIO)" || true

# Configure autostage
dvc config core.autostage true
git add .dvc/config
git commit -m "Enable DVC autostage" || true

# Track data with DVC
echo "Tracking data files with DVC..."
if [ -f "data/raw/churn.csv" ]; then
    dvc add data/raw/churn.csv
    git add data/raw/churn.csv.dvc data/raw/.gitignore
    git commit -m "Track raw data with DVC" || true
else
    echo "Warning: data/raw/churn.csv not found. Please add your data file first."
fi

# Create .gitignore entries for DVC-tracked directories
echo "Setting up .gitignore for DVC-tracked directories..."

cat > data/.gitignore << EOF
/raw
/processed
/reports
EOF

cat > models/.gitignore << EOF
/model.pkl
/feature_engineer.pkl
/plots
EOF

git add data/.gitignore models/.gitignore
git commit -m "Add .gitignore for DVC-tracked directories" || true

# Push to remote (will create bucket if doesn't exist)
echo "Pushing data to MinIO..."
dvc push || echo "Push failed - ensure MinIO is running (docker-compose up -d)"

echo ""
echo "DVC setup complete!"
echo ""
echo "Next steps:"
echo "  1. Ensure MinIO is running: docker-compose up -d"
echo "  2. Run the pipeline: dvc repro"
echo "  3. View metrics: dvc metrics show"
echo "  4. View plots: dvc plots show"
echo ""
