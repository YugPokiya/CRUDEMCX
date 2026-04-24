# CRUDEMCX Smart Trading Stack

This repository includes:
- Rust backend API (`backend/`)
- React frontend dashboard (`frontend/`)
- TensorFlow RNN/LSTM pipeline (`ml/`)
- PostgreSQL schema (`db/`)
- Design docs and requirements (`docs/`)
- Cloud deployment assets (`infra/terraform/`, Dockerfiles)

## Data Pipeline (Bronze → Silver → Gold)
The ML pipeline now implements explicit medallion layers:
- **Bronze**: raw OHLCV import as-is (`data/bronze/ohlcv_bronze.csv`)
- **Silver**: cleaned/typed/deduped OHLCV (`data/silver/ohlcv_silver.csv`)
- **Gold**: feature-engineered dataset for model training (`data/gold/ohlcv_gold_features.csv`)

## Quick start
1. `psql mcx_trade -f db/schema.sql`
2. `cd backend && cargo run`
3. `cd frontend && npm install && npm run dev`
4. `python ml/pipeline.py --csv data/raw_ohlcv.csv --data-dir data --epochs 20`

## Deploy full stack on AWS
1. Build and push container images:
   - `backend/Dockerfile`
   - `ml/Dockerfile`
2. Build frontend static files: `cd frontend && npm install && npm run build`
3. Configure Terraform variables from `infra/terraform/terraform.tfvars.example`.
4. Deploy infra: `cd infra/terraform && terraform init && terraform apply`
5. Apply DB schema to RDS endpoint output from Terraform.
6. Upload `frontend/dist` to S3 bucket output by Terraform.

See `infra/terraform/README.md` for full deployment instructions.

See `docs/Software_Nededd.pdf` for software requirements document.
