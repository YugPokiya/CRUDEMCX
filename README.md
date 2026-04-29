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
2. `cd backend && DATABASE_URL=postgres://<user>:<pass>@<host>:5432/mcx_trade DB_ACQUIRE_TIMEOUT_SECS=8 cargo run`
3. `cd frontend && npm install && npm run dev`
4. `python ml/pipeline.py --csv data/raw_ohlcv.csv --data-dir data --epochs 20`
5. `curl http://localhost:8080/health/db` to verify database connectivity separately from API boot.

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

## Advanced PostgreSQL Medallion Pipeline (Bronze/Silver/Gold)
Additional production-style scripts are available under `ml/medallion/`:

1. Install dependencies:
   ```bash
   pip install -r ml/medallion/requirements.txt
   ```
2. Create medallion schemas/tables:
   ```bash
   psql -U postgres -d crudemcx -f db/schema_medallion.sql
   ```
3. Run Bronze ingest (raw -> bronze.ohlcv_raw):
   ```bash
   DB_PASSWORD=<your_password> python ml/medallion/bronze.py
   ```
4. Run Silver cleaning/repair (bronze -> silver.ohlcv_clean):
   ```bash
   DB_PASSWORD=<your_password> python ml/medallion/silver.py
   ```
5. Run Gold feature generation (silver -> gold.ml_features):
   ```bash
   DB_PASSWORD=<your_password> python ml/medallion/gold_layer.py
   ```


See `docs/Software_Nededd.pdf` for software requirements document.
