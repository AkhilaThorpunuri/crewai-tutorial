# CrewAI Multi-Agent MCP Question-Answering

Multi-agent CrewAI pipeline with 5 specialized agents and dedicated MCP tool calls connected to the Docker containers from `tinitiate-ai-setup`:

1. **PostgreSQL Database Analyst** (`localhost:5432`) -> `postgres_list_tables`, `postgres_describe_table`, `postgres_run_query`
2. **Website HTML Portal Specialist** (`http://localhost:8080`) -> `website_list_pages`, `website_search_pages`, `website_fetch_page`
3. **Website Policy PDF Specialist** (`http://localhost:8080`) -> `pdf_list_documents`, `pdf_search_documents`, `pdf_read_document`
4. **MinIO Object Storage & CSV Analyst** (`http://localhost:9002`) -> `minio_list_objects`, `minio_query_csv`, `minio_read_object`
5. **Elasticsearch Document Search Specialist** (`http://localhost:9200`) -> `elastic_index_stats`, `elastic_search_documents`, then `minio_read_object`

## Prerequisites

Start the Docker containers in your `tinitiate-ai-setup` repository:
```powershell
docker compose -f tinitiate-ai-onboarding.yml up -d
```

## Setup & Run

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # Optional: add OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY

# Run built-in demo questions:
python ask_crewai.py

# Ask a custom question:
python ask_crewai.py -q "How many orders exist per status in PostgreSQL?"
python ask_crewai.py -q "What is OceanX Logistics policy on demurrage and detention?"

# Interactive prompt:
python ask_crewai.py -i
```

## MLflow Tracing

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open `http://localhost:5000` to inspect runs and nested execution traces.
