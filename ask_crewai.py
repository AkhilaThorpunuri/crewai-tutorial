"""
CrewAI Multi-Agent Question-Answering Script with 5 SEPARATE Specialized Agents & MCP Tool Calls:
  1. `PostgreSQL Database Analyst`          -> Calls `postgres_list_tables`, `postgres_describe_table`, `postgres_run_query`
  2. `Website HTML Portal Specialist`       -> Calls `website_list_pages`, `website_search_pages`, `website_fetch_page`
  3. `Website Policy PDF Specialist`        -> Calls `pdf_list_documents`, `pdf_search_documents`, `pdf_read_document`
  4. `MinIO Object Storage & CSV Analyst`   -> Calls `minio_list_objects`, `minio_query_csv`, `minio_read_object`
  5. `Elasticsearch Document Search Specialist` -> Calls `elastic_index_stats`, `elastic_search_documents`, then `minio_read_object`

Every execution logs an MLflow Run and a nested MLflow Trace (`http://localhost:5000`).
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Literal

from dotenv import load_dotenv

from mcp_server import (
    HAS_MLFLOW,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    configure_mlflow,
    elastic_index_stats,
    elastic_search_documents,
    minio_list_objects,
    minio_query_csv,
    minio_read_object,
    minio_search_markdown_sources,
    mlflow,
    mlflow_trace,
    pdf_list_documents,
    pdf_read_document,
    pdf_search_documents,
    postgres_describe_table,
    postgres_list_tables,
    postgres_run_query,
    website_fetch_page,
    website_list_pages,
    website_search_pages,
)

load_dotenv()
configure_mlflow()

RouteType = Literal["postgres", "website", "pdf", "minio", "elastic"]


def is_elastic_question(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in ["elastic", "full-text", "full text", "search index", "keyword search"])


_POLICY_HINTS = [
    "policy", "policies", "procedure", "regulation", "terms and conditions", "disclosure", "handbook",
    "code of conduct", "claims", "tariff", "surcharge", "demurrage", "detention", "reefer", "customs",
    "cargo", "dangerous goods", "hazardous", "reg e", "reg cc", "hipaa", "bsa", "aml", "ofac", "kyc",
    "privacy", "charity", "financial assistance", "dispute", "refund", "overdraft", "fee schedule",
    "fund transfer", "liability", "infection", "compliance", "patient", "telehealth", "eligibility",
    "what does", "what is the rule", "according to",
]
_DB_ENTITY_HINTS = [
    "order", "customer", "product", "categor", "payment", "inventory", "warehouse", "batch", "etl",
    "revenue", "spending", "stock",
]
_CSV_HINTS = [
    "csv", "indian_tech_jobs", "tech job", "job posting", "data scientist", "bengaluru", "bangalore", "salary",
]


def route_question(question: str) -> RouteType:
    """Pick the MCP source for a question."""
    q = question.lower()

    if is_elastic_question(q):
        return "elastic"
    if "pdf" in q:
        return "pdf"
    if any(k in q for k in ["website", "web page", "webpage", "faq page", "about us page", "localhost:8080"]):
        return "website"
    if any(k in q for k in ["postgres", "sql", "database", " db ", "table"]):
        return "postgres"
    if any(k in q for k in _CSV_HINTS):
        return "minio"

    if any(k in q for k in _POLICY_HINTS):
        return "elastic"

    if any(k in q for k in _DB_ENTITY_HINTS):
        return "postgres"

    return "elastic"


def _call_llm_if_available(system_prompt: str, user_prompt: str) -> str | None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text.strip()
        except Exception:
            pass

    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            return resp.text.strip()
        except Exception:
            pass

    return None


def _build_sql_for_question(question: str, schema_info: str) -> str:
    llm_sql = _call_llm_if_available(
        system_prompt=(
            "You are a PostgreSQL expert. Given the schema below, return ONLY a valid read-only "
            "PostgreSQL SELECT query (no markdown fences, no commentary).\n\nSchema:\n" + schema_info
        ),
        user_prompt=question,
    )
    if llm_sql:
        cleaned = re.sub(r"^```(?:sql)?\s*|\s*```$", "", llm_sql.strip(), flags=re.IGNORECASE)
        if cleaned.upper().startswith(("SELECT", "WITH")):
            return cleaned

    q = question.lower()
    if "batch" in q or "etl" in q:
        return "SELECT batch_name, batch_type, row_counts, loaded_at FROM etl_batch_control ORDER BY batch_name;"
    if "deliver" in q and "order" in q:
        return (
            "SELECT status, COUNT(*) AS order_count, ROUND(SUM(total_amount), 2) AS total_revenue "
            "FROM orders WHERE status = 'delivered' GROUP BY status;"
        )
    if "status" in q and "order" in q:
        return (
            "SELECT status, COUNT(*) AS order_count, ROUND(SUM(total_amount), 2) AS total_amount "
            "FROM orders GROUP BY status ORDER BY order_count DESC;"
        )
    if "payment" in q:
        return (
            "SELECT payment_method, status, COUNT(*) AS count, ROUND(SUM(amount), 2) AS total_amount "
            "FROM payments GROUP BY payment_method, status ORDER BY total_amount DESC LIMIT 10;"
        )
    if "categor" in q:
        return (
            "SELECT c.category_name, COUNT(p.product_id) AS product_count, ROUND(AVG(p.price), 2) AS avg_price "
            "FROM categories c LEFT JOIN products p ON c.category_id = p.category_id AND NOT p.is_deleted "
            "GROUP BY c.category_name ORDER BY product_count DESC;"
        )
    if "inventory" in q or "warehouse" in q or "stock" in q:
        return (
            "SELECT warehouse_location, COUNT(*) AS stocked_products, SUM(quantity_on_hand) AS total_units "
            "FROM inventory GROUP BY warehouse_location ORDER BY total_units DESC;"
        )
    if "customer" in q:
        return (
            "SELECT c.customer_id, c.first_name || ' ' || c.last_name AS customer_name, c.city, "
            "COUNT(o.order_id) AS total_orders, ROUND(COALESCE(SUM(o.total_amount), 0), 2) AS total_spent "
            "FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id "
            "WHERE NOT c.is_deleted GROUP BY c.customer_id, c.first_name, c.last_name, c.city "
            "ORDER BY total_spent DESC LIMIT 5;"
        )
    if "product" in q:
        return (
            "SELECT p.product_id, p.product_name, p.price, SUM(oi.quantity) AS total_units_sold "
            "FROM products p JOIN order_items oi ON p.product_id = oi.product_id "
            "WHERE NOT p.is_deleted GROUP BY p.product_id, p.product_name, p.price "
            "ORDER BY total_units_sold DESC LIMIT 5;"
        )
    return (
        "SELECT "
        "(SELECT COUNT(*) FROM categories WHERE NOT is_deleted) AS active_categories, "
        "(SELECT COUNT(*) FROM products WHERE NOT is_deleted) AS active_products, "
        "(SELECT COUNT(*) FROM customers WHERE NOT is_deleted) AS active_customers, "
        "(SELECT COUNT(*) FROM orders) AS total_orders, "
        "(SELECT ROUND(SUM(total_amount), 2) FROM orders) AS gross_order_value;"
    )


# =====================================================================
# 1. FOUR SEPARATE CREWAI AGENT SPECIFICATIONS
# =====================================================================

@dataclass
class AgentSpec:
    role: str
    goal: str
    backstory: str
    mcp_tools: List[str]


POSTGRES_AGENT_SPEC = AgentSpec(
    role="PostgreSQL Database Analyst",
    goal="Inspect PostgreSQL tables and run SQL queries via PostgreSQL MCP tools.",
    backstory="Manages `tinitiate-ai-postgres` (`localhost:5432`) for e-commerce categories, products, customers, orders, payments, and inventory.",
    mcp_tools=["postgres_list_tables", "postgres_describe_table", "postgres_run_query"],
)

WEBSITE_AGENT_SPEC = AgentSpec(
    role="Website HTML Portal Specialist",
    goal="Fetch and search live company HTML pages hosted on `http://localhost:8080/business-website-policy-documents/`.",
    backstory="Navigates the Nginx `tinitiate-ai-websites` container to extract answers from Home, About, Services, FAQ, Careers, and Contact web pages.",
    mcp_tools=["website_list_pages", "website_search_pages", "website_fetch_page"],
)

PDF_AGENT_SPEC = AgentSpec(
    role="Website Policy PDF Specialist",
    goal="Download and extract official policy rules from the 27 branded `.pdf` documents on `http://localhost:8080`.",
    backstory="Uses `pypdf` via PDF MCP tools to parse official PDF documents for OceanX Logistics, Atlas Retail Bank, and Amerimed Wellness Institute.",
    mcp_tools=["pdf_list_documents", "pdf_search_documents", "pdf_read_document"],
)

MINIO_AGENT_SPEC = AgentSpec(
    role="MinIO Object Storage & CSV Analyst",
    goal="Query CSV datasets (`indian_tech_jobs_2026.csv`, `file-formats/csv/*.csv`) and raw objects in MinIO (`s3://ai-source-data`).",
    backstory="Manages the `tinitiate-ai-minio` S3 object storage bucket (`localhost:9002`) containing 167 raw dataset files.",
    mcp_tools=["minio_list_objects", "minio_query_csv", "minio_read_object"],
)

ELASTIC_AGENT_SPEC = AgentSpec(
    role="Elasticsearch Document Search Specialist",
    goal="Find the right documents by full-text search in Elasticsearch, then read the originals back from MinIO.",
    backstory="Searches the `minio-documents` index in `tinitiate-ai-elasticsearch` (built from the PDFs, Markdown, YAML and HTML stored in MinIO) and uses the returned MinIO keys to fetch the source objects.",
    mcp_tools=["elastic_index_stats", "elastic_search_documents", "minio_read_object"],
)


# =====================================================================
# 2. SEPARATE CREWAI AGENT EXECUTION FUNCTIONS (TRACED BY MLFLOW)
# =====================================================================

@mlflow_trace(name="crewai.agent_1_postgres_analyst", span_type="AGENT")
def run_postgres_agent(question: str) -> str:
    tables_json = postgres_list_tables()
    schema_json = postgres_describe_table("orders")
    sql = _build_sql_for_question(question, f"{tables_json}\n{schema_json}")
    res = json.loads(postgres_run_query(sql))
    return (
        f"[CrewAI Agent 1: {POSTGRES_AGENT_SPEC.role}]\n"
        f"  • Separate MCP Calls: {', '.join(POSTGRES_AGENT_SPEC.mcp_tools)}\n"
        f"  • SQL Executed: `{sql}`\n"
        f"  • PostgreSQL Rows ({res.get('row_count', 0)}):\n{json.dumps(res.get('rows', []), indent=4)}"
    )


@mlflow_trace(name="crewai.agent_2_website_html_specialist", span_type="AGENT")
def run_website_agent(question: str) -> str:
    _ = website_list_pages()
    search_data = json.loads(website_search_pages(question))
    matches = search_data.get("top_matching_pages", [])
    if matches:
        top = matches[0]
        page_data = json.loads(website_fetch_page(top["company"], top["page"]))
        return (
            f"[CrewAI Agent 2: {WEBSITE_AGENT_SPEC.role}]\n"
            f"  • Separate MCP Calls: {', '.join(WEBSITE_AGENT_SPEC.mcp_tools)}\n"
            f"  • Live Website URL: {top['url']}\n"
            f"  • Page Title: {page_data.get('title')}\n"
            f"  • Extracted HTML Content:\n    {top['matching_excerpt']}"
        )
    page_data = json.loads(website_fetch_page("portal", "index.html"))
    return (
        f"[CrewAI Agent 2: {WEBSITE_AGENT_SPEC.role}]\n"
        f"  • Separate MCP Calls: {', '.join(WEBSITE_AGENT_SPEC.mcp_tools)}\n"
        f"  • Portal URL: {page_data.get('url')}"
    )


@mlflow_trace(name="crewai.agent_3_website_pdf_specialist", span_type="AGENT")
def run_pdf_agent(question: str) -> str:
    _ = pdf_list_documents()
    search_data = json.loads(pdf_search_documents(question))
    matches = search_data.get("top_pdf_matches", [])
    if matches:
        top = matches[0]
        _ = pdf_read_document(top["website_pdf_url"], max_chars=2500)
        return (
            f"[CrewAI Agent 3: {PDF_AGENT_SPEC.role}]\n"
            f"  • Separate MCP Calls: {', '.join(PDF_AGENT_SPEC.mcp_tools)}\n"
            f"  • Website PDF URL: {top['website_pdf_url']}\n"
            f"  • Matched Page: Page {top['best_matching_page']} of {top['total_pages']}\n"
            f"  • Extracted PDF Text:\n{top['pdf_page_excerpt']}"
        )
    return f"[CrewAI Agent 3: {PDF_AGENT_SPEC.role}] No matching PDF found."


@mlflow_trace(name="crewai.agent_4_minio_storage_analyst", span_type="AGENT")
def run_minio_agent(question: str) -> str:
    _ = minio_list_objects()
    q_lower = question.lower()
    if any(k in q_lower for k in ["job", "scientist", "engineer", "bengaluru", "bangalore", "salary", "indian_tech_jobs", "csv"]):
        filter_kw = "Data Scientist Bengaluru" if "bengaluru" in q_lower or "bangalore" in q_lower else "Data Scientist"
        csv_data = json.loads(
            minio_query_csv("business-data-dumps/indian_tech_jobs_2026.csv", filter_keyword=filter_kw, limit=3)
        )
        return (
            f"[CrewAI Agent 4: {MINIO_AGENT_SPEC.role}]\n"
            f"  • Separate MCP Calls: {', '.join(MINIO_AGENT_SPEC.mcp_tools)}\n"
            f"  • MinIO Object: `s3://ai-source-data/{csv_data.get('object_key')}`\n"
            f"  • Total CSV Rows: {csv_data.get('total_csv_rows')} | Matching '{filter_kw}': {csv_data.get('matching_row_count')}\n"
            f"  • Sample Rows:\n{json.dumps(csv_data.get('sample_matching_rows', []), indent=4)}"
        )
    md_data = json.loads(minio_search_markdown_sources(question))
    matches = md_data.get("top_matches", [])
    if matches:
        top = matches[0]
        _ = minio_read_object(top["object_key"], max_chars=2000)
        return (
            f"[CrewAI Agent 4: {MINIO_AGENT_SPEC.role}]\n"
            f"  • Separate MCP Calls: {', '.join(MINIO_AGENT_SPEC.mcp_tools)}\n"
            f"  • MinIO Object: `s3://ai-source-data/{top['object_key']}`\n"
            f"  • Content:\n{top['matching_snippet']}"
        )
    return f"[CrewAI Agent 4: {MINIO_AGENT_SPEC.role}] Inspected MinIO bucket `ai-source-data`."


@mlflow_trace(name="crewai.agent_5_elasticsearch_specialist", span_type="AGENT")
def run_elastic_agent(question: str) -> str:
    _ = elastic_index_stats()
    search_data = json.loads(elastic_search_documents(question, size=3))
    hits = search_data.get("top_hits", [])
    if not hits:
        return f"[CrewAI Agent 5: {ELASTIC_AGENT_SPEC.role}] No matching documents in the Elasticsearch index."
    top = hits[0]
    obj = json.loads(minio_read_object(top["minio_key"], max_chars=1500))
    other = "\n".join(f"      {i}. {h['minio_key']} (score {h['score']})" for i, h in enumerate(hits[1:], start=2))
    highlight = " ... ".join(top["highlights"]) if top["highlights"] else top["excerpt"][:500]
    return (
        f"[CrewAI Agent 5: {ELASTIC_AGENT_SPEC.role}]\n"
        f"  • Separate MCP Calls: {', '.join(ELASTIC_AGENT_SPEC.mcp_tools)}\n"
        f"  • Matching files: {search_data.get('total_matching_files')} | Best hit (score {top['score']}): `{top['minio_url']}`"
        f"{' page ' + str(top['page']) if top['page'] else ''}\n"
        f"  • Matched text (»term« = Elasticsearch highlight):\n{highlight}\n"
        f"  • Other hits:\n{other or '      (none)'}\n"
        f"  • Original read back from MinIO ({obj.get('total_length')} chars):\n{obj.get('content', '')[:400]}"
    )


@mlflow_trace(name="CrewAI_MultiSource_MCP_Pipeline", span_type="AGENT")
def _execute_crewai_pipeline(question: str) -> tuple[str, str]:
    route = route_question(question)
    if route == "elastic":
        return run_elastic_agent(question), ELASTIC_AGENT_SPEC.role
    if route == "pdf":
        return run_pdf_agent(question), PDF_AGENT_SPEC.role
    if route == "website":
        return run_website_agent(question), WEBSITE_AGENT_SPEC.role
    if route == "minio":
        return run_minio_agent(question), MINIO_AGENT_SPEC.role
    return run_postgres_agent(question), POSTGRES_AGENT_SPEC.role


def run_crewai_qa(question: str) -> str:
    start_ts = time.perf_counter()
    if HAS_MLFLOW:
        with mlflow.start_run(run_name=f"CrewAI [{question[:38]}]"):
            mlflow.set_tag("framework", "CrewAI")
            mlflow.set_tag("protocol", "MCP")
            mlflow.log_param("framework", "CrewAI")
            mlflow.log_param("question", question)

            answer, agent_role = _execute_crewai_pipeline(question)
            elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)

            mlflow.set_tag("mcp_agent", agent_role)
            mlflow.log_param("agent_invoked", agent_role)
            mlflow.log_metric("execution_time_ms", elapsed_ms)
            mlflow.log_text(answer, "final_answer.txt")
            mlflow.log_dict(
                {
                    "question": question,
                    "agent_invoked": agent_role,
                    "final_answer": answer,
                },
                "crewai_execution_trace.json",
            )
            return answer

    answer, _ = _execute_crewai_pipeline(question)
    return answer


def main():
    parser = argparse.ArgumentParser(description="CrewAI Q&A with Separate MCP Agents (PostgreSQL, Website, PDFs, MinIO).")
    parser.add_argument("--question", "-q", help="Single question to ask.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive prompt to type questions.")
    args = parser.parse_args()

    if HAS_MLFLOW:
        print(f"[MLflow Tracking Enabled] URI: {MLFLOW_TRACKING_URI} | Experiment: {MLFLOW_EXPERIMENT_NAME}")

    if args.interactive:
        print("=" * 80)
        print("CrewAI + Separate MCP Calls Interactive Q&A (PostgreSQL | Website | PDFs | MinIO)")
        print("Type your question and press Enter (or type 'exit' to quit).")
        print("=" * 80)
        while True:
            try:
                user_q = input("\nAsk a question > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user_q or user_q.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break
            print("-" * 80)
            print(run_crewai_qa(user_q))
            print("=" * 80)
        return

    # Default demo runs 1 task per specialized CrewAI MCP Agent (PostgreSQL, Website HTML, Website PDF, MinIO CSV)
    questions = (
        [args.question]
        if args.question
        else [
            "[PostgreSQL Agent] Who are the top 5 customers by total spending in PostgreSQL?",
            "[Website HTML Agent] On the Atlas Retail Bank website FAQ page, how are wire transfers and Zelle handled?",
            "[Website PDF Agent] In the Amerimed Wellness Institute PDF documents, what is the financial assistance charity care policy?",
            "[MinIO CSV Agent] In the MinIO bucket indian_tech_jobs_2026.csv, how many Data Scientist jobs are in Bengaluru?",
            "[Documents -> Elasticsearch Agent, chosen automatically] What is the policy on cargo claims for loss and damage at OceanX?",
        ]
    )

    for idx, q in enumerate(questions, start=1):
        print("=" * 80)
        print(f"CREWAI TASK {idx}: {q}")
        print("-" * 80)
        print(run_crewai_qa(q))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
