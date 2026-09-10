"""
bq_upload.py — CSV upload logic, plus per-clinic account setup.

Lets clinic staff push a fresh export of sales (dispensing_records) or stock
counts (inventory_snapshots) straight into BigQuery, instead of someone
manually running `bq load` from a terminal every time.

Every row that goes into BigQuery is tagged with clinic_id — the tenant key
that keeps one clinic's data from ever being visible to another. Uploaded
rows are always stamped with the *caller's own* clinic_id server-side; the
CSV's own clinic_id column (if it has one) is ignored, so a user can never
write into another clinic's data by editing a spreadsheet.
"""
import io
import re
import secrets
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "medstock-patchamomma"
DATASET = "medstock"

client = bigquery.Client(project=PROJECT_ID)

# The original demo clinic — every brand-new clinic's product catalog is
# seeded as a copy of this one's, and any signed-in user who was never
# routed through /register-clinic (e.g. the original staff@medstock.demo
# account) defaults to this clinic.
DEFAULT_CLINIC_ID = "clinic_001"

# Which tables staff are allowed to upload into, and which column in each
# one holds a date (so it can be parsed correctly before loading).
ALLOWED_TABLES = {
    "dispensing_records": "transaction_date",
    "inventory_snapshots": "snapshot_date",
}


def _run_query(sql: str, params: list = None):
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    job = client.query(sql, job_config=job_config)
    job.result()  # wait for it to finish; raises on failure


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "clinic"


def get_clinic_id_for_uid(uid: str) -> str:
    """Looks up which clinic a signed-in user belongs to, by their Firebase
    uid. Falls back to the original demo clinic for accounts that never went
    through /register-clinic."""
    query = f"""
        SELECT clinic_id FROM `{PROJECT_ID}.{DATASET}.clinics`
        WHERE owner_uid = @uid
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
    )
    rows = list(client.query(query, job_config=job_config).result())
    return rows[0]["clinic_id"] if rows else DEFAULT_CLINIC_ID


def get_clinic_name(clinic_id: str) -> str:
    """Returns the display name for a clinic_id, for showing in the UI."""
    if clinic_id == DEFAULT_CLINIC_ID:
        return "Demo Clinic"
    query = f"""
        SELECT clinic_name FROM `{PROJECT_ID}.{DATASET}.clinics`
        WHERE clinic_id = @clinic_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("clinic_id", "STRING", clinic_id)]
    )
    rows = list(client.query(query, job_config=job_config).result())
    return rows[0]["clinic_name"] if rows else "Your Clinic"


def register_clinic(uid: str, email: str, clinic_name: str) -> dict:
    """Called once, right after a new user signs up. Creates a new clinic_id
    for them and seeds it with a copy of the sample product catalog (from
    the original demo clinic), so they have products to attach their own
    uploaded sales/stock data to right away instead of starting with an
    empty catalog."""
    slug = _slugify(clinic_name)
    clinic_id = f"{slug}_{secrets.token_hex(3)}"

    params = [
        bigquery.ScalarQueryParameter("clinic_id", "STRING", clinic_id),
        bigquery.ScalarQueryParameter("clinic_name", "STRING", clinic_name),
        bigquery.ScalarQueryParameter("uid", "STRING", uid),
        bigquery.ScalarQueryParameter("email", "STRING", email),
    ]

    insert_clinic_sql = f"""
        INSERT INTO `{PROJECT_ID}.{DATASET}.clinics`
          (clinic_id, clinic_name, owner_uid, owner_email, created_at)
        VALUES (@clinic_id, @clinic_name, @uid, @email, CURRENT_TIMESTAMP())
    """
    seed_products_sql = f"""
        INSERT INTO `{PROJECT_ID}.{DATASET}.products`
          (product_id, clinic_id, name, category, prescription_required, cost_price, lead_time_days)
        SELECT product_id, @clinic_id, name, category, prescription_required, cost_price, lead_time_days
        FROM `{PROJECT_ID}.{DATASET}.products`
        WHERE clinic_id = '{DEFAULT_CLINIC_ID}'
    """
    _run_query(insert_clinic_sql, params)
    _run_query(seed_products_sql, params)

    return {"clinic_id": clinic_id, "clinic_name": clinic_name}


# Recomputes the derived analytics tables (daily_sales, forecast_results,
# trend_analysis, reorder_recommendation) for ONE clinic — DELETE the old
# rows for that clinic_id, then INSERT freshly computed ones. This never
# touches other clinics' rows in the same tables, unlike a blanket
# CREATE OR REPLACE TABLE would.
RECOMPUTE_QUERIES = [
    # 1. daily_sales — zero-filled date spine per product, so moving averages
    # correctly count days with no sales as 0 rather than skipping them.
    # Spans from the earliest to the latest transaction_date actually in
    # this clinic's data (not wall-clock "today" — the synthetic dataset
    # doesn't extend to the present day).
    """
    DELETE FROM `{project}.{dataset}.daily_sales` WHERE clinic_id = @clinic_id;

    INSERT INTO `{project}.{dataset}.daily_sales` (product_id, date, quantity, clinic_id)
    WITH bounds AS (
      SELECT MIN(transaction_date) AS min_date, MAX(transaction_date) AS max_date
      FROM `{project}.{dataset}.dispensing_records`
      WHERE clinic_id = @clinic_id
    ),
    date_spine AS (
      SELECT date
      FROM bounds, UNNEST(GENERATE_DATE_ARRAY(bounds.min_date, bounds.max_date)) AS date
    ),
    products_x_dates AS (
      SELECT p.product_id, d.date
      FROM `{project}.{dataset}.products` p
      CROSS JOIN date_spine d
      WHERE p.clinic_id = @clinic_id
    ),
    daily_totals AS (
      SELECT product_id, transaction_date AS date, SUM(quantity) AS quantity
      FROM `{project}.{dataset}.dispensing_records`
      WHERE clinic_id = @clinic_id
      GROUP BY product_id, transaction_date
    )
    SELECT
      pxd.product_id,
      pxd.date,
      COALESCE(dt.quantity, 0) AS quantity,
      @clinic_id AS clinic_id
    FROM products_x_dates pxd
    LEFT JOIN daily_totals dt USING (product_id, date)
    """,

    # 2. forecast_results — moving-average demand forecast over the 30 days
    # up to the latest date actually present in this clinic's daily_sales.
    """
    DELETE FROM `{project}.{dataset}.forecast_results` WHERE clinic_id = @clinic_id;

    INSERT INTO `{project}.{dataset}.forecast_results`
      (product_id, forecast_date, predicted_daily_demand, predicted_demand_30d, clinic_id)
    WITH ref AS (
      SELECT MAX(date) AS reference_date FROM `{project}.{dataset}.daily_sales` WHERE clinic_id = @clinic_id
    )
    SELECT
      product_id,
      ANY_VALUE(ref.reference_date) AS forecast_date,
      ROUND(AVG(quantity), 1) AS predicted_daily_demand,
      ROUND(AVG(quantity) * 30, 1) AS predicted_demand_30d,
      @clinic_id AS clinic_id
    FROM `{project}.{dataset}.daily_sales`, ref
    WHERE clinic_id = @clinic_id
      AND date >= DATE_SUB(ref.reference_date, INTERVAL 30 DAY)
    GROUP BY product_id
    """,

    # 3. trend_analysis — recent 28 days vs the 28 days before that, both
    # measured back from the latest date in this clinic's data.
    """
    DELETE FROM `{project}.{dataset}.trend_analysis` WHERE clinic_id = @clinic_id;

    INSERT INTO `{project}.{dataset}.trend_analysis`
      (product_id, name, recent_avg_daily, prior_avg_daily, pct_change, clinic_id)
    WITH ref AS (
      SELECT MAX(date) AS reference_date FROM `{project}.{dataset}.daily_sales` WHERE clinic_id = @clinic_id
    ),
    recent AS (
      SELECT product_id, ROUND(AVG(quantity), 1) AS recent_avg_daily
      FROM `{project}.{dataset}.daily_sales`, ref
      WHERE clinic_id = @clinic_id AND date >= DATE_SUB(ref.reference_date, INTERVAL 28 DAY)
      GROUP BY product_id
    ),
    prior AS (
      SELECT product_id, ROUND(AVG(quantity), 1) AS prior_avg_daily
      FROM `{project}.{dataset}.daily_sales`, ref
      WHERE clinic_id = @clinic_id
        AND date >= DATE_SUB(ref.reference_date, INTERVAL 56 DAY)
        AND date < DATE_SUB(ref.reference_date, INTERVAL 28 DAY)
      GROUP BY product_id
    )
    SELECT
      p.product_id,
      p.name,
      r.recent_avg_daily,
      pr.prior_avg_daily,
      ROUND(SAFE_DIVIDE(r.recent_avg_daily - pr.prior_avg_daily, pr.prior_avg_daily) * 100, 1) AS pct_change,
      @clinic_id AS clinic_id
    FROM `{project}.{dataset}.products` p
    JOIN recent r USING (product_id)
    JOIN prior pr USING (product_id)
    WHERE p.clinic_id = @clinic_id
    """,

    # 4. reorder_recommendation — reorder_point = (avg_daily_demand * lead_time_days)
    # + safety_stock, where safety_stock = 1.65 * stddev(daily demand) * sqrt(lead_time_days).
    """
    DELETE FROM `{project}.{dataset}.reorder_recommendation` WHERE clinic_id = @clinic_id;

    INSERT INTO `{project}.{dataset}.reorder_recommendation`
      (product_id, name, lead_time_days, avg_daily_demand, reorder_point, clinic_id)
    WITH ref AS (
      SELECT MAX(date) AS reference_date FROM `{project}.{dataset}.daily_sales` WHERE clinic_id = @clinic_id
    ),
    demand_stats AS (
      SELECT
        product_id,
        ROUND(AVG(quantity), 1) AS avg_daily_demand,
        STDDEV(quantity) AS stddev_daily_demand
      FROM `{project}.{dataset}.daily_sales`, ref
      WHERE clinic_id = @clinic_id AND date >= DATE_SUB(ref.reference_date, INTERVAL 30 DAY)
      GROUP BY product_id
    )
    SELECT
      p.product_id,
      p.name,
      p.lead_time_days,
      ds.avg_daily_demand,
      ROUND(
        (ds.avg_daily_demand * p.lead_time_days)
        + (1.65 * COALESCE(ds.stddev_daily_demand, 0) * SQRT(p.lead_time_days))
      ) AS reorder_point,
      @clinic_id AS clinic_id
    FROM `{project}.{dataset}.products` p
    JOIN demand_stats ds USING (product_id)
    WHERE p.clinic_id = @clinic_id
    """,
]


def recompute_insights(clinic_id: str):
    """Re-run the forecasting/trend/reorder pipeline for ONE clinic, so
    newly uploaded data is reflected in what that clinic's dashboard shows.
    Called automatically after a successful dispensing_records upload.

    daily_sales has to finish first — everything else reads from it. But
    forecast_results, trend_analysis, and reorder_recommendation don't
    depend on each other, so those three run concurrently instead of one
    after another, cutting the wait roughly in half.
    """
    params = [bigquery.ScalarQueryParameter("clinic_id", "STRING", clinic_id)]

    daily_sales_sql = RECOMPUTE_QUERIES[0].format(project=PROJECT_ID, dataset=DATASET)
    _run_query(daily_sales_sql, params)

    remaining = [q.format(project=PROJECT_ID, dataset=DATASET) for q in RECOMPUTE_QUERIES[1:]]
    with ThreadPoolExecutor(max_workers=len(remaining)) as executor:
        futures = [executor.submit(_run_query, q, params) for q in remaining]
        for f in futures:
            f.result()  # re-raise any exception from the worker thread


def upload_csv(clinic_id: str, table_name: str, csv_bytes: bytes) -> dict:
    """Append the rows in a CSV file to a BigQuery table, tagged with the
    caller's own clinic_id.

    Args:
        clinic_id: The uploading user's clinic — resolved server-side from
            their verified login, never taken from the request.
        table_name: Must be one of ALLOWED_TABLES ("dispensing_records" or
            "inventory_snapshots").
        csv_bytes: Raw bytes of the uploaded CSV file.
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(
            f"Unknown table '{table_name}'. Allowed values: {', '.join(ALLOWED_TABLES)}"
        )

    date_column = ALLOWED_TABLES[table_name]
    df = pd.read_csv(io.BytesIO(csv_bytes), parse_dates=[date_column])
    df[date_column] = df[date_column].dt.date
    df["clinic_id"] = clinic_id  # always the authenticated caller's clinic — never trust the CSV's own value

    table_ref = f"{PROJECT_ID}.{DATASET}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # waits for the load to finish; raises on failure

    # Only sales data (dispensing_records) feeds the forecasting pipeline —
    # a new inventory snapshot alone doesn't need a recompute.
    if table_name == "dispensing_records":
        recompute_insights(clinic_id)

    return {"table": table_name, "rows_loaded": len(df)}
