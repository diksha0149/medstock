"""
bq_tools.py — functions that answer MedStock's core questions by querying BigQuery.

Every function takes clinic_id as its first argument and every query is
scoped to it — this is the tenant-isolation boundary that keeps one clinic's
stock and sales data from ever being visible to another. clinic_id is never
something Gemini or the caller gets to choose; it's resolved server-side
from the signed-in user's account (see main.py / gemini_agent.py) before
these functions are ever called.
"""
from datetime import date, datetime
from decimal import Decimal

from google.cloud import bigquery

PROJECT_ID = "medstock-patchamomma"
DATASET = "medstock"

client = bigquery.Client(project=PROJECT_ID)


def _clean(value):
    """Convert BigQuery types that json.dumps can't handle (date, datetime,
    Decimal) into plain JSON-safe types, so results can be passed back to
    Gemini during automatic function calling."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _rows_to_dicts(query_job):
    return [{k: _clean(v) for k, v in dict(row).items()} for row in query_job.result()]


def _clinic_param(clinic_id: str) -> bigquery.ScalarQueryParameter:
    return bigquery.ScalarQueryParameter("clinic_id", "STRING", clinic_id)


def get_stockout_risk(clinic_id: str, horizon_days: int = 30) -> list[dict]:
    """Get the list of products likely to run out of stock within the next
    given number of days, soonest first.

    Args:
        clinic_id: Which clinic's data to look at.
        horizon_days: How many days ahead to check for stockout risk. Defaults to 30.
    """
    query = f"""
        WITH latest_snapshot AS (
          SELECT product_id, clinic_id, stock_level
          FROM `{PROJECT_ID}.{DATASET}.inventory_snapshots`
          WHERE clinic_id = @clinic_id
          QUALIFY ROW_NUMBER() OVER (PARTITION BY product_id, clinic_id ORDER BY snapshot_date DESC) = 1
        )
        SELECT
          p.name AS product,
          s.stock_level AS current_stock,
          f.predicted_daily_demand AS expected_daily_sales,
          ROUND(SAFE_DIVIDE(s.stock_level, f.predicted_daily_demand), 1) AS days_until_out_of_stock
        FROM `{PROJECT_ID}.{DATASET}.forecast_results` f
        JOIN latest_snapshot s USING(product_id, clinic_id)
        JOIN `{PROJECT_ID}.{DATASET}.products` p USING(product_id, clinic_id)
        WHERE f.clinic_id = @clinic_id
          AND SAFE_DIVIDE(s.stock_level, f.predicted_daily_demand) <= @horizon_days
        ORDER BY days_until_out_of_stock ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            _clinic_param(clinic_id),
            bigquery.ScalarQueryParameter("horizon_days", "INT64", horizon_days),
        ]
    )
    return _rows_to_dicts(client.query(query, job_config=job_config))


def get_demand_forecast(clinic_id: str, top_n: int = 5) -> list[dict]:
    """Get the products expected to see the highest demand growth over the
    next month, ranked highest growth first.

    Args:
        clinic_id: Which clinic's data to look at.
        top_n: How many top products to return. Defaults to 5.
    """
    query = f"""
        SELECT
          name AS product,
          recent_avg_daily AS current_daily_sales,
          pct_change AS demand_growth_percent
        FROM `{PROJECT_ID}.{DATASET}.trend_analysis`
        WHERE clinic_id = @clinic_id
        ORDER BY demand_growth_percent DESC
        LIMIT @top_n
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            _clinic_param(clinic_id),
            bigquery.ScalarQueryParameter("top_n", "INT64", top_n),
        ]
    )
    return _rows_to_dicts(client.query(query, job_config=job_config))


def get_declining_products(clinic_id: str, threshold_percent: float = -15) -> list[dict]:
    """Get products whose sales have declined by more than the given
    percentage over the last month, most declined first.

    Args:
        clinic_id: Which clinic's data to look at.
        threshold_percent: The decline percentage cutoff (negative number). Defaults to -15.
    """
    query = f"""
        SELECT
          name AS product,
          recent_avg_daily AS current_daily_sales,
          pct_change AS sales_change_percent
        FROM `{PROJECT_ID}.{DATASET}.trend_analysis`
        WHERE clinic_id = @clinic_id
          AND pct_change <= @threshold_percent
        ORDER BY sales_change_percent ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            _clinic_param(clinic_id),
            bigquery.ScalarQueryParameter("threshold_percent", "FLOAT64", threshold_percent),
        ]
    )
    return _rows_to_dicts(client.query(query, job_config=job_config))


def get_reorder_recommendation(clinic_id: str) -> list[dict]:
    """Get how much stock to keep for each product, and whether it's time to
    reorder right now, most urgent first.

    Args:
        clinic_id: Which clinic's data to look at.
    """
    query = f"""
        WITH latest_snapshot AS (
          SELECT product_id, clinic_id, stock_level
          FROM `{PROJECT_ID}.{DATASET}.inventory_snapshots`
          WHERE clinic_id = @clinic_id
          QUALIFY ROW_NUMBER() OVER (PARTITION BY product_id, clinic_id ORDER BY snapshot_date DESC) = 1
        )
        SELECT
          p.name AS product,
          s.stock_level AS current_stock,
          r.reorder_point AS reorder_when_stock_drops_to,
          CASE WHEN s.stock_level <= r.reorder_point THEN 'Reorder Now' ELSE 'OK' END AS status
        FROM `{PROJECT_ID}.{DATASET}.reorder_recommendation` r
        JOIN latest_snapshot s USING(product_id, clinic_id)
        JOIN `{PROJECT_ID}.{DATASET}.products` p USING(product_id, clinic_id)
        WHERE r.clinic_id = @clinic_id
        ORDER BY (s.stock_level - r.reorder_point) ASC
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[_clinic_param(clinic_id)])
    return _rows_to_dicts(client.query(query, job_config=job_config))


def get_expiry_risk(clinic_id: str, horizon_days: int = 60) -> list[dict]:
    """Get stock batches that are likely to expire before they're expected
    to sell through, soonest-expiring first.

    Note: batch data only exists for the original demo clinic today (there's
    no upload path for batches yet) — a newly signed-up clinic will simply
    see no results here until that's added.

    Args:
        clinic_id: Which clinic's data to look at.
        horizon_days: Only include batches expiring within this many days. Defaults to 60.
    """
    query = f"""
        SELECT
          p.name AS product,
          b.batch_id,
          b.expiry_date,
          DATE_DIFF(b.expiry_date, CURRENT_DATE(), DAY) AS days_until_expiry,
          ROUND(SAFE_DIVIDE(b.quantity_received, f.predicted_daily_demand),1) AS days_needed_to_sell_remaining_stock
        FROM `{PROJECT_ID}.{DATASET}.batches` b
        JOIN `{PROJECT_ID}.{DATASET}.products` p USING(product_id, clinic_id)
        JOIN `{PROJECT_ID}.{DATASET}.forecast_results` f USING(product_id, clinic_id)
        WHERE p.clinic_id = @clinic_id
          AND DATE_DIFF(b.expiry_date, CURRENT_DATE(), DAY) < SAFE_DIVIDE(b.quantity_received, f.predicted_daily_demand)
          AND DATE_DIFF(b.expiry_date, CURRENT_DATE(), DAY) <= @horizon_days
        ORDER BY days_until_expiry ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            _clinic_param(clinic_id),
            bigquery.ScalarQueryParameter("horizon_days", "INT64", horizon_days),
        ]
    )
    return _rows_to_dicts(client.query(query, job_config=job_config))
