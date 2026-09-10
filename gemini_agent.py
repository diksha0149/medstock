"""
gemini_agent.py — wires Gemini up to the bq_tools functions using automatic
function calling. Gemini reads each function's name, docstring, and type
hints to decide which one to call for a given question, runs it, and turns
the result into a plain-language answer.

Every bq_tools function requires a clinic_id — but clinic_id must never be
something Gemini gets to pick (a user could otherwise ask a leading question
and get another clinic's data). So instead of handing Gemini the raw
bq_tools functions, build_tools() wraps each one in a small closure that has
clinic_id baked in ahead of time. Gemini only ever sees the "public"
parameters (horizon_days, top_n, etc.) — clinic_id is resolved server-side
per request and is invisible to the model.
"""
from google import genai
from google.genai import types
import bq_tools

PROJECT_ID = "medstock-patchamomma"
LOCATION = "global"  # newer Gemini 3.x models are served via the global endpoint,
                      # not regional endpoints like us-central1

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

SYSTEM_INSTRUCTION = (
    "You are MedStock's assistant for a clinic's medical store. The staff "
    "member's questions should be about stock levels, sales demand, "
    "reorder timing, or expiry risk — that's what the available tools cover.\n\n"
    "If the question clearly relates to one of those topics, call exactly "
    "one matching tool, then summarize the result in plain, friendly "
    "language. Mention specific product names and numbers from the tool "
    "result. Keep it concise — a few sentences, not a wall of text.\n\n"
    "If the question is unclear, unrelated to stock/sales/reorder/expiry, "
    "or just gibberish, do NOT call a tool — instead ask a brief, friendly "
    "clarifying question about what they'd like to know regarding their "
    "clinic's stock.\n\n"
    "If a tool returns an empty result, do not assume everything is healthy "
    "— an empty result can also mean this clinic hasn't uploaded any sales "
    "or stock data yet. In that case, say something like: \"There's no data "
    "to show yet — upload your clinic's sales and stock data using the "
    "'+ Add Data' button to get insights here.\" Only give a reassuring "
    "'everything looks fine' answer if the conversation has already shown "
    "real data for this clinic (e.g. an earlier answer had real numbers in "
    "it), since that confirms data actually exists."
)


MODEL_NAME = "gemini-3.5-flash"  # current Flash-tier model on Vertex AI as of Aug 2026;
                                  # gemini-2.0-flash was retired June 1, 2026


def build_tools(clinic_id: str):
    """Returns the five tools Gemini is allowed to call for one request,
    each scoped to a single clinic. Nothing else is exposed — this is what
    keeps the AI layer safe: it can only ever run one of these five
    pre-built, parameterized queries, never arbitrary SQL, and never for a
    clinic other than the one making the request."""

    def get_stockout_risk(horizon_days: int = 30) -> list[dict]:
        """Get the list of products likely to run out of stock within the
        next given number of days, soonest first.

        Args:
            horizon_days: How many days ahead to check for stockout risk. Defaults to 30.
        """
        return bq_tools.get_stockout_risk(clinic_id, horizon_days=horizon_days)

    def get_demand_forecast(top_n: int = 5) -> list[dict]:
        """Get the products expected to see the highest demand growth over
        the next month, ranked highest growth first.

        Args:
            top_n: How many top products to return. Defaults to 5.
        """
        return bq_tools.get_demand_forecast(clinic_id, top_n=top_n)

    def get_declining_products(threshold_percent: float = -15) -> list[dict]:
        """Get products whose sales have declined by more than the given
        percentage over the last month, most declined first.

        Args:
            threshold_percent: The decline percentage cutoff (negative number). Defaults to -15.
        """
        return bq_tools.get_declining_products(clinic_id, threshold_percent=threshold_percent)

    def get_reorder_recommendation() -> list[dict]:
        """Get how much stock to keep for each product, and whether it's
        time to reorder right now, most urgent first.
        """
        return bq_tools.get_reorder_recommendation(clinic_id)

    def get_expiry_risk(horizon_days: int = 60) -> list[dict]:
        """Get stock batches that are likely to expire before they're
        expected to sell through, soonest-expiring first.

        Args:
            horizon_days: Only include batches expiring within this many days. Defaults to 60.
        """
        return bq_tools.get_expiry_risk(clinic_id, horizon_days=horizon_days)

    return [
        get_stockout_risk,
        get_demand_forecast,
        get_declining_products,
        get_reorder_recommendation,
        get_expiry_risk,
    ]


def ask(question: str, clinic_id: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=question,
        config=types.GenerateContentConfig(
            tools=build_tools(clinic_id),
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )
    return response.text


if __name__ == "__main__":
    # Quick manual test from the command line:
    #   python gemini_agent.py "which products will run out in 30 days?"
    import sys
    q = " ".join(sys.argv[1:]) or "Which products are likely to run out of stock in the next 30 days?"
    print(f"Q: {q}\n")
    print(ask(q, "clinic_001"))
