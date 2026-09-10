"""
main.py — FastAPI backend for MedStock. Exposes the /ask endpoint that
clinic staff use to ask natural-language questions about stock and demand,
plus a /dashboard/summary endpoint that powers the at-a-glance dashboard
(queries BigQuery directly, skipping Gemini, so it loads fast).

Run locally with:
    uvicorn main:app --reload

Then test at http://127.0.0.1:8000/docs (FastAPI's built-in interactive UI),
or open static/index.html in your browser for the actual dashboard.
"""
from concurrent.futures import ThreadPoolExecutor

import firebase_admin
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from firebase_admin import auth as firebase_auth
from pydantic import BaseModel
import gemini_agent
import bq_tools
import bq_upload

app = FastAPI(title="MedStock API")

# Uses Application Default Credentials — the same auth already set up for
# BigQuery — so no separate service account key file is needed, either
# locally or on Cloud Run.
firebase_admin.initialize_app()


def verify_token(authorization: str = Header(None)):
    """FastAPI dependency that checks for a valid Firebase ID token in the
    Authorization header. Attach with `Depends(verify_token)` on any route
    that should require a logged-in user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    id_token = authorization.split(" ", 1)[1]
    try:
        return firebase_auth.verify_id_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# uid -> clinic_id. A BigQuery lookup per request would add real latency to
# every single call, and a user's clinic never changes after sign-up, so a
# simple in-memory cache is enough here (resets on redeploy — harmless).
_clinic_cache: dict[str, str] = {}


def get_current_clinic_id(user=Depends(verify_token)) -> str:
    """FastAPI dependency that resolves the signed-in user's clinic_id.
    This is the ONLY place clinic_id is allowed to come from — never a
    request body or query param — so a user can never read or write another
    clinic's data by passing a different value themselves."""
    uid = user["uid"]
    if uid not in _clinic_cache:
        _clinic_cache[uid] = bq_upload.get_clinic_id_for_uid(uid)
    return _clinic_cache[uid]

# Allow the static dashboard page (opened directly in a browser, or served
# from a different port) to call this API. Fine for local development;
# tighten this to your real domain before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class ClinicRequest(BaseModel):
    clinic_name: str


class ClinicResponse(BaseModel):
    clinic_id: str
    clinic_name: str


@app.get("/")
def root():
    return {"status": "MedStock API is running"}


@app.post("/register-clinic", response_model=ClinicResponse)
def register_clinic(request: ClinicRequest, user=Depends(verify_token)):
    """Called once, right after a brand-new user signs up. Creates their
    clinic_id and seeds it with a copy of the sample product catalog."""
    uid = user["uid"]
    if uid in _clinic_cache:
        # Already registered (e.g. a page refresh re-firing this call) —
        # just return their existing clinic rather than creating a duplicate.
        clinic_id = _clinic_cache[uid]
        return ClinicResponse(clinic_id=clinic_id, clinic_name=bq_upload.get_clinic_name(clinic_id))

    result = bq_upload.register_clinic(
        uid=uid,
        email=user.get("email", ""),
        clinic_name=request.clinic_name,
    )
    _clinic_cache[uid] = result["clinic_id"]
    return ClinicResponse(**result)


@app.get("/me", response_model=ClinicResponse)
def me(clinic_id: str = Depends(get_current_clinic_id)):
    """Returns the signed-in user's clinic, for display in the dashboard header."""
    return ClinicResponse(clinic_id=clinic_id, clinic_name=bq_upload.get_clinic_name(clinic_id))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, clinic_id: str = Depends(get_current_clinic_id)):
    answer = gemini_agent.ask(request.question, clinic_id)
    return AskResponse(answer=answer)


@app.get("/dashboard/summary")
def dashboard_summary(clinic_id: str = Depends(get_current_clinic_id)):
    """Returns the raw data behind all five core questions, straight from
    BigQuery — no Gemini involved, so this loads fast for dashboard charts.
    Every query is scoped to the signed-in user's own clinic.

    The five queries don't depend on each other, so they're run concurrently
    (each one waits on network I/O, not CPU) instead of one after another —
    this cuts wall-clock time roughly to the slowest single query instead of
    the sum of all five.
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        stockout = executor.submit(bq_tools.get_stockout_risk, clinic_id, horizon_days=30)
        rising = executor.submit(bq_tools.get_demand_forecast, clinic_id, top_n=5)
        declining = executor.submit(bq_tools.get_declining_products, clinic_id, threshold_percent=-15)
        reorder = executor.submit(bq_tools.get_reorder_recommendation, clinic_id)
        expiry = executor.submit(bq_tools.get_expiry_risk, clinic_id, horizon_days=60)

        return {
            "stockout_risk": stockout.result(),
            "rising_demand": rising.result(),
            "declining_products": declining.result(),
            "reorder_recommendation": reorder.result(),
            "expiry_risk": expiry.result(),
        }


@app.post("/upload")
async def upload_csv(
    table: str = Form(...),
    file: UploadFile = File(...),
    clinic_id: str = Depends(get_current_clinic_id),
):
    """Append a CSV export to a BigQuery table, tagged with the signed-in
    user's own clinic_id (never taken from the file itself).

    table: "dispensing_records" (sales) or "inventory_snapshots" (stock counts).
    file: the CSV file, with columns matching the target table.
    """
    contents = await file.read()
    try:
        return bq_upload.upload_csv(clinic_id, table, contents)
    except ValueError as e:
        return {"error": str(e)}


# Serves everything in the "static" folder — visit http://127.0.0.1:8000/static/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")
