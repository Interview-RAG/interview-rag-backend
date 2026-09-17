"""Job listings, via the Adzuna API.

Adzuna was chosen over JSearch/Jooble because it is the only free option with a
native recency filter (`max_days_old` + `sort_by=date`), which is what "show me
newly opened roles" actually needs. It covers India as country code `in`.

Two quirks of the data worth knowing before you render it:
  * `description` is an excerpt, not the full posting.
  * `salary_is_predicted` means Adzuna *estimated* the figure. Never show it as
    an employer-stated salary — `Job.salary_text()` labels it "est.".

Get a free instant key at https://developer.adzuna.com/signup and set
ADZUNA_APP_ID / ADZUNA_APP_KEY in backend/.env.
"""
import asyncio
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
# Adzuna scopes every query to one country. India by default; override per deploy.
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "in")

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

# "Newly opening" when the user gives no timeframe of their own.
DEFAULT_MAX_DAYS_OLD = 7
MAX_RESULTS_PER_PAGE = 50

# Adzuna returns these when it is throttling or having a bad day. Anything else
# (400 for a malformed filter, 401 for a bad key) is our fault and not retried.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class ProviderUnavailable(Exception):
    """The job board could not be reached, or refused us.

    Deliberately distinct from "the search ran and matched nothing" — conflating
    the two is how you end up telling someone there are no jobs when in fact you
    never asked.
    """


class ProviderNotConfigured(ProviderUnavailable):
    """No API credentials were set."""


@dataclass
class Job:
    id: str
    title: str
    company: str
    location: str
    url: str
    posted_at: str | None          # ISO date string, as Adzuna gives it
    salary_min: float | None
    salary_max: float | None
    salary_is_predicted: bool
    contract_type: str | None
    source: str
    snippet: str

    def age_days(self) -> int | None:
        if not self.posted_at:
            return None
        try:
            posted = datetime.fromisoformat(self.posted_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        if posted.tzinfo is None:
            posted = posted.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - posted).days)

    def posted_text(self) -> str:
        days = self.age_days()
        if days is None:
            return "date unknown"
        if days == 0:
            return "today"
        if days == 1:
            return "yesterday"
        return f"{days} days ago"

    def salary_text(self) -> str | None:
        """Human-readable pay, always flagged when Adzuna guessed it."""
        if not self.salary_min and not self.salary_max:
            return None

        def money(value):
            if value is None:
                return None
            value = float(value)
            if value >= 100000:
                return f"{value / 100000:.1f}L".replace(".0L", "L")
            return f"{value:,.0f}"

        low, high = money(self.salary_min), money(self.salary_max)
        if low and high and low != high:
            text = f"{low}-{high}"
        else:
            text = low or high
        suffix = " (est.)" if self.salary_is_predicted else ""
        return f"{text}{suffix}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["posted_text"] = self.posted_text()
        data["salary_text"] = self.salary_text()
        return data


def _clean(text) -> str:
    """Collapse the whitespace Adzuna leaves in description excerpts."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _dedupe(jobs: list) -> list:
    """Drop repeats. Adzuna returns the same role several times in one page —
    measured at roughly one duplicate per ten results — usually because an
    agency has posted it more than once."""
    seen = set()
    unique = []
    for job in jobs:
        key = (job.id, re.sub(r"\W+", "", job.title.lower()), job.company.lower())
        loose = (key[1], key[2])
        if loose in seen:
            continue
        seen.add(loose)
        unique.append(job)
    return unique


def _to_job(raw: dict) -> Job:
    return Job(
        id=str(raw.get("id") or ""),
        title=_clean(raw.get("title")) or "Untitled role",
        company=_clean((raw.get("company") or {}).get("display_name")) or "Unknown company",
        location=_clean((raw.get("location") or {}).get("display_name")),
        url=raw.get("redirect_url") or "",
        posted_at=raw.get("created"),
        salary_min=raw.get("salary_min"),
        salary_max=raw.get("salary_max"),
        salary_is_predicted=str(raw.get("salary_is_predicted", "0")) == "1",
        contract_type=raw.get("contract_time") or raw.get("contract_type"),
        source="adzuna",
        snippet=_clean(raw.get("description"))[:400],
    )


class AdzunaProvider:
    name = "adzuna"

    def __init__(self, app_id: str | None = None, app_key: str | None = None,
                 country: str | None = None):
        # None means "read the environment"; an empty string means "explicitly
        # no credentials". Collapsing the two would make it impossible to
        # construct an unconfigured provider on a machine that has keys set.
        self.app_id = ADZUNA_APP_ID if app_id is None else app_id
        self.app_key = ADZUNA_APP_KEY if app_key is None else app_key
        self.country = (ADZUNA_COUNTRY if country is None else country).lower()

    @property
    def configured(self) -> bool:
        return bool(self.app_id and self.app_key)

    async def search(
        self,
        what: str = "",
        where: str = "",
        max_days_old: int = DEFAULT_MAX_DAYS_OLD,
        salary_min: int | None = None,
        salary_max: int | None = None,
        distance: int | None = None,
        full_time: bool = False,
        part_time: bool = False,
        contract: bool = False,
        permanent: bool = False,
        what_exclude: str = "",
        sort_by: str = "date",
        results: int = 10,
        page: int = 1,
        attempts: int = 3,
    ) -> list[Job]:
        """One page of listings, newest first by default.

        Raises ProviderNotConfigured when there are no credentials and
        ProviderUnavailable when Adzuna is throttling or unreachable.
        """
        if not self.configured:
            raise ProviderNotConfigured(
                "ADZUNA_APP_ID / ADZUNA_APP_KEY are not set. Get a free key at "
                "https://developer.adzuna.com/signup"
            )

        wanted = max(1, min(int(results or 10), MAX_RESULTS_PER_PAGE))
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            # Over-fetch so de-duplication does not leave us short.
            "results_per_page": min(wanted * 2, MAX_RESULTS_PER_PAGE),
            "content-type": "application/json",
        }
        if what:
            params["what"] = what
        if what_exclude:
            params["what_exclude"] = what_exclude
        if where:
            params["where"] = where
        if distance:
            params["distance"] = int(distance)
        if max_days_old:
            params["max_days_old"] = max(1, int(max_days_old))
        if salary_min:
            params["salary_min"] = int(salary_min)
        if salary_max:
            params["salary_max"] = int(salary_max)
        if sort_by in ("date", "salary", "relevance"):
            params["sort_by"] = sort_by
        for flag, enabled in (("full_time", full_time), ("part_time", part_time),
                              ("contract", contract), ("permanent", permanent)):
            if enabled:
                params[flag] = 1

        url = BASE_URL.format(country=self.country, page=max(1, int(page or 1)))
        last_error = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(attempts):
                try:
                    resp = await client.get(url, params=params)

                    if resp.status_code in TRANSIENT_STATUS:
                        last_error = f"Adzuna returned {resp.status_code}"
                        delay = _retry_after(resp) or 2 ** attempt
                    elif resp.status_code >= 400:
                        # Our request is wrong; retrying cannot help.
                        raise ProviderUnavailable(
                            f"Adzuna rejected the search ({resp.status_code}): "
                            f"{resp.text[:200]}"
                        )
                    else:
                        payload = resp.json()
                        jobs = [_to_job(r) for r in (payload.get("results") or [])]
                        return _dedupe(jobs)[:wanted]

                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last_error = f"{type(e).__name__}: {e}"
                    delay = 2 ** attempt

                if attempt < attempts - 1:
                    print(f"Adzuna unavailable ({last_error}), retrying in {delay}s")
                    await asyncio.sleep(delay)

        raise ProviderUnavailable(str(last_error))


def _retry_after(resp) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return min(float(raw), 30.0)
    except ValueError:
        return None


def format_jobs_markdown(jobs: list[Job], new_ids: set | None = None) -> str:
    """Render listings as Markdown.

    The agent passes this straight through to the user, so it has to be readable
    on its own — the model is told not to re-summarise it.
    """
    if not jobs:
        return "No listings matched."

    new_ids = new_ids or set()
    lines = []
    for job in jobs:
        flag = " **· new**" if job.id in new_ids else ""
        lines.append(f"**[{job.title}]({job.url})** — {job.company}{flag}")

        meta = [job.location or "location not stated", f"posted {job.posted_text()}"]
        salary = job.salary_text()
        if salary:
            meta.append(salary)
        if job.contract_type:
            meta.append(str(job.contract_type).replace("_", " "))
        lines.append("  " + " · ".join(meta))

        if job.snippet:
            lines.append(f"  {job.snippet[:200]}…")
        lines.append("")

    return "\n".join(lines).strip()


# Module-level default, mirroring how rag.py exposes a ready client.
adzuna = AdzunaProvider()
