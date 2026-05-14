"""FastAPI receiver for GitHub webhooks."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse

from robomp import github_events
from robomp.config import Settings, get_settings
from robomp.dashboard import INDEX_HTML, tail_jsonl
from robomp.db import (
    INACTIVE_EVENT_STATES,
    Database,
    get_database,
    iso_seconds_ago,
)
from robomp.db import (
    issue_key as make_issue_key,
)
from robomp.github_client import GitHubClient, GitHubError
from robomp.manual_triage import (
    InvalidIssueRef,
    ManualTriageConflict,
    ManualTriageError,
    enqueue_manual_triage,
    parse_issue_ref,
)
from robomp.queue import WorkerPool
from robomp.sandbox import SandboxManager

log = logging.getLogger(__name__)


def _build_state(settings: Settings) -> dict[str, Any]:
    db = get_database(settings.sqlite_path)
    github = GitHubClient(settings.github_token.get_secret_value())
    sandbox = SandboxManager(settings.workspace_root)
    pool = WorkerPool(settings=settings, db=db, github=github, sandbox=sandbox)
    return {"settings": settings, "db": db, "github": github, "sandbox": sandbox, "pool": pool}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. `settings` parameter is for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        cfg.ensure_paths()
        app.state.bag = _build_state(cfg)
        app.state.bag["started_at"] = time.time()
        pool: WorkerPool = app.state.bag["pool"]
        await pool.start()
        try:
            yield
        finally:
            await pool.stop()

    app = FastAPI(title="robomp", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        pool = request.app.state.bag.get("pool")
        if pool is None:
            raise HTTPException(503, "not initialized")
        return {"status": "ready"}

    @app.post("/webhook/github")
    async def webhook(
        request: Request,
        x_github_event: str = Header(..., alias="X-GitHub-Event"),
        x_github_delivery: str = Header(..., alias="X-GitHub-Delivery"),
        x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
    ) -> JSONResponse:
        bag = request.app.state.bag
        cfg: Settings = bag["settings"]
        body = await request.body()
        if not github_events.verify_signature(
            cfg.github_webhook_secret.get_secret_value(),
            body,
            x_hub_signature_256,
        ):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid signature")
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"invalid json: {exc}") from exc

        db: Database = bag["db"]

        def _resolve(repo_full: str, pr_number: int) -> str | None:
            row = db.find_issue_by_pr(repo_full, pr_number)
            return row.key if row else None

        decision = github_events.route(
            x_github_event,
            payload,
            allowlist=cfg.repo_allowlist,
            bot_login=cfg.bot_login,
            maintainers=cfg.maintainer_logins,
            reviewer_bots=cfg.reviewer_bots,
            resolve_issue_from_pr=_resolve,
        )

        # Persist directive metadata on the stored payload so the durable
        # queue (and any replay) carries the maintainer signal forward.
        if decision.directive:
            payload = dict(payload)
            payload["_robomp_directive"] = {
                "body": decision.directive_body,
                "author": decision.directive_author,
            }

        if not decision.should_queue:
            log.info("skip", extra={"event": x_github_event, "reason": decision.reason})
            db.record_event(
                delivery_id=x_github_delivery,
                event_type=x_github_event,
                repo=decision.repo,
                issue_key=decision.issue_key,
                payload=payload,
                state="skipped",
                last_error=decision.reason,
            )
            return JSONResponse({"delivery": x_github_delivery, "state": "skipped"}, status_code=202)

        # Per-user rate limiting. Lifecycle events (cleanup) carry no submitter
        # and are not gated. For everything user-driven, atomically record the
        # accepted delivery while checking the rolling window against the tier cap.
        submitter = decision.submitter
        if submitter:
            cap = github_events.rate_limit_cap(
                submitter,
                decision.association,
                unlimited=cfg.rate_limit_unlimited | cfg.maintainer_logins,
                default=cfg.rate_limit_default,
                contributor=cfg.rate_limit_contributor,
            )
            since = iso_seconds_ago(cfg.rate_limit_window_seconds)
            admission = db.admit_submission(
                delivery_id=x_github_delivery,
                login=submitter,
                repo=decision.repo,
                since=since,
                cap=cap,
            )
            if not admission.accepted:
                window = int(cfg.rate_limit_window_seconds)
                reason = f"rate limit: @{submitter} has used {admission.used}/{cap} submissions in the last {window}s"
                log.info(
                    "rate_limited",
                    extra={
                        "event": x_github_event,
                        "delivery": x_github_delivery,
                        "login": submitter,
                        "association": decision.association,
                        "used": admission.used,
                        "cap": cap,
                    },
                )
                db.record_event(
                    delivery_id=x_github_delivery,
                    event_type=x_github_event,
                    repo=decision.repo,
                    issue_key=decision.issue_key,
                    payload=payload,
                    state="skipped",
                    last_error=reason,
                )
                return JSONResponse(
                    {"delivery": x_github_delivery, "state": "skipped", "reason": "rate_limited"},
                    status_code=202,
                )

        inserted = db.record_event(
            delivery_id=x_github_delivery,
            event_type=x_github_event,
            repo=decision.repo,
            issue_key=decision.issue_key,
            payload=payload,
            state="queued",
        )
        if inserted:
            pool: WorkerPool = bag["pool"]
            pool.wake()
            log.info(
                "queued", extra={"event": x_github_event, "delivery": x_github_delivery, "key": decision.issue_key}
            )
        else:
            log.info("duplicate", extra={"event": x_github_event, "delivery": x_github_delivery})
        return JSONResponse({"delivery": x_github_delivery, "state": "queued"}, status_code=202)

    @app.post("/replay")
    async def replay(
        request: Request,
        x_robomp_token: str | None = Header(None, alias="X-Robomp-Replay-Token"),
        delivery_id: str = "",
    ) -> JSONResponse:
        bag = request.app.state.bag
        cfg: Settings = bag["settings"]
        if cfg.replay_token is None:
            raise HTTPException(404, "replay disabled")
        if x_robomp_token != cfg.replay_token.get_secret_value():
            raise HTTPException(401, "invalid replay token")
        db: Database = bag["db"]
        row = db.get_event(delivery_id)
        if row is None:
            raise HTTPException(404, "unknown delivery")
        if not db.requeue_event(delivery_id, from_states=INACTIVE_EVENT_STATES):
            raise HTTPException(409, f"delivery {delivery_id} is {row.state}; only inactive events can be replayed")
        bag["pool"].wake()
        return JSONResponse({"delivery": delivery_id, "state": "queued"})

    def _require_trigger_token(cfg: Settings, token: str | None) -> None:
        if cfg.replay_token is None:
            raise HTTPException(404, "trigger disabled (set ROBOMP_REPLAY_TOKEN to enable)")
        if token != cfg.replay_token.get_secret_value():
            raise HTTPException(401, "invalid replay token")

    @app.get("/api/github/issues")
    async def api_github_issues(
        request: Request,
        state: str = "open",
        limit: int = 30,
        x_robomp_token: str | None = Header(None, alias="X-Robomp-Replay-Token"),
    ) -> dict[str, Any]:
        """Browse issues across `ROBOMP_REPO_ALLOWLIST` for the trigger picker.

        Token-gated identically to `/api/trigger`: this hits the live GitHub API
        with the bot's PAT and would otherwise leak titles from private repos.
        """
        bag = request.app.state.bag
        cfg: Settings = bag["settings"]
        _require_trigger_token(cfg, x_robomp_token)

        if state not in ("open", "closed", "all"):
            raise HTTPException(400, "state must be open|closed|all")
        capped = max(1, min(int(limit), 100))
        github: GitHubClient = bag["github"]
        repos = sorted(cfg.repo_allowlist)
        if not repos:
            return {"issues": [], "errors": [], "repos": []}

        # Fan out across allowlisted repos; per-repo failures don't take down the panel.
        async def _one(repo: str) -> tuple[str, list, str | None]:
            try:
                items = await github.list_issues(repo, state=state, limit=capped)
                return repo, items, None
            except Exception as exc:  # GitHubError, network, etc.
                log.warning("list_issues failed", extra={"repo": repo, "err": str(exc)})
                return repo, [], str(exc)

        results = await asyncio.gather(*(_one(r) for r in repos))
        merged = []
        errors = []
        for repo, items, err in results:
            if err is not None:
                errors.append({"repo": repo, "error": err})
            merged.extend(items)
        # Newest-updated first across all repos.
        merged.sort(key=lambda s: s.updated_at, reverse=True)
        merged = merged[:capped]
        return {
            "issues": [
                {
                    "repo": s.repo,
                    "number": s.number,
                    "title": s.title,
                    "state": s.state,
                    "author": s.author,
                    "labels": list(s.labels),
                    "comments": s.comments,
                    "updated_at": s.updated_at,
                    "created_at": s.created_at,
                    "html_url": s.html_url,
                }
                for s in merged
            ],
            "errors": errors,
            "repos": repos,
        }

    @app.post("/api/trigger")
    async def api_trigger(
        request: Request,
        payload: dict[str, Any] = Body(...),
        x_robomp_token: str | None = Header(None, alias="X-Robomp-Replay-Token"),
    ) -> JSONResponse:
        """Manually queue an issue. Modes:

        - `triage`: fetch fresh from GitHub and enqueue (or re-enqueue) as if `issues.opened`.
        - `retry`:  requeue an existing stored event. Identify it by `delivery_id` or `issue`.
        """
        bag = request.app.state.bag
        cfg: Settings = bag["settings"]
        _require_trigger_token(cfg, x_robomp_token)

        db: Database = bag["db"]
        github: GitHubClient = bag["github"]
        pool: WorkerPool = bag["pool"]

        mode = str(payload.get("mode") or "").strip().lower()
        if mode not in ("triage", "retry"):
            raise HTTPException(400, "mode must be 'triage' or 'retry'")

        issue_ref = payload.get("issue")
        delivery_id = payload.get("delivery_id")

        if mode == "triage":
            if not isinstance(issue_ref, str) or not issue_ref:
                raise HTTPException(400, "triage requires 'issue' = 'owner/repo#NN'")
            try:
                repo_full, number = parse_issue_ref(issue_ref)
            except InvalidIssueRef as exc:
                raise HTTPException(400, str(exc)) from exc
            if not cfg.allows(repo_full):
                raise HTTPException(403, f"{repo_full} not in ROBOMP_REPO_ALLOWLIST")
            try:
                delivery = await enqueue_manual_triage(
                    db=db,
                    github=github,
                    repo_full=repo_full,
                    number=number,
                )
            except ManualTriageConflict as exc:
                raise HTTPException(409, str(exc)) from exc
            except ManualTriageError as exc:
                raise HTTPException(400, str(exc)) from exc
            except GitHubError as exc:
                raise HTTPException(502, f"github error: {exc.status} {exc.message}") from exc
            pool.wake()
            log.info("manual triage", extra={"delivery": delivery, "issue": f"{repo_full}#{number}"})
            return JSONResponse(
                {"delivery": delivery, "state": "queued", "mode": "triage"},
                status_code=202,
            )

        # mode == "retry"
        if isinstance(delivery_id, str) and delivery_id:
            target = delivery_id
        elif isinstance(issue_ref, str) and issue_ref:
            try:
                repo_full, number = parse_issue_ref(issue_ref)
            except InvalidIssueRef as exc:
                raise HTTPException(400, str(exc)) from exc
            if not cfg.allows(repo_full):
                raise HTTPException(403, f"{repo_full} not in ROBOMP_REPO_ALLOWLIST")
            row = db.latest_event_for_issue(make_issue_key(repo_full, number))
            if row is None:
                raise HTTPException(404, f"no stored event for {repo_full}#{number}")
            target = row.delivery_id
        else:
            raise HTTPException(400, "retry requires 'delivery_id' or 'issue'")

        event = db.get_event(target)
        if event is None:
            raise HTTPException(404, f"unknown delivery {target}")
        if not db.requeue_event(target, from_states=INACTIVE_EVENT_STATES):
            raise HTTPException(409, f"delivery {target} is {event.state}; only inactive events can be retried")
        pool.wake()
        log.info("manual retry", extra={"delivery": target})
        return JSONResponse(
            {"delivery": target, "state": "queued", "mode": "retry"},
            status_code=202,
        )

    @app.get("/events")
    async def events(request: Request, limit: int = 50) -> dict[str, Any]:
        rows = request.app.state.bag["db"].list_events(limit=limit)
        return {
            "events": [
                {
                    "delivery_id": r.delivery_id,
                    "event_type": r.event_type,
                    "repo": r.repo,
                    "issue_key": r.issue_key,
                    "state": r.state,
                    "attempts": r.attempts,
                    "received_at": r.received_at,
                    "last_error": r.last_error,
                }
                for r in rows
            ]
        }

    @app.get("/issues")
    async def issues(request: Request, limit: int = 100) -> dict[str, Any]:
        rows = request.app.state.bag["db"].list_issues(limit=limit)
        return {
            "issues": [
                {
                    "key": r.key,
                    "repo": r.repo,
                    "number": r.number,
                    "branch": r.branch,
                    "pr_number": r.pr_number,
                    "state": r.state,
                    "classification": r.classification,
                    "updated_at": r.updated_at,
                }
                for r in rows
            ]
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/status")
    async def api_status(request: Request) -> dict[str, Any]:
        bag = request.app.state.bag
        cfg: Settings = bag["settings"]
        db: Database = bag["db"]
        pool: WorkerPool = bag["pool"]
        started = float(bag.get("started_at") or time.time())
        issues_rows = db.list_issues(limit=200)
        events_rows = db.list_events(limit=25)
        return {
            "runtime": {
                "bot_login": cfg.bot_login,
                "repo_allowlist": sorted(cfg.repo_allowlist),
                "max_concurrency": cfg.max_concurrency,
                "model": cfg.model,
                "thinking_level": cfg.thinking_level,
                "uptime_seconds": max(0.0, time.time() - started),
            },
            "event_counts": db.event_state_counts(),
            "running_events": db.list_running_events(),
            "inflight": await pool.inflight_snapshot(),
            "issues": [
                {
                    "key": r.key,
                    "repo": r.repo,
                    "number": r.number,
                    "branch": r.branch,
                    "pr_number": r.pr_number,
                    "state": r.state,
                    "classification": r.classification,
                    "updated_at": r.updated_at,
                }
                for r in issues_rows
            ],
            "recent_events": [
                {
                    "delivery_id": r.delivery_id,
                    "event_type": r.event_type,
                    "repo": r.repo,
                    "issue_key": r.issue_key,
                    "state": r.state,
                    "attempts": r.attempts,
                    "received_at": r.received_at,
                    "last_error": r.last_error,
                }
                for r in events_rows
            ],
        }

    @app.get("/api/logs")
    async def api_logs(request: Request, limit: int = 400) -> dict[str, Any]:
        cfg: Settings = request.app.state.bag["settings"]
        capped = max(1, min(int(limit), 2000))
        entries = tail_jsonl(cfg.log_dir / "robomp.log.jsonl", limit=capped)
        return {"entries": entries, "count": len(entries), "limit": capped}

    return app


__all__ = ["create_app"]
