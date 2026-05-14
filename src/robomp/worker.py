"""Per-task RpcClient driver.

The orchestrator calls `run_task(...)` from within an asyncio loop. The
function spins up `RpcClient` on a worker thread, drives the kickoff/follow-up
prompt, and returns when the agent emits `agent_end`.

Host tools call back into the orchestrator's GitHub client and DB. Because the
RpcClient runs in its own subprocess and the host-tool callbacks are dispatched
on the RpcClient's stdout-reader thread, the callbacks block until coroutines
scheduled onto the parent loop complete (`asyncio.run_coroutine_threadsafe`).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from omp_rpc import (
    MessageUpdateEvent,
    RpcClient,
    RpcError,
    ToolExecutionEndEvent,
)

from robomp import host_tools, persona
from robomp.config import Settings
from robomp.db import Database, issue_key
from robomp.github_client import CommentInfo, GitHubClient, IssueInfo, RepoInfo
from robomp.host_tools import ToolBindings
from robomp.sandbox import Workspace

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskInputs:
    """Common context shared by every task type."""

    settings: Settings
    db: Database
    github: GitHubClient
    repo: RepoInfo
    issue: IssueInfo
    workspace: Workspace


@dataclass(slots=True, frozen=True)
class ThreadMessage:
    """One entry in the conversation a directive carries to the agent."""
    kind: str          # issue_body | pr_body | comment | review_comment | review
    author: str
    body: str
    created_at: str
    path: str | None = None     # review_comment only
    line: int | None = None     # review_comment only
    state: str | None = None    # review only (APPROVED / CHANGES_REQUESTED / COMMENTED)


@dataclass(slots=True, frozen=True)
class DirectiveInfo:
    """A maintainer's `@bot` mention captured as an authoritative instruction.

    `thread` is the full conversation context (issue/PR body + every prior
    comment + every review) up to the moment the directive fired.
    """

    body: str
    author: str
    thread: tuple[ThreadMessage, ...] = ()


def _build_extra_env(settings: Settings) -> dict[str, str]:
    """Pass the GitHub token to subprocesses that need it (git push uses the credentialed remote)."""
    env: dict[str, str] = {}
    return env


def _build_prompt(
    task_kind: str,
    inputs: TaskInputs,
    *,
    comment: CommentInfo | None,
    pr_number: int | None,
    review_payload: dict[str, Any] | None,
    directive: DirectiveInfo | None = None,
) -> str:
    if task_kind == "triage_issue":
        if directive is not None:
            return persona.kickoff_directive(
                repo=inputs.repo,
                issue=inputs.issue,
                workspace=inputs.workspace,
                directive=directive,
            )
        return persona.kickoff(repo=inputs.repo, issue=inputs.issue, workspace=inputs.workspace)
    if task_kind == "handle_comment":
        assert comment is not None
        issue_row = inputs.db.get_issue(issue_key(inputs.repo.full_name, inputs.issue.number))
        if issue_row is None:
            pr_status = "no PR opened yet"
        elif issue_row.pr_number is None:
            pr_status = "no PR opened yet"
        elif issue_row.state == "merged":
            pr_status = f"PR #{issue_row.pr_number} was merged"
        elif issue_row.state in ("closed", "abandoned"):
            pr_status = f"PR #{issue_row.pr_number} was closed without merge"
        else:
            pr_status = f"PR #{issue_row.pr_number} is open"
        if directive is not None:
            return persona.directive(
                repo=inputs.repo,
                issue=inputs.issue,
                workspace=inputs.workspace,
                comment=comment,
                directive=directive,
                pr_status=pr_status,
            )
        return persona.followup_comment(
            repo=inputs.repo,
            issue=inputs.issue,
            workspace=inputs.workspace,
            comment=comment,
            pr_status=pr_status,
        )
    if task_kind == "handle_review":
        assert review_payload is not None
        path = str(review_payload.get("path") or "")
        start = review_payload.get("start_line") or review_payload.get("line")
        end = review_payload.get("line") or review_payload.get("original_line")
        if isinstance(start, int) and isinstance(end, int) and start != end:
            line_range = f":L{start}-L{end}"
        elif isinstance(end, int):
            line_range = f":L{end}"
        else:
            line_range = ""
        body = str(review_payload.get("body") or "")
        author = str(review_payload.get("author") or "")
        return persona.followup_review(
            repo=inputs.repo,
            workspace=inputs.workspace,
            pr_number=int(pr_number or 0),
            comment_author=author,
            comment_body=body,
            comment_path=path,
            comment_line_range=line_range,
        )
    raise ValueError(f"unknown task kind: {task_kind!r}")


def _run_rpc_blocking(
    inputs: TaskInputs,
    *,
    task_kind: str,
    prompt: str,
    loop: asyncio.AbstractEventLoop,
    bindings: ToolBindings,
) -> str | None:
    """Run a full RPC turn synchronously. Returns final assistant text (or None)."""
    settings = inputs.settings

    def _on_tool_end(event: ToolExecutionEndEvent) -> None:
        try:
            tool_name = event.tool.get("name") if isinstance(event.tool, dict) else getattr(event.tool, "name", None)
        except Exception:
            tool_name = None
        log.info(
            "tool_end",
            extra={
                "issue": bindings.issue_key,
                "tool": tool_name,
                "ok": event.result is not None,
            },
        )

    def _on_msg(event: MessageUpdateEvent) -> None:
        ev = event.assistant_message_event
        if isinstance(ev, dict) and ev.get("type") == "text_delta":
            log.debug("delta", extra={"issue": bindings.issue_key, "delta": str(ev.get("delta", ""))[:200]})

    rpc_env = _build_extra_env(settings)
    chosen_model = settings.pick_model()
    log.info(
        "rpc_model_pick",
        extra={
            "issue": bindings.issue_key,
            "model": chosen_model,
            "pool": list(settings.model_pool),
        },
    )

    with RpcClient(
        executable=settings.omp_command,
        cwd=bindings.workspace.repo_dir,
        session_dir=bindings.workspace.session_dir,
        env=rpc_env,
        no_session=False,
        no_title=True,
        model=chosen_model,
        provider=settings.provider,
        thinking=settings.thinking_level if settings.thinking_level != "off" else None,
        append_system_prompt=persona.system_append(repo=inputs.repo, issue=inputs.issue, workspace=inputs.workspace),
        custom_tools=host_tools.build(bindings),
        request_timeout=settings.request_timeout_seconds,
        startup_timeout=60.0,
        max_event_history=50_000,
    ) as client:
        client.install_headless_ui()
        client.on_tool_execution_end(_on_tool_end)
        client.on_message_update(_on_msg)

        phases = persona.seed_phases(task_kind)
        if phases:
            try:
                if task_kind == "triage_issue":
                    # Fresh session: seed the full plan.
                    client.set_todos(phases)
                else:
                    # Resumed session: keep prior phases (e.g. Reproduce / Fix / PR)
                    # so the agent still sees the context, but append the
                    # follow-up phase at the end.
                    existing = list(client.get_todos())
                    merged = [
                        {
                            "id": p.id,
                            "name": p.name,
                            "tasks": [
                                {
                                    "id": t.id,
                                    "content": t.content,
                                    "status": t.status,
                                    "notes": t.notes,
                                    "details": t.details,
                                }
                                for t in p.tasks
                            ],
                        }
                        for p in existing
                    ] + phases
                    client.set_todos(merged)
            except RpcError as exc:
                log.warning("set_todos failed", extra={"err": str(exc)})

        log.info(
            "rpc_start",
            extra={"issue": bindings.issue_key, "task": task_kind, "branch": bindings.workspace.branch},
        )
        turn = client.prompt_and_wait(prompt, timeout=settings.task_timeout_seconds)
        log.info(
            "rpc_done",
            extra={
                "issue": bindings.issue_key,
                "task": task_kind,
                "messages": len(turn.messages),
                "events": len(turn.events),
            },
        )
        return turn.assistant_text


async def run_task(
    *,
    task_kind: str,
    inputs: TaskInputs,
    comment: CommentInfo | None = None,
    pr_number: int | None = None,
    review_payload: dict[str, Any] | None = None,
    directive: DirectiveInfo | None = None,
) -> str | None:
    """Async wrapper that runs the synchronous RPC driver on a worker thread."""
    loop = asyncio.get_running_loop()
    bindings = ToolBindings(
        db=inputs.db,
        github=inputs.github,
        repo=inputs.repo,
        issue=inputs.issue,
        workspace=inputs.workspace,
        loop=loop,
        author_name=inputs.settings.resolved_author_name,
        author_email=inputs.settings.git_author_email,
    )
    prompt = _build_prompt(
        task_kind, inputs, comment=comment, pr_number=pr_number, review_payload=review_payload, directive=directive
    )
    return await asyncio.to_thread(
        _run_rpc_blocking,
        inputs,
        task_kind=task_kind,
        prompt=prompt,
        loop=loop,
        bindings=bindings,
    )


__all__ = ["DirectiveInfo", "TaskInputs", "ThreadMessage", "run_task"]
