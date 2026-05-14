"""Host tool tests against a mocked GitHub via httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from omp_rpc import HostToolContext, RpcCommandError

from robomp.db import Database
from robomp.github_client import GitHubClient, IssueInfo, RepoInfo
from robomp.host_tools import ToolBindings, build
from robomp.sandbox import Workspace


def _stub_workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    repo_dir = root / "repo"
    session_dir = root / ".omp-session"
    context_dir = root / "context"
    artifacts_dir = root / "artifacts"
    for p in (root, repo_dir, session_dir, context_dir, context_dir / "repro", artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)
    return Workspace(
        root=root,
        repo_dir=repo_dir,
        session_dir=session_dir,
        context_dir=context_dir,
        artifacts_dir=artifacts_dir,
        branch="farm/abc12345/some-issue",
        repo_full_name="octo/widget",
        issue_number=42,
    )


def _stub_issue() -> IssueInfo:
    return IssueInfo(
        repo="octo/widget",
        number=42,
        title="boom",
        body="b",
        state="open",
        author="alice",
        labels=("bug",),
        is_pull_request=False,
    )


def _stub_repo() -> RepoInfo:
    return RepoInfo(
        full_name="octo/widget",
        default_branch="main",
        clone_url="https://x/octo/widget.git",
        private=False,
    )


def _make_loop_in_background() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop: asyncio.AbstractEventLoop, t: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)
    loop.close()


def _bindings(
    db: Database, tmp_path: Path, transport: httpx.MockTransport
) -> tuple[ToolBindings, asyncio.AbstractEventLoop, threading.Thread]:
    github = GitHubClient("token", transport=transport)
    loop, thread = _make_loop_in_background()
    bindings = ToolBindings(
        db=db,
        github=github,
        repo=_stub_repo(),
        issue=_stub_issue(),
        workspace=_stub_workspace(tmp_path),
        loop=loop,
        author_name="robomp-bot",
        author_email="robomp-bot@example.invalid",
    )
    db.upsert_issue(
        key=bindings.issue_key,
        repo="octo/widget",
        number=42,
        state="reproducing",
        branch=bindings.workspace.branch,
        session_dir=str(bindings.workspace.session_dir),
    )
    return bindings, loop, thread


def _ctx() -> HostToolContext[Any]:
    return HostToolContext(tool_call_id="tc-1", _cancel_event=threading.Event(), _send_update=lambda _payload: None)


def test_gh_post_comment_happy_path(db: Database, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(201, json={"id": 999, "user": {"login": "robomp-bot"}, "body": "hi", "created_at": "t"})

    transport = httpx.MockTransport(handler)
    bindings, loop, t = _bindings(db, tmp_path, transport)
    try:
        tool = next(x for x in build(bindings) if x.name == "gh_post_comment")
        result = tool.execute({"body": "hi"}, _ctx())
    finally:
        _stop_loop(loop, t)

    assert result.startswith("comment posted")
    assert captured["url"].endswith("/repos/octo/widget/issues/42/comments")
    assert captured["body"] == {"body": "hi"}
    assert captured["auth"] == "Bearer token"


def test_gh_post_comment_validates_body(db: Database, tmp_path: Path) -> None:
    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        tool = next(x for x in build(bindings) if x.name == "gh_post_comment")
        with pytest.raises(RpcCommandError):
            tool.execute({"body": ""}, _ctx())
    finally:
        _stop_loop(loop, t)


def test_gh_post_comment_propagates_github_error(db: Database, tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(422, json={"message": "Validation failed"}))
    bindings, loop, t = _bindings(db, tmp_path, transport)
    try:
        tool = next(x for x in build(bindings) if x.name == "gh_post_comment")
        with pytest.raises(RpcCommandError) as exc:
            tool.execute({"body": "hi"}, _ctx())
        assert "422" in str(exc.value)
    finally:
        _stop_loop(loop, t)


def test_gh_open_pr_requires_template_sections(db: Database, tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    bindings, loop, t = _bindings(db, tmp_path, transport)
    try:
        tool = next(x for x in build(bindings) if x.name == "gh_open_pr")
        with pytest.raises(RpcCommandError) as exc:
            tool.execute({"title": "t", "body": "no sections"}, _ctx())
        assert "Repro" in str(exc.value)
    finally:
        _stop_loop(loop, t)


def test_repro_record_writes_transcript(db: Database, tmp_path: Path) -> None:
    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        tool = next(x for x in build(bindings) if x.name == "repro_record")
        result = tool.execute(
            {
                "title": "panic on empty input",
                "command": "bun test foo.test.ts",
                "output": "Error: boom",
                "exit_code": 1,
                "reproduced": True,
            },
            _ctx(),
        )
        assert "saved transcript" in result
        files = list(bindings.workspace.repro_dir.iterdir())
        assert len(files) == 1
        assert "exit_code: 1" in files[0].read_text()
    finally:
        _stop_loop(loop, t)


def test_repro_record_rejects_bad_args(db: Database, tmp_path: Path) -> None:
    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        tool = next(x for x in build(bindings) if x.name == "repro_record")
        with pytest.raises(RpcCommandError):
            tool.execute({"title": "", "command": "x", "output": "y", "exit_code": 1}, _ctx())
        with pytest.raises(RpcCommandError):
            tool.execute({"title": "t", "command": "x", "output": "y", "exit_code": "bad"}, _ctx())
    finally:
        _stop_loop(loop, t)


def test_mark_unable_posts_comment_and_abandons(db: Database, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": 77, "user": {"login": "robomp-bot"}, "body": "x", "created_at": "t"})

    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(handler))
    try:
        tool = next(x for x in build(bindings) if x.name == "mark_unable_to_reproduce")
        result = tool.execute({"diagnosis": "needed exact version", "info_needed": "post bun --version"}, _ctx())
    finally:
        _stop_loop(loop, t)
    assert "abandonment" in result
    assert "Could not reproduce" in captured["body"]["body"]
    issue = db.get_issue(bindings.issue_key)
    assert issue and issue.state == "abandoned"


def test_fetch_issue_thread_returns_markdown(db: Database, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/comments"):
            return httpx.Response(
                200,
                json=[
                    {"id": 1, "user": {"login": "alice"}, "body": "still broken", "created_at": "t1"},
                ],
            )
        return httpx.Response(
            200,
            json={
                "number": 42,
                "title": "boom",
                "body": "b",
                "state": "open",
                "user": {"login": "alice"},
                "labels": [{"name": "bug"}],
            },
        )

    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(handler))
    try:
        tool = next(x for x in build(bindings) if x.name == "fetch_issue_thread")
        result = tool.execute({}, _ctx())
    finally:
        _stop_loop(loop, t)
    assert "octo/widget#42" in result
    assert "@alice" in result
    assert "still broken" in result


def test_classify_issue_applies_labels_and_persists_primary(db: Database, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=[{"name": n} for n in captured["body"]["labels"]],
        )

    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(handler))
    try:
        tool = next(x for x in build(bindings) if x.name == "classify_issue")
        result = tool.execute(
            {
                "primary": "bug",
                "priority": "prio:p1",
                "functional": ["tool", "agent"],
                "provider": "provider:openai",
                "platform": "platform:macos",
                "rationale": "tool call panics on empty arg on macOS",
            },
            _ctx(),
        )
    finally:
        _stop_loop(loop, t)

    assert "classified as bug" in result
    assert "reproduce" in result.lower()
    assert captured["path"].endswith("/issues/42/labels")
    assert captured["body"]["labels"] == [
        "bug",
        "prio:p1",
        "tool",
        "agent",
        "providers",
        "provider:openai",
        "platform:macos",
        "triaged",
    ]
    row = db.get_issue(bindings.issue_key)
    assert row is not None and row.classification == "bug"


def test_classify_issue_question_skips_repro_path(db: Database, tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=[{"name": "question"}, {"name": "triaged"}]))
    bindings, loop, t = _bindings(db, tmp_path, transport)
    try:
        tool = next(x for x in build(bindings) if x.name == "classify_issue")
        result = tool.execute(
            {"primary": "question", "rationale": "how-to about config"},
            _ctx(),
        )
    finally:
        _stop_loop(loop, t)
    assert "question" in result
    assert "no PR" in result
    row = db.get_issue(bindings.issue_key)
    assert row is not None and row.classification == "question"


def test_classify_issue_rejects_bug_without_priority(db: Database, tmp_path: Path) -> None:
    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        tool = next(x for x in build(bindings) if x.name == "classify_issue")
        with pytest.raises(RpcCommandError):
            tool.execute({"primary": "bug", "rationale": "yes a bug"}, _ctx())
    finally:
        _stop_loop(loop, t)


def test_classify_issue_rejects_priority_on_non_bug(db: Database, tmp_path: Path) -> None:
    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        tool = next(x for x in build(bindings) if x.name == "classify_issue")
        with pytest.raises(RpcCommandError):
            tool.execute(
                {"primary": "question", "priority": "prio:p1", "rationale": "x"},
                _ctx(),
            )
    finally:
        _stop_loop(loop, t)


def test_classify_issue_rejects_unknown_primary(db: Database, tmp_path: Path) -> None:
    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        tool = next(x for x in build(bindings) if x.name == "classify_issue")
        with pytest.raises(RpcCommandError):
            tool.execute({"primary": "nonsense", "rationale": "x"}, _ctx())
    finally:
        _stop_loop(loop, t)


def test_set_issue_labels_appends(db: Database, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=[{"name": n} for n in captured["body"]["labels"]])

    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(handler))
    try:
        tool = next(x for x in build(bindings) if x.name == "set_issue_labels")
        result = tool.execute({"labels": ["wontfix"]}, _ctx())
    finally:
        _stop_loop(loop, t)
    assert "wontfix" in result
    assert captured["body"]["labels"] == ["wontfix"]


def test_set_issue_labels_rejects_empty(db: Database, tmp_path: Path) -> None:
    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        tool = next(x for x in build(bindings) if x.name == "set_issue_labels")
        with pytest.raises(RpcCommandError):
            tool.execute({"labels": []}, _ctx())
        with pytest.raises(RpcCommandError):
            tool.execute({"labels": ["   ", ""]}, _ctx())
    finally:
        _stop_loop(loop, t)


def test_gh_push_branch_rejects_wrong_identity(db: Database, tmp_path: Path) -> None:
    """Pre-push gate refuses to push commits authored by anyone other than the configured identity."""
    import os
    import subprocess

    # Build a real local upstream + worktree so git operations actually work.
    bare = tmp_path / "upstream.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "seed",
        "GIT_AUTHOR_EMAIL": "seed@x",
        "GIT_COMMITTER_NAME": "seed",
        "GIT_COMMITTER_EMAIL": "seed@x",
    }
    subprocess.run(["git", "init", "--initial-branch=main", str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("init\n")
    for cmd in (
        ["git", "-C", str(seed), "add", "."],
        ["git", "-C", str(seed), "-c", "user.email=seed@x", "-c", "user.name=seed", "commit", "-m", "init"],
        ["git", "-C", str(seed), "remote", "add", "origin", str(bare)],
        ["git", "-C", str(seed), "push", "origin", "main"],
    ):
        subprocess.run(cmd, check=True, capture_output=True, env=env)

    from robomp.sandbox import SandboxManager

    mgr = SandboxManager(tmp_path / "workspaces")
    ws = mgr.ensure_workspace(
        repo="octo/widget",
        number=42,
        title="identity test",
        clone_url=str(bare),
        default_branch="main",
        author_name="robomp-bot",
        author_email="robomp-bot@example.invalid",
    )
    # Commit with a different identity to provoke the gate.
    bad_env = os.environ | {
        "GIT_AUTHOR_NAME": "wrong",
        "GIT_AUTHOR_EMAIL": "wrong@nope",
        "GIT_COMMITTER_NAME": "wrong",
        "GIT_COMMITTER_EMAIL": "wrong@nope",
    }
    (ws.repo_dir / "x.txt").write_text("hi\n")
    subprocess.run(["git", "-C", str(ws.repo_dir), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(ws.repo_dir), "-c", "user.email=wrong@nope", "-c", "user.name=wrong", "commit", "-m", "bad"],
        check=True,
        capture_output=True,
        env=bad_env,
    )

    github = GitHubClient("tok", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    loop, thread = _make_loop_in_background()
    try:
        bindings = ToolBindings(
            db=db,
            github=github,
            repo=_stub_repo(),
            issue=IssueInfo(
                repo="octo/widget",
                number=42,
                title="t",
                body="",
                state="open",
                author="alice",
                labels=(),
                is_pull_request=False,
            ),
            workspace=ws,
            loop=loop,
            author_name="robomp-bot",
            author_email="robomp-bot@example.invalid",
        )
        db.upsert_issue(
            key=bindings.issue_key,
            repo="octo/widget",
            number=42,
            state="reproducing",
            branch=ws.branch,
            session_dir=str(ws.session_dir),
        )
        tool = next(x for x in build(bindings) if x.name == "gh_push_branch")
        with pytest.raises(RpcCommandError) as exc:
            tool.execute({}, _ctx())
        msg = str(exc.value)
        assert "identity mismatch" in msg
        assert "wrong <wrong@nope>" in msg
        assert "robomp-bot <robomp-bot@example.invalid>" in msg
        # Branch must NOT have been pushed.
        refs = subprocess.run(
            ["git", "-C", str(bare), "for-each-ref", "--format=%(refname)"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not any(r.startswith("refs/heads/farm/") for r in refs.stdout.splitlines()), refs.stdout
    finally:
        _stop_loop(loop, thread)


def test_gh_open_pr_rejects_wrong_identity_before_push_or_pr(db: Database, tmp_path: Path) -> None:
    """gh_open_pr uses the guarded push path before creating the pull request."""
    import os
    import subprocess

    bare = tmp_path / "upstream.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "seed",
        "GIT_AUTHOR_EMAIL": "seed@x",
        "GIT_COMMITTER_NAME": "seed",
        "GIT_COMMITTER_EMAIL": "seed@x",
    }
    subprocess.run(["git", "init", "--initial-branch=main", str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("init\n")
    for cmd in (
        ["git", "-C", str(seed), "add", "."],
        ["git", "-C", str(seed), "-c", "user.email=seed@x", "-c", "user.name=seed", "commit", "-m", "init"],
        ["git", "-C", str(seed), "remote", "add", "origin", str(bare)],
        ["git", "-C", str(seed), "push", "origin", "main"],
    ):
        subprocess.run(cmd, check=True, capture_output=True, env=env)

    from robomp.sandbox import SandboxManager

    mgr = SandboxManager(tmp_path / "workspaces")
    ws = mgr.ensure_workspace(
        repo="octo/widget",
        number=42,
        title="identity test",
        clone_url=str(bare),
        default_branch="main",
        author_name="robomp-bot",
        author_email="robomp-bot@example.invalid",
    )
    bad_env = os.environ | {
        "GIT_AUTHOR_NAME": "wrong",
        "GIT_AUTHOR_EMAIL": "wrong@nope",
        "GIT_COMMITTER_NAME": "wrong",
        "GIT_COMMITTER_EMAIL": "wrong@nope",
    }
    (ws.repo_dir / "x.txt").write_text("hi\n")
    subprocess.run(["git", "-C", str(ws.repo_dir), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(ws.repo_dir), "-c", "user.email=wrong@nope", "-c", "user.name=wrong", "commit", "-m", "bad"],
        check=True,
        capture_output=True,
        env=bad_env,
    )

    opened_pr = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal opened_pr
        opened_pr = True
        return httpx.Response(
            201,
            json={
                "number": 7,
                "html_url": "https://github.com/octo/widget/pull/7",
                "head": {"ref": ws.branch},
                "base": {"ref": "main"},
            },
        )

    github = GitHubClient("tok", transport=httpx.MockTransport(handler))
    loop, thread = _make_loop_in_background()
    try:
        bindings = ToolBindings(
            db=db,
            github=github,
            repo=_stub_repo(),
            issue=IssueInfo(
                repo="octo/widget",
                number=42,
                title="t",
                body="",
                state="open",
                author="alice",
                labels=(),
                is_pull_request=False,
            ),
            workspace=ws,
            loop=loop,
            author_name="robomp-bot",
            author_email="robomp-bot@example.invalid",
        )
        db.upsert_issue(
            key=bindings.issue_key,
            repo="octo/widget",
            number=42,
            state="reproducing",
            branch=ws.branch,
            session_dir=str(ws.session_dir),
        )
        tool = next(x for x in build(bindings) if x.name == "gh_open_pr")
        body = "## Repro\nrepro\n\n## Cause\ncause\n\n## Fix\nfix\n\n## Verification\nran tests\n\nFixes #42\n"
        with pytest.raises(RpcCommandError) as exc:
            tool.execute({"title": "fix: x", "body": body}, _ctx())
        assert "identity mismatch" in str(exc.value)
        assert not opened_pr
        refs = subprocess.run(
            ["git", "-C", str(bare), "for-each-ref", "--format=%(refname)"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not any(r.startswith("refs/heads/farm/") for r in refs.stdout.splitlines()), refs.stdout
    finally:
        _stop_loop(loop, thread)


def test_gh_push_branch_rejects_invalid_identity_scan_range(db: Database, tmp_path: Path) -> None:
    """A failing git-log author scan is a push rejection, not an empty successful scan."""
    import os
    import subprocess

    bare = tmp_path / "upstream.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "robomp-bot",
        "GIT_AUTHOR_EMAIL": "robomp-bot@example.invalid",
        "GIT_COMMITTER_NAME": "robomp-bot",
        "GIT_COMMITTER_EMAIL": "robomp-bot@example.invalid",
    }
    subprocess.run(["git", "init", "--initial-branch=main", str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("init\n")
    for cmd in (
        ["git", "-C", str(seed), "add", "."],
        [
            "git",
            "-C",
            str(seed),
            "-c",
            "user.email=robomp-bot@example.invalid",
            "-c",
            "user.name=robomp-bot",
            "commit",
            "-m",
            "init",
        ],
        ["git", "-C", str(seed), "remote", "add", "origin", str(bare)],
        ["git", "-C", str(seed), "push", "origin", "main"],
    ):
        subprocess.run(cmd, check=True, capture_output=True, env=env)

    from robomp.sandbox import SandboxManager

    mgr = SandboxManager(tmp_path / "workspaces")
    ws = mgr.ensure_workspace(
        repo="octo/widget",
        number=42,
        title="missing base ref",
        clone_url=str(bare),
        default_branch="main",
        author_name="robomp-bot",
        author_email="robomp-bot@example.invalid",
    )
    subprocess.run(
        ["git", "-C", str(ws.repo_dir), "update-ref", "-d", "refs/remotes/origin/main"], check=True, capture_output=True
    )
    (ws.repo_dir / "x.txt").write_text("hi\n")
    subprocess.run(["git", "-C", str(ws.repo_dir), "add", "x.txt"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(ws.repo_dir),
            "-c",
            "user.email=robomp-bot@example.invalid",
            "-c",
            "user.name=robomp-bot",
            "commit",
            "-m",
            "ok",
        ],
        check=True,
        capture_output=True,
        env=env,
    )

    github = GitHubClient("tok", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    loop, thread = _make_loop_in_background()
    try:
        bindings = ToolBindings(
            db=db,
            github=github,
            repo=_stub_repo(),
            issue=IssueInfo(
                repo="octo/widget",
                number=42,
                title="t",
                body="",
                state="open",
                author="alice",
                labels=(),
                is_pull_request=False,
            ),
            workspace=ws,
            loop=loop,
            author_name="robomp-bot",
            author_email="robomp-bot@example.invalid",
        )
        db.upsert_issue(
            key=bindings.issue_key,
            repo="octo/widget",
            number=42,
            state="reproducing",
            branch=ws.branch,
            session_dir=str(ws.session_dir),
        )
        tool = next(x for x in build(bindings) if x.name == "gh_push_branch")
        with pytest.raises(RpcCommandError) as exc:
            tool.execute({}, _ctx())
        msg = str(exc.value)
        assert "could not inspect commit authors" in msg
        assert "origin/main..HEAD" in msg
        refs = subprocess.run(
            ["git", "-C", str(bare), "for-each-ref", "--format=%(refname)"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not any(r.startswith("refs/heads/farm/") for r in refs.stdout.splitlines()), refs.stdout
        row = db._conn.execute(
            "SELECT error FROM tool_calls WHERE tool='gh_push_branch' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None and "could not inspect commit authors" in row["error"]
        assert "origin/main..HEAD" in row["error"]
    finally:
        _stop_loop(loop, thread)


def test_gh_open_pr_requires_closes_keyword(db: Database, tmp_path: Path) -> None:
    """gh_open_pr refuses if the body has the four sections but no Fixes/Closes/Resolves keyword."""
    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        tool = next(x for x in build(bindings) if x.name == "gh_open_pr")
        body = "## Repro\nrepro\n\n## Cause\ncause\n\n## Fix\nfix\n\n## Verification\nran tests\n"
        with pytest.raises(RpcCommandError) as exc:
            tool.execute({"title": "fix: x", "body": body}, _ctx())
        assert "Fixes #42" in str(exc.value)
    finally:
        _stop_loop(loop, t)


def test_gh_open_pr_refuses_failed_bun_check_before_push_or_pr(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gh_open_pr sends a failing pre-PR check back to the agent without creating a PR."""
    import os

    opened_pr = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal opened_pr
        opened_pr = True
        return httpx.Response(
            201,
            json={
                "number": 7,
                "html_url": "https://github.com/octo/widget/pull/7",
                "head": {"ref": "farm/abc12345/some-issue"},
                "base": {"ref": "main"},
            },
        )

    bindings, loop, t = _bindings(db, tmp_path, httpx.MockTransport(handler))
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    fake_bun = fakebin / "bun"
    fake_bun.write_text(
        "#!/bin/sh\n"
        'if [ "$1" != "check" ]; then printf "wrong command: %s\\n" "$1" >&2; exit 2; fi\n'
        'printf "TypeError: property missing\\n" >&2\n'
        "exit 1\n",
        encoding="utf-8",
    )
    fake_bun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}{os.pathsep}{os.environ['PATH']}")
    (bindings.workspace.repo_dir / "package.json").write_text(
        json.dumps({"scripts": {"check": "tsc --noEmit"}}) + "\n",
        encoding="utf-8",
    )

    try:
        tool = next(x for x in build(bindings) if x.name == "gh_open_pr")
        body = "## Repro\nrepro\n\n## Cause\ncause\n\n## Fix\nfix\n\n## Verification\nran tests\n\nFixes #42\n"
        with pytest.raises(RpcCommandError) as exc:
            tool.execute({"title": "fix: x", "body": body}, _ctx())
    finally:
        _stop_loop(loop, t)

    msg = str(exc.value)
    assert "`bun check` failed before PR creation" in msg
    assert "TypeError: property missing" in msg
    assert "retry `gh_open_pr`" in msg
    assert not opened_pr
    row = db._conn.execute("SELECT error FROM tool_calls WHERE tool='gh_open_pr' ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert "TypeError: property missing" in row["error"]


def test_gh_push_branch_rejects_dirty_worktree(db: Database, tmp_path: Path) -> None:
    """Pre-push gate refuses if the working tree has uncommitted changes."""
    import os
    import subprocess

    # Real upstream + worktree so git status works.
    bare = tmp_path / "upstream.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "robomp-bot",
        "GIT_AUTHOR_EMAIL": "robomp-bot@example.invalid",
        "GIT_COMMITTER_NAME": "robomp-bot",
        "GIT_COMMITTER_EMAIL": "robomp-bot@example.invalid",
    }
    subprocess.run(["git", "init", "--initial-branch=main", str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("init\n")
    for cmd in (
        ["git", "-C", str(seed), "add", "."],
        [
            "git",
            "-C",
            str(seed),
            "-c",
            "user.email=robomp-bot@example.invalid",
            "-c",
            "user.name=robomp-bot",
            "commit",
            "-m",
            "init",
        ],
        ["git", "-C", str(seed), "remote", "add", "origin", str(bare)],
        ["git", "-C", str(seed), "push", "origin", "main"],
    ):
        subprocess.run(cmd, check=True, capture_output=True, env=env)

    from robomp.sandbox import SandboxManager

    mgr = SandboxManager(tmp_path / "workspaces")
    ws = mgr.ensure_workspace(
        repo="octo/widget",
        number=42,
        title="dirty test",
        clone_url=str(bare),
        default_branch="main",
        author_name="robomp-bot",
        author_email="robomp-bot@example.invalid",
    )
    # Make a proper commit (so the identity gate passes).
    (ws.repo_dir / "a.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(ws.repo_dir), "add", "a.txt"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(ws.repo_dir),
            "-c",
            "user.email=robomp-bot@example.invalid",
            "-c",
            "user.name=robomp-bot",
            "commit",
            "-m",
            "ok",
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    # Now dirty the worktree — uncommitted edit.
    (ws.repo_dir / "a.txt").write_text("a-modified\n")

    github = GitHubClient("tok", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    loop, thread = _make_loop_in_background()
    try:
        bindings = ToolBindings(
            db=db,
            github=github,
            repo=_stub_repo(),
            issue=IssueInfo(
                repo="octo/widget",
                number=42,
                title="t",
                body="",
                state="open",
                author="alice",
                labels=(),
                is_pull_request=False,
            ),
            workspace=ws,
            loop=loop,
            author_name="robomp-bot",
            author_email="robomp-bot@example.invalid",
        )
        db.upsert_issue(
            key=bindings.issue_key,
            repo="octo/widget",
            number=42,
            state="reproducing",
            branch=ws.branch,
            session_dir=str(ws.session_dir),
        )
        tool = next(x for x in build(bindings) if x.name == "gh_push_branch")
        with pytest.raises(RpcCommandError) as exc:
            tool.execute({}, _ctx())
        assert "working tree is dirty" in str(exc.value)
        # Nothing pushed.
        refs = subprocess.run(
            ["git", "-C", str(bare), "for-each-ref", "--format=%(refname)"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not any(r.startswith("refs/heads/farm/") for r in refs.stdout.splitlines()), refs.stdout
    finally:
        _stop_loop(loop, thread)


def test_gh_push_branch_does_not_run_repository_bun_scripts(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean push must not execute repository-controlled package scripts during preflight."""
    import os
    import subprocess

    bare = tmp_path / "upstream.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "robomp-bot",
        "GIT_AUTHOR_EMAIL": "robomp-bot@example.invalid",
        "GIT_COMMITTER_NAME": "robomp-bot",
        "GIT_COMMITTER_EMAIL": "robomp-bot@example.invalid",
    }
    subprocess.run(["git", "init", "--initial-branch=main", str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("init\n")
    for cmd in (
        ["git", "-C", str(seed), "add", "."],
        [
            "git",
            "-C",
            str(seed),
            "-c",
            "user.email=robomp-bot@example.invalid",
            "-c",
            "user.name=robomp-bot",
            "commit",
            "-m",
            "init",
        ],
        ["git", "-C", str(seed), "remote", "add", "origin", str(bare)],
        ["git", "-C", str(seed), "push", "origin", "main"],
    ):
        subprocess.run(cmd, check=True, capture_output=True, env=env)

    from robomp.sandbox import SandboxManager

    mgr = SandboxManager(tmp_path / "workspaces")
    ws = mgr.ensure_workspace(
        repo="octo/widget",
        number=42,
        title="no repo scripts during push",
        clone_url=str(bare),
        default_branch="main",
        author_name="robomp-bot",
        author_email="robomp-bot@example.invalid",
    )

    marker = tmp_path / "bun-invoked"
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    fake_bun = fakebin / "bun"
    fake_bun.write_text(f"#!/bin/sh\nprintf invoked > {marker}\nprintf dirty > formatter-output.txt\nexit 0\n")
    fake_bun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}{os.pathsep}{os.environ['PATH']}")

    (ws.repo_dir / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "fix:tools": "printf dirty > formatter-output.txt",
                    "fix": "printf dirty > formatter-output.txt",
                },
            }
        )
        + "\n"
    )
    (ws.repo_dir / "feature.txt").write_text("feature\n")
    subprocess.run(
        ["git", "-C", str(ws.repo_dir), "add", "package.json", "feature.txt"], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(ws.repo_dir),
            "-c",
            "user.email=robomp-bot@example.invalid",
            "-c",
            "user.name=robomp-bot",
            "commit",
            "-m",
            "ok",
        ],
        check=True,
        capture_output=True,
        env=env,
    )

    github = GitHubClient("tok", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    loop, thread = _make_loop_in_background()
    try:
        bindings = ToolBindings(
            db=db,
            github=github,
            repo=_stub_repo(),
            issue=IssueInfo(
                repo="octo/widget",
                number=42,
                title="t",
                body="",
                state="open",
                author="alice",
                labels=(),
                is_pull_request=False,
            ),
            workspace=ws,
            loop=loop,
            author_name="robomp-bot",
            author_email="robomp-bot@example.invalid",
        )
        db.upsert_issue(
            key=bindings.issue_key,
            repo="octo/widget",
            number=42,
            state="reproducing",
            branch=ws.branch,
            session_dir=str(ws.session_dir),
        )
        tool = next(x for x in build(bindings) if x.name == "gh_push_branch")
        result = tool.execute({}, _ctx())
    finally:
        _stop_loop(loop, thread)

    assert result.startswith(f"pushed {ws.branch} ")
    assert not marker.exists()
    assert not (ws.repo_dir / "formatter-output.txt").exists()
    refs = subprocess.run(
        ["git", "-C", str(bare), "for-each-ref", "--format=%(refname)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"refs/heads/{ws.branch}" in refs.stdout.splitlines()
