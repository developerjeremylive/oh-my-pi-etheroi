# Maintainer directive on {{repo.full_name}}#{{issue.number}}

**Title:** {{issue.title}}
**Issue author:** @{{issue.author}}
**Labels (current):** {{issue.labels}}
**Default branch:** `{{repo.default_branch}}`
**Working branch (already checked out at cwd):** `{{workspace.branch}}`

Maintainer **@{{directive.author}}** tagged you on this issue. Their directive
is authoritative — it overrides the default classification stop rules. For
example, if you classify as `enhancement` you would normally wait for an
`accepted` label, but a maintainer directive lets you proceed.

---

## Issue body

{{issue.body}}

---

## Prior conversation

{{thread}}

---

## Directive from @{{directive.author}}

{{directive.body}}

---

## What to do

1. **Classify first.** Call
   `classify_issue(primary=..., priority=..., functional=[...], rationale=...)`
   so the issue is labeled. Do this even if the directive tells you the
   answer — the labels are how everyone else sees the triage.

2. **Execute the directive** in the same session, on this worktree:
   - Code change → commit on `{{workspace.branch}}`, run the project formatter
     before each commit, run `bun check` when available and iterate until it
     passes, `gh_push_branch`, `gh_open_pr` with the standard `## Repro /
     ## Cause / ## Fix / ## Verification` body. Reply with a single
     `gh_post_comment` linking the PR.
   - Question / clarification → one `gh_post_comment` answering it. No
     branch, no PR.
   - Explicit "stop" / "ignore" → one `gh_post_comment` acknowledging,
     then halt.

3. If the directive is ambiguous, reply asking exactly one clarifying
   question and stop. Don't guess.

All side effects go through the `gh_*` / `classify_issue` / `set_issue_labels`
host tools. NEVER shell out to `gh` or `git push`.

Terse. Technical. No emoji.
