# Agent guidelines

## Git attribution

- Never add AI attribution to commits. This includes `Co-Authored-By: Claude ...`,
  any `@anthropic.com` address, and "Generated with ..." trailers.
- Every commit must have only the repository owner as author and committer, so
  all changes are credited to their GitHub account.
- The `.githooks/commit-msg` hook strips such trailers automatically. It is
  enabled with `git config core.hooksPath .githooks`; do not bypass it with
  `--no-verify` and do not re-add removed attribution.
