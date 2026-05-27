# Anonymization Plan (Double-Blind Submission)

This document describes how the repository is sanitized for double-blind review. The plan is split into **content sanitization** (done) and **history sanitization** (deferred to submission time).

## Threat Model

A reviewer who downloads or clones the public artifact must not be able to identify the authors, their institution, or any internal infrastructure from:

- File contents under any tracked path.
- Git commit author names and emails.
- Git commit message footers (`Co-authored-by`, `Signed-off-by`).
- CI workflow definitions or labels.
- Issue / PR metadata if the repo is exposed via a hosting platform.

## A. Content Sanitization (status: complete)

All tracked files have been audited for real names, institutional identifiers, and personal handles. The audit uses the following preflight grep, which must return zero matches before the artifact is shared:

```bash
git grep -nE -i 'nicolas|schuler|kit\.edu|kit[._-]?toolbox|kit\.gemma|@nicol' -- ':!uv.lock' ':!docs/anonymization.md'
git grep -nE 'Co-authored|Signed-off-by' -- ':!uv.lock' ':!docs/anonymization.md'
```

This preflight only scans tracked files. Local manuscript drafts, prompt bundles, and assistant-orchestration scripts are intentionally ignored by `.gitignore`; inspect and anonymize any such file separately before forcing it into a submission branch.

Replacement convention:

- Reviewer identities use the anonymous slot IDs already adopted by `docs/weak_modality_construct_review.csv`: `R1`, `R2`.
- Author / contributor mentions in narrative text are replaced with "the authors" or "we".
- Institutional URLs and emails are removed.
- `.github/workflows/ci.yml` is anonymous by construction and is checked for identity leakage as part of the preflight.

Extend the preflight grep whenever a new identifier becomes relevant.

## B. Git History Sanitization (status: deferred to submission time)

Per project decision, the multi-commit history is preserved during development and rewritten only at the moment of submission. The rewrite produces a new, clean, anonymous history on a submission branch; the original history is retained locally for the authors' own records.

### Planned procedure

1. Install `git-filter-repo` (one-time).
2. From a fresh clone of the repository, create the submission branch:

   ```bash
   git clone . ../re-uq-submission
   cd ../re-uq-submission
   git checkout -b submission/anon
   ```

3. Rewrite author and committer identities:

   ```bash
   git filter-repo --force --commit-callback '
     commit.author_name  = b"Anonymous Authors"
     commit.author_email = b"anonymous@example.invalid"
     commit.committer_name  = b"Anonymous Authors"
     commit.committer_email = b"anonymous@example.invalid"
   '
   ```

4. Strip `Co-authored-by` and `Signed-off-by` trailers from commit messages:

   ```bash
   git filter-repo --force --message-callback '
     import re
     return re.sub(rb"\n(Co-authored-by|Signed-off-by):[^\n]*", b"", message)
   '
   ```

5. Re-run the content preflight grep on the rewritten branch. It must return zero matches.

6. Tag the submission state:

   ```bash
   git tag submission/v1
   ```

7. Push the anonymized branch to the public hosting location used for the artifact (a fresh anonymous account or an artifact-sharing service that strips author metadata). The original repository is **not** pushed to the public location.

### Items to verify on the rewritten branch

- `git log --format='%an %ae' | sort -u` shows only `Anonymous Authors anonymous@example.invalid`.
- `git log --grep='Co-authored-by\|Signed-off-by' --all` returns nothing.
- The content preflight grep returns zero matches.
- `.github/workflows/*.yml` contains no author-bound runner labels, secret names, or environment URLs.
- File timestamps and ordering still reproduce.

## C. Pre-Publish Checklist

Run this immediately before pushing the submission branch:

```bash
# 1. Working tree is clean.
git status --short

# 2. Content sanitization.
git grep -nE -i 'nicolas|schuler|kit\.edu|kit[._-]?toolbox|kit\.gemma|@nicol' -- ':!uv.lock' ':!docs/anonymization.md'
git grep -nE 'Co-authored|Signed-off-by' -- ':!uv.lock' ':!docs/anonymization.md'

# 3. History sanitization (after the filter-repo rewrite).
git log --format='%an %ae' | sort -u

# 4. Tests still pass.
.venv/bin/python -m unittest discover -s tests -v

# 5. Reproduction guide and architecture doc render.
ls docs/reproduction.md docs/architecture.md docs/results_mapping.md
```

If any of (2), (3), or (4) returns non-empty / non-zero, do not publish the artifact.
