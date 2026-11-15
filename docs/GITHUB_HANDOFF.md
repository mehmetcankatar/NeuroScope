# Local repository handoff

No GitHub repository has been created or changed. This folder contains the complete local source project. The ZIP delivery excludes generated data, installed dependencies, weights, and local Git metadata.

## Create your remote when ready

Create an empty repository named `NeuroScope` in your own GitHub account. Do not initialize it with another README if using these files.

From this directory, inspect what will be included:

```bash
git status --short
git add README.md LICENSE THIRD_PARTY_NOTICES.md pyproject.toml .gitignore .env.example backend frontend scripts configs tests evaluation docs .github Dockerfile .dockerignore
git diff --cached --stat
```

If starting from the ZIP, first run `git init -b main`. If your Git configuration is missing an identity, configure your own name and email before committing.

Check that no EEG files, model weights, `.env`, `data/`, `models/`, `vendor/`, or `node_modules/` are staged. Then:

```bash
git commit -m "Build NeuroScope EEG review prototype"
git remote add origin https://github.com/YOUR_USERNAME/NeuroScope.git
git push -u origin main
```

Those commands are instructions for you; they were not executed against GitHub during delivery. Replace `YOUR_USERNAME` with your account.

## Submission checklist

- Explain that MORGOTH is an existing pretrained model; credit its authors.
- State which results came from actual checkpoints and which screens show synthetic data.
- Include only dataset material you are allowed to share.
- Report measured results and unresolved model validation accurately.
- Disclose Codex and other tools according to the competition rules.
- Keep a synthetic demo usable even when reviewers lack checkpoint credentials, but do not describe it as real model performance.

Current competition-specific eligibility and deadline rules are not asserted by this repository; verify them before submitting.
