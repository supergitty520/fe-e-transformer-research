# Local Preparation Report for a Private GitHub Repository

[中文](PRIVATE_REPO_PREP_REPORT.md) | **English**

Date: 2026-08-07  
Status: **Local Git initialization and the pre-publication audit are complete. Remote creation and
upload remain pending GitHub security verification.**

## Recommendation

The project is suitable for limited private collaboration, not immediate public release. A private
repository can preserve the full research path, negative results, active stops, isomorphic controls, and
raw logs for review by a small group. It must not imply that FE-E is a mature optimizer and does not
replace patent or confidentiality arrangements.

## Candidate set

- 452 candidate files;
- about 186.2 MiB;
- largest candidate file is about 9 MiB;
- about 185 MiB of formal and key active-stop JSONL can be versioned without Git LFS;
- no candidate file exceeds 50 MiB;
- no private key or common cloud-token pattern was detected;
- portable documentation has no residual absolute user path;
- local Markdown links pass validation.

## Included

- FE-E and Gradient Smoothing implementations;
- PyTorch and MLX unit tests;
- experiment, analysis, and plotting scripts;
- raw JSONL, manifests, run summaries, and structured audits;
- formal, exploratory, invalid, and actively stopped reports;
- English preprint, blind manuscript, Chinese essay, figures, and Release-attachment policy;
- Apache-2.0 and CC BY 4.0 texts, bilingual scope, NOTICE, and citation metadata;
- paired Chinese/English release entries and research-governance documents;
- interactive explainer source without dependencies.

## Explicitly excluded

- `.secrets/`;
- Windows SSH, administrator-creation, and revocation scripts;
- blog deployment copies and Cloudflare infrastructure;
- conversation and local-workspace metadata in `PROJECT.md`;
- `tmp/`, Python caches, `node_modules`, and website build output;
- editor, operating-system, and runtime temporary files;
- 202 smoke, pilot, calibration, superseded, protocol-invalid, or non-conclusive intermediate artifacts
  totaling about 26.3 MiB;
- repository-duplicate ZIP/checksum bundles and a superseded single-seed figure.

Nothing was deleted. These artifacts are preserved under ignored `local_research_archive/`. The main
candidate set retains only two incomplete logs with independent research value: the 1024-layer frozen
extrapolation and the actively stopped 128-layer 5000-step comparison.

## Completed validation

- PyTorch/Gradient Smoothing: 15 tests passed.
- Apple MLX configuration, VJP, gating, sham, and dose scheduling: 18 tests passed.
- Interactive website: production build and one server-render test passed.
- Private-repository readiness audit: passed.
- The isolated system Python lacks PyTorch and Metal access; equivalent validation was completed in the
  local PyTorch Docker environment and the host environment with Metal access.

## Actions not taken

- no GitHub repository;
- no remote;
- no commit, push, Release, or collaborator invitation;
- no second legal review of the active layered licenses for a future public release.

## Decisions reserved for the author

1. private-repository owner and name;
2. whether to keep about 185 MiB of formal raw logs in Git history or use a private Release;
3. initial collaborators and permissions;
4. patent, submission, or NDA constraints;
5. whether the active layered licenses need professional legal review before a public release;
6. whether every detailed Chinese experiment report also requires a full English translation.

The author has explicitly authorized a private GitHub repository. Remote creation will proceed after
the account security verification completes.
