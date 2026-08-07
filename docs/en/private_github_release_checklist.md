# Private GitHub Release Checklist

[中文](../private_github_release_checklist.md) | **English**

## A. Local preparation

- [ ] Run `python scripts/check_github_readiness.py`.
- [ ] Run PyTorch unit tests; run MLX tests when Apple Silicon is available.
- [ ] Confirm `.secrets/`, SSH scripts, `publication/`, `tmp/`, `node_modules/`, and
      `local_research_archive/` are excluded.
- [ ] Confirm generated `output/attachments/*.zip` and `*.sha256` stay outside the main Git history.
- [ ] Confirm no candidate file exceeds 50 MB; anything above 100 MB must stay out of Git history.
- [ ] Read the latest conclusion at the top of both READMEs; old positive results must not dominate the
      current negative evidence.
- [ ] Check that the signed paper names `XUEZHENG WANG` and the blind manuscript has no author.
- [ ] Confirm active stops and invalid runs remain clearly labeled.
- [ ] Verify every paired file in the [bilingual publication map](../bilingual_publication_map.md).

## B. Creating a private repository

Suggested name: `fe-e-transformer-research`.

When the author later decides to create it:

- choose `Private` visibility;
- do not auto-generate a README, `.gitignore`, or license; the audited layered licenses are included;
- leave GitHub Pages disabled;
- disable forking if organization policy allows;
- give collaborators the minimum necessary role.

Only after creating the remote:

```bash
git remote add origin git@github.com:<owner>/fe-e-transformer-research.git
git push -u origin main
```

Do not replace `<owner>` with a real account inside reusable scripts or documentation.

## C. Suggested first commit

The first commit should contain only the audited research snapshot:

```bash
git add .
git status --short
git diff --cached --stat
git commit -m "research: archive FE-E private preview v0.1"
```

Before committing, confirm no SSH administration script is under `scripts/`; no dependency or build
cache is under `website/`; raw logs were not rewritten; and `output/` contains only referenced figures,
papers, and small archives.

## D. GitHub settings

- [ ] Set `main` as the default branch and require pull requests for collaborator changes.
- [ ] Enable secret scanning and push protection when supported by the private plan.
- [ ] Enable Dependabot security updates but do not automatically merge numerical dependency upgrades.
- [ ] Disable unused Wiki, Projects, and Discussions.
- [ ] Use the description: “private research preview; no general optimizer claim”.
- [ ] Suggested topics: `transformer`, `finite-element`, `gradient-analysis`, `mlx`, `research`.

## E. Collaborator entry path

Ask collaborators to read:

1. [English README](../../README_EN.md) or [中文 README](../../README.md);
2. [current status](research_status.md);
3. [methodology](methodology.md);
4. [reproducibility](../../REPRODUCIBILITY_EN.md);
5. [license policy](../../LICENSE_POLICY_EN.md).

New experiments use the bilingual issue template and preregister hypotheses, seeds, endpoints, compute,
and stop rules. Result pull requests include raw evidence, reconstruction scripts, negative findings, and
pre-intervention matching audits.

## F. Additional threshold before public release

- [ ] Review patent, submission, and confidentiality effects.
- [ ] Sanitize absolute machine paths in manifests.
- [ ] Confirm the existing Apache-2.0 / CC BY 4.0 allocation still matches public-release intent.
- [ ] Audit third-party Gradient Smoothing implementation and citations.
- [ ] Reproduce tests and at least one representative analysis from a fresh clone.
- [ ] Clearly label single-seed, synthetic-task, and unverified 7B-scale limits.
- [ ] Remove internal issues, machine details, and publication infrastructure.
- [ ] Perform a bilingual claim-parity review.

A private release is a research-collaboration checkpoint, not a public paper release or an upgrade of the
performance claims.
