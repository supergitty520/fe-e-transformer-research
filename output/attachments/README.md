# Release attachments

ZIP bundles and adjacent SHA-256 files are generated release artifacts, not source-of-truth research
files. They are excluded from the main Git history because they duplicate code, logs, figures, and
reports already present in the repository.

Generate them locally with:

```bash
python scripts/build_preprint.py
python scripts/package_preprint.py
```

When a remote release is explicitly authorized, upload the generated bundles to that release and verify
their checksums there. The repository itself retains this policy file only.
