# Contributing to FE-E Research

[中文](CONTRIBUTING.md) | **English**

This repository is a private research preview. Contributions should improve falsifiability and
reproducibility, not merely search for positive results.

## Before proposing an experiment

An issue must specify:

- the hypothesis and the result that would falsify it;
- method, baseline, and isomorphic sham;
- frozen seeds, task, model, and validation batches;
- primary and secondary endpoints, safety ceiling, and stop conditions;
- expected compute, memory, wall-clock time, and storage;
- results that would terminate the research direction.

Use development seeds for tuning and untouched seeds for confirmation. Do not change the primary endpoint
after seeing results.

## Submitting results

A pull request must include:

- configuration or manifest;
- raw per-step logs or their controlled archive location;
- an analysis script that reconstructs results from raw evidence;
- positive, negative, failed, and actively stopped outcomes;
- fixed-update and wall-clock cost;
- pre-intervention trajectory matching and system-load notes.

Do not manually edit raw JSONL. Correct analysis errors with a new script or derived artifact.

## Code and claim standards

- Python 3.10+;
- tests for every new algorithm;
- no detached gradient represented as a trainable regularizer;
- no large-model claim derived from a small synthetic task;
- every table in documentation traceable to a result file or analysis script;
- claim changes mirrored in the paired Chinese and English status documents listed in the
  [bilingual publication map](docs/bilingual_publication_map.md).

## Security

Do not commit secrets, SSH material, personal data, administrator scripts, blog deployment credentials,
or restricted third-party data. Follow [Security](SECURITY_EN.md) and never paste secret values into an
issue.

