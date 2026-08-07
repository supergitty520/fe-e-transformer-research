#!/usr/bin/env python3
"""Audit the local FE-E tree before a private GitHub push without printing secrets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent.parent
REQUIRED = (
    "README.md",
    "README_EN.md",
    "REPRODUCIBILITY.md",
    "REPRODUCIBILITY_EN.md",
    "DATA_AND_ARTIFACTS.md",
    "DATA_AND_ARTIFACTS_EN.md",
    "LICENSE_POLICY.md",
    "LICENSE_POLICY_EN.md",
    "LICENSE",
    "LICENSE_SCOPE.md",
    "LICENSES/CC-BY-4.0.txt",
    "NOTICE",
    "SECURITY.md",
    "SECURITY_EN.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING_EN.md",
    "CITATION.cff",
    "docs/bilingual_publication_map.md",
    "docs/research_process.md",
    "docs/en/research_process.md",
    "docs/methodology.md",
    "docs/en/methodology.md",
    "docs/research_status.md",
    "docs/en/research_status.md",
    "docs/experiment_registry.md",
    "docs/en/experiment_registry.md",
    "docs/en/private_github_release_checklist.md",
    "output/attachments/README.md",
)
EXCLUDED_PARTS = {
    ".git",
    ".secrets",
    "__pycache__",
    "node_modules",
    ".next",
    ".vinext",
    ".open-next",
    ".wrangler",
    "dist",
    "tmp",
    "publication",
    "local_research_archive",
}
EXCLUDED_FILES = {
    "PROJECT.md",
    "scripts/setup_windows_fee_ssh.ps1",
    "scripts/setup_windows_fee_admin_ssh.ps1",
    "scripts/revoke_windows_fee_admin.ps1",
}
FORBIDDEN_PUBLICATION_PREFIXES = (
    "results/mlx_d24_smoke/",
    "results/mlx_eval5_smoke/",
    "results/mlx_gate_v2_smoke/",
    "results/mlx_gate_v5_matched_control_smoke/",
    "results/mlx_smoke_adamw_fee_sham/",
    "results/mlx_d128_fee_dose_smoke/",
    "results/mlx_d24_calibration/",
    "results/mlx_d24_pilot/",
    "results/mlx_d24_pilot_calibrated/",
    "results/mlx_d24_stress_pilot/",
    "results/mlx_d24_stress_pilot_v2/",
    "results/mlx_d24_stress_pilot_v3/",
    "results/mlx_d96_benchmark/",
    "results/mlx_d192_benchmark/",
    "results/mlx_d1024_benchmark/",
    "results/mlx_d128_gate_v4_baseline_calibration/",
    "results/mlx_d128_gate_v6_mid_calibration/",
    "results/mlx_d128_s1_acc99_persistent_gate_v2_hybrid_first/",
    "results/mlx_d128_s1_acc99_persistent_gate_v4_adaptive_hybrid_first/",
    "results/mlx_d128_s1_acc99_persistent_gate_v5_matched_hybrid_first/",
    "results/mlx_d128_s1_step10000_eval125_threecase/",
    "results/mlx_d128_s1_step10000_eval125_fourcase/",
    "results/mlx_d128_s1_step10000_eval125_threecase_reuse_adamw/",
    "results/mlx_d192_formal_clean/",
    "results/mlx_d192_s47_acc99_gs_sham_vs_gsf/",
)
FORBIDDEN_PUBLICATION_FILES = {
    "results/smoke.json",
    "results/gradient_smoothing_smoke.json",
    "results/standard_depth24.json",
    "results/ablation_depth24.json",
    "results/direct_compare_pilot_d24_seed7.json",
    "output/figures/fee_d192_validation_loss_curves.png",
    "output/figures/fee_d192_validation_loss_curves.svg",
    "scripts/plot_mlx_d192_curves.py",
}
SECRET_PATTERNS = (
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def candidates() -> list[Path]:
    found = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = relative(path)
        if rel in EXCLUDED_FILES or any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        if rel.startswith("output/attachments/") and rel.endswith((".zip", ".sha256")):
            continue
        found.append(path)
    return found


def git_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    for item in REQUIRED:
        if not (ROOT / item).is_file():
            failures.append(f"missing required document: {item}")

    if (ROOT / ".git").is_dir():
        for item in sorted(
            EXCLUDED_FILES
            | {
                ".secrets/example",
                "tmp/example",
                "publication/example",
                "local_research_archive/example",
                "website/node_modules/example",
                "output/attachments/example.zip",
                "output/attachments/example.zip.sha256",
            }
        ):
            if not git_ignored(item):
                failures.append(f"expected ignored path is not ignored: {item}")
    else:
        warnings.append("git repository has not been initialized yet")

    files = candidates()
    candidate_names = {relative(path) for path in files}
    for item in sorted(candidate_names & FORBIDDEN_PUBLICATION_FILES):
        failures.append(f"development artifact returned to publication candidate set: {item}")
    for prefix in FORBIDDEN_PUBLICATION_PREFIXES:
        if any(item.startswith(prefix) for item in candidate_names):
            failures.append(f"development artifact directory returned to candidate set: {prefix}")
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        if size > 50 * 1024 * 1024:
            failures.append(f"candidate file exceeds 50 MiB: {relative(path)}")
        if size > 20 * 1024 * 1024:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            failures.append(f"could not read candidate file: {relative(path)}")
            continue
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            failures.append(f"credential-like content detected: {relative(path)}")

    for path in (item for item in files if item.suffix.lower() == ".md"):
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for raw_target in MARKDOWN_LINK.findall(source):
            target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            clean = unquote(target.split("#", 1)[0])
            if clean and not (path.parent / clean).resolve().exists():
                failures.append(f"broken local link in {relative(path)}: {target}")

    portable_roots = (ROOT / "src", ROOT / "tests", ROOT / "docs", ROOT / "paper")
    absolute_path_hits = 0
    for base in portable_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if "/Users/" in source or re.search(r"[A-Za-z]:\\Users\\", source):
                absolute_path_hits += 1
                warnings.append(f"machine-specific path remains in portable material: {relative(path)}")

    print(f"GitHub readiness: {len(files)} candidate files, {total_bytes / 1024 / 1024:.1f} MiB")
    print(f"Machine-path warnings: {absolute_path_hits}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    print("OK: private-repository candidate set passed readiness checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
