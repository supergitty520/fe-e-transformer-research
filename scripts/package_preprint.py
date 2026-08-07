#!/usr/bin/env python3
"""Create the FE-E reproducibility archive and embed it in the preprint PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]


def selected_files() -> list[Path]:
    explicit = [
        ROOT / "README.md",
        ROOT / "PROJECT.md",
        ROOT / "pyproject.toml",
        ROOT / "CITATION.cff",
        ROOT / "docs" / "research_report.md",
        ROOT / "paper" / "fe_e_preprint.md",
        ROOT / "paper" / "CODE_ATTACHMENT_README.md",
    ]
    trees = [ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "results"]
    files = list(explicit)
    for tree in trees:
        files.extend(
            path
            for path in tree.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".DS_Store"}
        )
    return sorted(set(files), key=lambda path: str(path.relative_to(ROOT)))


def build_zip(output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = selected_files()
    digests: list[tuple[str, str]] = []
    total_bytes = 0
    for path in files:
        data = path.read_bytes()
        relative = str(path.relative_to(ROOT))
        digests.append((hashlib.sha256(data).hexdigest(), relative))
        total_bytes += len(data)
    metadata = {
        "title": "FE-E: Finite-Element and Entropy Control of Adjoint Propagation in Deep Transformers",
        "author": "XUEZHENG WANG",
        "version": "0.1",
        "release_date": "2026-08-04",
        "file_count": len(files),
        "uncompressed_bytes": total_bytes,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr(
            "MANIFEST-SHA256.txt",
            "".join(f"{digest}  {relative}\n" for digest, relative in digests),
        )
        archive.writestr("ATTACHMENT_METADATA.json", json.dumps(metadata, indent=2) + "\n")
    metadata["archive_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata["archive_bytes"] = output.stat().st_size
    return metadata


def embed_attachment(source_pdf: Path, attachment: Path, output_pdf: Path) -> None:
    reader = PdfReader(source_pdf)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.add_attachment(attachment.name, attachment.read_bytes())
    writer.add_metadata(
        {
            "/Title": "FE-E: Finite-Element and Entropy Control of Adjoint Propagation in Deep Transformers",
            "/Author": "XUEZHENG WANG",
            "/Subject": "FE-E preprint with embedded code and results attachment",
            "/Keywords": "Transformer, finite element, entropy, gradient propagation, adjoint",
        }
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as stream:
        writer.write(stream)


def verify(pdf: Path, attachment: Path) -> None:
    reader = PdfReader(pdf)
    attached = reader.attachments
    if attachment.name not in attached:
        raise RuntimeError("attachment missing from final PDF")
    copies = attached[attachment.name]
    if len(copies) != 1 or copies[0] != attachment.read_bytes():
        raise RuntimeError("embedded attachment does not match release archive")
    if len(reader.pages) < 1:
        raise RuntimeError("final PDF has no pages")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()
    metadata = build_zip(args.archive)
    embed_attachment(args.source_pdf, args.archive, args.output_pdf)
    verify(args.output_pdf, args.archive)
    print(json.dumps(metadata, indent=2))
    print(args.output_pdf)


if __name__ == "__main__":
    main()

