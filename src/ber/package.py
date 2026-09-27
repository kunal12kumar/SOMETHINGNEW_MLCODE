"""Build the final submission zip in the required layout.

<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src, tests, notebooks, models, README.md, requirements.txt}
└── Documentation_template.md

Usage:
    python -m ber.package --repo REPO --outputs DRIVE/output_v6 --model DRIVE/model_v6 \
        --doc REPO/docs/Documentation_template.md --team TeamName --out DRIVE/TeamName_submission.zip
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

CODE_ITEMS = ["src", "tests", "notebooks", "README.md", "requirements.txt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, required=True, help="folder with the two submission TSVs")
    ap.add_argument("--model", type=Path, required=True, help="trained model folder to ship as models/<name>")
    ap.add_argument("--model-name", default="v6")
    ap.add_argument("--extra-model", type=Path, nargs="*", default=[], help="more model folders, shipped under their own names")
    ap.add_argument("--doc", type=Path, required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / f"{args.team}_submission"
        (root / "output").mkdir(parents=True)
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy2(args.outputs / f, root / "output" / f)
        code = root / "code" / "business_entity_resolution"
        code.mkdir(parents=True)
        for item in CODE_ITEMS:
            src = args.repo / item
            if src.is_dir():
                shutil.copytree(src, code / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, code / item)
        shutil.copytree(args.model, code / "models" / args.model_name)
        for extra in args.extra_model:
            shutil.copytree(extra, code / "models" / extra.name)
        shutil.copy2(args.doc, root / "Documentation_template.md")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root))  # output/, code/, Documentation at the zip root
    with zipfile.ZipFile(args.out) as z:
        names = z.namelist()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB, {len(names)} files)")
    for n in names:
        if n.count("/") <= 2:
            print("  " + n)


if __name__ == "__main__":
    main()
