"""Report whether a handwriting profile exists and what it covers.

Usage: status.py [--profile NAME] [--require]

--require exits with code 2 and a "train first" message when there is no usable
profile; generation runs this before doing anything else.
"""

import argparse
import sys

from common import profile_dir, read_json, read_settings, samples_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--require", action="store_true")
    args = ap.parse_args()

    pdir = profile_dir(args.profile)
    profile = read_json(pdir / "profile.json")
    if not profile or not profile.get("glyphs"):
        pending = sorted(p.name for p in samples_dir(args.profile).glob("*") if p.is_dir())
        print(f'NO PROFILE: there is no handwriting profile "{args.profile}" yet.')
        if pending:
            print(f"  Training was started ({len(pending)} pages added) but not finished; run /handwritten train again.")
        else:
            print("  Run /handwritten train <photos or folder> with 5-8 photos of your normal handwriting first.")
        sys.exit(2 if args.require else 0)

    print((pdir / "coverage.txt").read_text(encoding="utf-8").rstrip())
    settings = read_settings(args.profile)
    if settings.get("line_pitch_mm"):
        print(f"  settings: line spacing {settings['line_pitch_mm']} mm (set by you, used instead of the measured value)")
    ident = read_json(pdir / "identity" / "identity.json")
    if ident:
        print(f"  identity block: {' / '.join(ident['lines'])} (written on the first page with --identity)")
    print(f"  updated: {profile['updated']}")
    print(f"  preview: {pdir / 'preview.png'}")
    for p in sorted(pdir.glob("specimen_*.png")):
        print(f"  review:  {p}")


if __name__ == "__main__":
    main()
