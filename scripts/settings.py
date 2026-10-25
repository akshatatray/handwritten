"""Show or change a profile's settings: values the user chose that override what training measured.
They are kept in settings.json, so rebuilding or retraining the profile keeps them.

Usage: settings.py [--profile NAME]                     show the settings
       settings.py [--profile NAME] line_pitch_mm=9.6   set one
       settings.py [--profile NAME] line_pitch_mm=      remove one (back to the measured value)
"""

import argparse
import sys

from common import SETTINGS, profile_dir, read_json, read_settings, write_json


def describe(name, settings, style):
    for key, (label, unit, _, _) in SETTINGS.items():
        measured = (style.get(key) or {}).get("median")
        measured_txt = f"measured {measured:.1f} {unit}" if measured else "not measured"
        if key in settings:
            print(f"  {label}: {settings[key]} {unit} (set by you; {measured_txt})")
        else:
            print(f"  {label}: {measured_txt} (no setting)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("changes", nargs="*", help="key=value, or key= to remove")
    ap.add_argument("--profile", default="default")
    args = ap.parse_args()

    pdir = profile_dir(args.profile)
    profile = read_json(pdir / "profile.json")
    if not profile:
        sys.exit(f'NO PROFILE: there is no handwriting profile "{args.profile}".')
    settings = read_settings(args.profile)
    for change in args.changes:
        key, _, value = change.partition("=")
        if key not in SETTINGS:
            sys.exit(f"ERROR: unknown setting '{key}' (known: {', '.join(SETTINGS)})")
        label, unit, lo, hi = SETTINGS[key]
        if value == "":
            settings.pop(key, None)
            continue
        number = float(value)
        if not lo <= number <= hi:
            sys.exit(f"ERROR: {label} must be between {lo} and {hi} {unit}")
        settings[key] = number
    if args.changes:
        write_json(pdir / "settings.json", settings)
        from build_profile import write_preview
        write_preview(pdir, profile)
    print(f'Profile "{args.profile}" settings:')
    describe(args.profile, settings, profile["style"])
    print(f"  preview: {pdir / 'preview.png'}")


if __name__ == "__main__":
    main()
