"""Validate a plan or audit captured CPU evidence; never launch work."""
import argparse
import json
import os
from pathlib import Path

from .protocol import digest, read_json, schedule


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check schema and print the fixed paired schedule")
    check.add_argument("--protocol", required=True, type=Path)
    old = commands.add_parser("replay-legacy", help="Recompute the already published 24 CPU pairs")
    old.add_argument("--evidence-dir", required=True, type=Path)
    old.add_argument("--output-dir", required=True, type=Path)
    audit = commands.add_parser("audit-scheduling", help="Recompute scheduling quality from pinned native evidence")
    audit.add_argument("--capture", required=True, type=Path)
    audit.add_argument("--capture-sha256", required=True)
    audit.add_argument("--scope", required=True, type=Path)
    audit.add_argument("--scope-sha256", required=True)
    audit.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        plan = read_json(args.protocol.read_text())
        output = {"protocol_sha256": digest(plan), "schedule": schedule(plan),
                  "scope": "Schema only; source availability, preregistration and execution authorization not established"}
    elif args.command == "audit-scheduling":
        from .scheduling import audit_scheduling, read_capture
        try:
            record = read_capture(args.capture, args.capture_sha256)
            scope = read_capture(args.scope, args.scope_sha256)
            result = audit_scheduling(record, **scope)
            result['capture_sha256'] = args.capture_sha256
            result['scope_sha256'] = args.scope_sha256
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n');stream.flush();os.fsync(stream.fileno())
        except (ValueError, OSError, TypeError) as exc:
            parser.exit(1, 'scheduling audit unavailable: ' + type(exc).__name__ + '\n')
        output = {'output':str(args.output.resolve()), 'quality':result['measurement']['quality'],
                  'campaign_identity_verified':False, 'new_research_evidence':False}
    else:
        from .legacy import replay
        if args.output_dir.exists():
            raise ValueError("output directory already exists")
        plan, output = replay(args.evidence_dir)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, value in (("protocol.json", plan), ("report.json", output)):
            with (args.output_dir / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        output = {"output_dir": str(args.output_dir.resolve()), "counts": output["counts"],
                  "historical_summary_exact": output["historical_summary_exact"],
                  "new_research_evidence": False}
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
