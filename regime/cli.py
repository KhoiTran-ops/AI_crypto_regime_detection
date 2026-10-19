"""Pipeline stages are explicit; published results are independent of the dashboard."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Binance crypto regime research")
    parser.add_argument("command", choices=["data", "select", "adequacy", "evaluate", "intervals",
                                            "publish", "verify", "reproduce", "external-data", "extend", "verify-extensions", "all"])
    parser.add_argument("--refresh", action="store_true", help="Download external data only in a new snapshot checkout")
    args = parser.parse_args()
    if args.command == "external-data":
        from .external import build_external
        from .supply import add_supply
        add_supply(build_external(refresh=args.refresh))
    if args.command == "extend":
        from .extensions import run_extensions
        run_extensions()
        from .supply import run_supply
        run_supply()
        from .reproduce import seal_extensions
        seal_extensions()
    if args.command == "verify-extensions":
        from .reproduce import verify_extensions
        verify_extensions()
    if args.command in ("data", "all"):
        from .data import build_spot, download_spot
        from .derivatives import funding, open_interest
        _, failures = download_spot()
        if failures:
            raise RuntimeError(f"{len(failures)} spot archive downloads failed")
        build_spot()
        funding()
        audits = open_interest()
        if any(item["coverage"] < .995 for item in audits):
            raise ValueError("OI coverage below the declared 99.5% gate")
    if args.command in ("select", "all"):
        import json

        from .data import ROOT
        from .research import input_fingerprint
        checkpoint = ROOT / "outputs/research/benchmark_selection.json"
        if checkpoint.exists():
            prior = json.loads(checkpoint.read_text(encoding="utf-8"))
            if prior["input_fingerprint"] != input_fingerprint():
                raise ValueError("Inputs changed: preserve the published run and use a separate checkout; "
                                 "do not reuse validation caches from different data/config.")
        from .research import benchmark_selection, cohort_selection, select
        select()
        benchmark_selection()
        cohort_selection()
    if args.command in ("adequacy", "all"):
        from .adequacy import adequacy
        adequacy()
    if args.command in ("evaluate", "all"):
        from .final import evaluate
        evaluate()
    if args.command in ("intervals", "all"):
        from .uncertainty import parameter_intervals
        parameter_intervals()
    if args.command in ("publish", "all"):
        from .publish import publish
        publish()
    if args.command == "verify":
        from .reproduce import verify
        verify()
    if args.command == "reproduce":
        from .reproduce import reproduce
        reproduce()


if __name__ == "__main__":
    main()
