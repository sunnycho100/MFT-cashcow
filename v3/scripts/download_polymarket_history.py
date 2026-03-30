"""CLI downloader for Polymarket event and market price history."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3.src.data.polymarket import (
    PolymarketClient,
    download_event_history,
    download_market_history,
    make_metadata,
    write_history_csv,
    write_metadata_json,
)
from v3.src.utils.config import load_config
from v3.src.utils.logger import get_logger

logger = get_logger("v3.scripts.download_polymarket_history")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--event-slug", help="Polymarket event slug to download")
    scope.add_argument("--market-slug", help="Polymarket market slug to download")
    parser.add_argument(
        "--interval",
        default=None,
        help="Polymarket history interval, for example 1h, 6h, 1d, 1w, max",
    )
    parser.add_argument("--fidelity", type=int, default=None, help="Price history fidelity in minutes")
    parser.add_argument("--start-ts", type=int, default=None, help="Unix timestamp lower bound")
    parser.add_argument("--end-ts", type=int, default=None, help="Unix timestamp upper bound")
    parser.add_argument(
        "--outcomes",
        default="",
        help="Comma-separated outcome labels to keep, for example 'Yes' or 'Yes,No'",
    )
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--metadata-output", default=None, help="Metadata JSON output path")
    return parser


def build_default_output(scope: str, slug: str, output_dir: str) -> Path:
    return Path(output_dir) / f"{scope}_{slug}.csv"


def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    settings = config.get("polymarket", {})

    client = PolymarketClient(
        gamma_base_url=settings.get("gamma_base_url", "https://gamma-api.polymarket.com"),
        clob_base_url=settings.get("clob_base_url", "https://clob.polymarket.com"),
        user_agent=settings.get("user_agent", "MFT-Cashcow-v3/0.1"),
        timeout_sec=settings.get("timeout_sec", 30),
    )

    interval = args.interval or settings.get("default_interval", "1h")
    fidelity = args.fidelity or settings.get("default_fidelity", 60)
    output_dir = settings.get("output_dir", "v3/data/polymarket")
    requested_outcomes = [label.strip() for label in args.outcomes.split(",") if label.strip()]
    allowed_outcomes = {label.casefold() for label in requested_outcomes} or None

    if args.event_slug:
        scope = "event"
        subject, rows = download_event_history(
            client=client,
            event_slug=args.event_slug,
            interval=interval,
            fidelity=fidelity,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            allowed_outcomes=allowed_outcomes,
        )
        slug = args.event_slug
    else:
        scope = "market"
        subject, rows = download_market_history(
            client=client,
            market_slug=args.market_slug,
            interval=interval,
            fidelity=fidelity,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            allowed_outcomes=allowed_outcomes,
        )
        slug = args.market_slug

    output_path = Path(args.output) if args.output else build_default_output(scope, slug, output_dir)
    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".metadata.json")

    write_history_csv(rows, output_path)
    metadata = make_metadata(
        scope=scope,
        interval=interval,
        fidelity=fidelity,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        requested_outcomes=requested_outcomes,
        row_count=len(rows),
        subject=subject,
    )
    write_metadata_json(metadata, metadata_path)

    logger.info("Saved {} rows to {}", len(rows), output_path)
    logger.info("Saved metadata to {}", metadata_path)


if __name__ == "__main__":
    main()
