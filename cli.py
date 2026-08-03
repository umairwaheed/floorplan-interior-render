#!/usr/bin/env python
"""Command-line interface.

Runs the identical pipeline the API runs, through the same service layer. That
is the point of it: if the CLI can produce renders and a bill of materials with
no web server involved, the architecture demonstrably doesn't depend on the UI.

    python cli.py plan floorplan.pdf --page 2
    python cli.py design <floorplan-id> --style scandinavian --views 2
    python cli.py run floorplan.pdf --style japandi          # both, in one go
    python cli.py catalog --category sofa --style industrial
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from backend.catalog.service import get_catalog_service
from backend.config import get_settings
from backend.design.styles import list_style_profiles
from backend.ingest.service import FloorPlanIngestService, IngestionError
from backend.schemas.product import DesignStyle, ProductCategory, ProductQuery
from backend.schemas.render import DesignRequest
from backend.services.pipeline import ProgressEvent
from backend.services.store import FLOORPLAN_STORE, JOB_STORE


def _log(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)-7s %(message)s",
    )


# --- commands --------------------------------------------------------------


def cmd_plan(args) -> int:
    """Extract and calibrate a floor plan."""
    service = FloorPlanIngestService(get_settings())
    try:
        floorplan, working = service.ingest(
            Path(args.path),
            page=args.page - 1,
            manual_px_per_m=args.px_per_m,
            ceiling_height_m=args.ceiling,
        )
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    FLOORPLAN_STORE.add(floorplan, Path(args.path), working)
    cal = floorplan.calibration

    print(f"floor plan   {floorplan.id}")
    print(f"scale        {cal.px_per_m:.2f} px/m via {cal.method}")
    print(f"             residual {cal.residual_pct}%, confidence {cal.confidence:.0%}")
    print(f"total area   {floorplan.total_area_m2:.1f} m²")
    print()
    print(f"{'room':<22}{'type':<12}{'measured':>10}{'printed':>10}")
    for room in floorplan.rooms:
        printed = f"{room.area_label_m2:.1f}" if room.area_label_m2 else "—"
        print(f"{room.name[:21]:<22}{room.room_type.value:<12}{room.area_m2:>9.1f}{printed:>10}")

    # Warnings are the point of the calibration self-check; never swallow them.
    for warning in cal.warnings:
        print(f"\n  ! {warning}")

    if args.json:
        print(json.dumps(floorplan.model_dump(mode="json"), indent=2))
    return 0


def cmd_design(args, floorplan_id: str | None = None) -> int:
    """Design and render an already-extracted plan."""
    floorplan_id = floorplan_id or args.floorplan_id
    stored = FLOORPLAN_STORE.get(floorplan_id)
    if stored is None:
        print(f"error: unknown floor plan {floorplan_id}", file=sys.stderr)
        print(
            "hint: `plan` and `design` are separate processes; use `run` to do both.",
            file=sys.stderr,
        )
        return 1

    request = DesignRequest(
        floorplan_id=floorplan_id,
        style=DesignStyle(args.style),
        palette_name=args.palette,
        views_per_room=args.views,
        variations=args.variations,
        budget_max=args.budget,
        seed=args.seed,
    )
    job = JOB_STORE.create(request)

    last_stage = {"value": ""}

    def on_progress(event: ProgressEvent) -> None:
        if event.render is not None:
            render = event.render
            score = (
                f"{render.scores.overall:.2f}"
                if render.scores and render.scores.verified
                else "unverified"
            )
            flag = " [anchor]" if render.is_anchor else ""
            print(f"  {render.camera_id:<18} {render.status.value:<10} {score}{flag}")
        elif event.detail != last_stage["value"]:
            last_stage["value"] = event.detail
            print(f"[{event.progress:5.0%}] {event.detail}")

    print(f"job {job.id} — {request.style.value}, {request.views_per_room} view(s)/room\n")
    JOB_STORE.run_sync(job, stored.floorplan, on_progress=on_progress)

    if job.error:
        print(f"\nerror: {job.error}", file=sys.stderr)
        return 1

    summary = JOB_STORE.pipeline.summary(job)
    print("\nconsistency")
    if summary.get("verified"):
        print(f"  mean  {summary['mean_consistency']}   worst {summary['worst_consistency']}")
        layout = summary["mean_layout_fidelity"]
        identity = summary["mean_object_identity"]
        print(f"  layout {layout}   identity {identity}")
        if summary.get("missing_objects"):
            print(f"  ! missing from some view: {', '.join(summary['missing_objects'])}")
        if summary.get("hallucinated_objects"):
            print(
                f"  ! rendered but not in the scene: {', '.join(summary['hallucinated_objects'])}"
            )
    else:
        print(f"  {summary.get('note')}")

    for variation in job.variations:
        print(f"\nvariation {variation.variation_index} — scene {variation.scene_id}")
        for render in variation.renders:
            if render.image_path:
                print(f"  {render.image_path}")
        if variation.bom:
            print(f"\n  {'product':<44}{'qty':>5}{'total':>11}")
            for line in variation.bom.lines[:15]:
                print(f"  {line.name[:43]:<44}{line.quantity:>5.0f}{line.line_total:>11,.0f}")
            more = variation.bom.item_count - min(15, variation.bom.item_count)
            if more:
                print(f"  … and {more} more line(s)")
            print(
                f"  {'TOTAL':<44}{'':>5}{variation.bom.total_cost:>11,.0f} {variation.bom.currency}"
            )
    return 0


def cmd_run(args) -> int:
    """Extract, then design, in one process."""
    if cmd_plan(args) != 0:
        return 1
    plans = FLOORPLAN_STORE.list()
    if not plans:
        return 1
    print()
    return cmd_design(args, floorplan_id=plans[0].id)


def cmd_catalog(args) -> int:
    """Search the product catalog."""
    matches = get_catalog_service().search(
        ProductQuery(
            text=args.text,
            categories=[ProductCategory(c) for c in (args.category or [])],
            styles=[DesignStyle(s) for s in (args.style or [])],
            colors=args.color or [],
            materials=args.material or [],
            max_width_m=args.max_width,
            max_price=args.max_price,
            limit=args.limit,
        )
    )
    if not matches:
        print("no products matched")
        return 0

    print(f"{'score':>6}  {'product':<46}{'size (mm)':>18}{'price':>10}")
    for match in matches:
        product = match.product
        dims = product.dimensions
        size = f"{dims.width_mm:.0f}×{dims.depth_mm:.0f}×{dims.height_mm:.0f}"
        print(f"{match.score:>6.3f}  {product.name[:45]:<46}{size:>18}{product.price:>10,.0f}")
    return 0


def cmd_styles(_args) -> int:
    for profile in list_style_profiles():
        palettes = ", ".join(p.name for p in profile.palettes)
        print(f"{profile.style.value:<15}{profile.label:<24}{palettes}")
    return 0


# --- argument parsing ------------------------------------------------------


def _add_design_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--style", default="scandinavian", choices=[s.value for s in DesignStyle])
    parser.add_argument("--palette", help="Palette name; defaults to the style's first")
    parser.add_argument("--views", type=int, default=2, help="Viewpoints per room")
    parser.add_argument("--variations", type=int, default=1)
    parser.add_argument("--budget", type=float, help="Rough budget cap, in catalog currency")
    parser.add_argument("--seed", type=int, default=0)


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="PNG, JPG or PDF floor plan")
    parser.add_argument("--page", type=int, default=1, help="PDF page (1-based)")
    parser.add_argument("--ceiling", type=float, help="Ceiling height in metres")
    parser.add_argument(
        "--px-per-m",
        type=float,
        dest="px_per_m",
        help="Override the drawing scale. Needed only when the plan prints "
        "neither m² labels nor dimensions.",
    )
    parser.add_argument("--json", action="store_true", help="Also dump the extracted geometry")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Floor plan → photorealistic interior renders."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Extract and calibrate a floor plan")
    _add_plan_args(plan)
    plan.set_defaults(func=cmd_plan)

    design = subparsers.add_parser("design", help="Design and render an extracted plan")
    design.add_argument("floorplan_id")
    _add_design_args(design)
    design.set_defaults(func=cmd_design)

    run = subparsers.add_parser("run", help="Extract, design and render in one go")
    _add_plan_args(run)
    _add_design_args(run)
    run.set_defaults(func=cmd_run)

    catalog = subparsers.add_parser("catalog", help="Search the product catalog")
    catalog.add_argument("--text")
    catalog.add_argument("--category", action="append", choices=[c.value for c in ProductCategory])
    catalog.add_argument("--style", action="append", choices=[s.value for s in DesignStyle])
    catalog.add_argument("--color", action="append")
    catalog.add_argument("--material", action="append")
    catalog.add_argument("--max-width", type=float, dest="max_width")
    catalog.add_argument("--max-price", type=float, dest="max_price")
    catalog.add_argument("--limit", type=int, default=15)
    catalog.set_defaults(func=cmd_catalog)

    styles = subparsers.add_parser("styles", help="List styles and palettes")
    styles.set_defaults(func=cmd_styles)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    _log(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
