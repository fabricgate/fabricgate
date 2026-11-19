"""fabricgate build command — generate Platform Manifest and Design Index from a HWH or .npq file."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import yaml
from pydantic import TypeAdapter, ValidationError

from fabricgate.cli.exit_codes import exit_with
from fabricgate.cli.formatting import sanitize
from fabricgate.client.errors import FailureKind
from fabricgate.client.publisher import MAX_NPQ_BYTES, PublishError, check_npq
from fabricgate.client.validator import is_single_filename, sha256_hex
from fabricgate.models.board_db import BoardDb
from fabricgate.models.common import DesignName, SemVer
from fabricgate.models.design_index import DesignIndex
from fabricgate.models.platform_manifest import Interface, parse_platform_manifest
from fabricgate.sdk.hwh_parser import HwhInfo, HwhParseError, ParsedInterface, find_hwh_files, parse_hwh

# Artifact file names the manifest stub declares, next to manifest.yaml.
_BITSTREAM_FILE = "design.bit"
_HWH_FILE = "design.hwh"
_MANIFEST_FILE = "manifest.yaml"
# Of the runtimes build offers, only pynq's manifest has an hwh artifact.
_HWH_RUNTIME = "pynq"

_DESIGN_NAME_ADAPTER: TypeAdapter[str] = TypeAdapter(DesignName)
_SEMVER_ADAPTER: TypeAdapter[str] = TypeAdapter(SemVer)

# ---------------------------------------------------------------------------
# Board resolution helpers
# ---------------------------------------------------------------------------


def _stdin_is_tty() -> bool:
    """Whether interactive prompts are possible (patched in tests)."""
    return sys.stdin.isatty()


def _resolve_board_id(
    hwh_info: HwhInfo,
    db: BoardDb,
) -> tuple[str, str | None]:
    """Return ``(board_id, device_family)`` from *hwh_info* and *db*.

    Resolution order:
    1. BOARDPART lookup in Board DB.
    2. Virtual board based on *design_portability*.
    """
    if hwh_info.boardpart:
        entry = db.lookup_by_boardpart(hwh_info.boardpart)
        if entry:
            return entry.board_id, entry.device_family

    portability = hwh_info.design_portability
    df = hwh_info.device_family
    pkg = hwh_info.package

    if portability == "board-specific" and df and pkg:
        return f"custom-{df}-{pkg}", df
    if portability == "board-specific" and df:
        return f"custom-{df}", df
    # axi-only or io-constrained
    if df:
        return f"generic-{df}", df
    return "generic-unknown", None


def _interfaces_to_manifest(parsed: list[ParsedInterface]) -> list[Interface]:
    """Convert parsed HWH interfaces to manifest :class:`Interface` objects."""
    result: list[Interface] = []
    for p in parsed:
        iface_type = "axi-lite"
        if p.protocol:
            proto = p.protocol.lower()
            if "stream" in proto:
                iface_type = "axi-stream"
            elif "lite" in proto:
                iface_type = "axi-lite"
            elif "axi4" in proto:
                iface_type = "axi"
        elif "stream" in p.vlnv.lower() or "axis_rtl" in p.vlnv.lower():
            iface_type = "axi-stream"
        result.append(
            Interface(
                name=p.name,
                type=iface_type,
                base=p.base_address,
            )
        )
    return result


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


def _make_manifest_stub(
    *,
    runtime: str,
    board: str,
    device_family: str | None,
    design_ref: str,
    interfaces: list[Interface],
    design_portability: str | None,
    bitstream_file: str = _BITSTREAM_FILE,
    npq_file: str | None = None,
) -> str:
    """Return a YAML string for a platform manifest stub.

    Artifact hashes are placeholder values; ``fabricgate push`` replaces them
    with the hashes of the files it uploads.
    """
    placeholder = "a" * 64
    artifacts = {"bitstream": {"file": bitstream_file, "sha256": placeholder}}
    if runtime == _HWH_RUNTIME:
        artifacts["hwh"] = {"file": _HWH_FILE, "sha256": placeholder}
    if npq_file is not None:
        artifacts["metadata"] = {"file": npq_file, "sha256": placeholder}
    data: dict[str, object] = {
        "schema": "fabricgate-platform/v1",
        "runtime": runtime,
        "board": board,
        "design_ref": design_ref,
        "artifacts": artifacts,
    }
    if device_family:
        data["device_family"] = device_family
    if design_portability:
        data["design_portability"] = design_portability
    if interfaces:
        data["interfaces"] = [
            {k: v for k, v in {"name": i.name, "type": i.type, "base": i.base}.items() if v is not None}
            for i in interfaces
        ]
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _make_index_stub(
    *,
    design_name: str,
    version: str,
    summary: str | None,
    tags: list[str],
    platform_id: str,
) -> str:
    """Return a YAML string for a design index stub."""
    placeholder = "sha256:" + "a" * 64
    data: dict[str, object] = {
        "schema": "fabricgate-index/v1",
        "name": design_name,
        "version": version,
        "platforms": [{"platform": platform_id, "digest": placeholder}],
    }
    if summary:
        data["summary"] = summary
    if tags:
        data["tags"] = tags
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _render_stubs(inputs: _BuildInputs) -> tuple[str, str]:
    """Return ``(index_yaml, manifest_yaml)`` for *inputs*."""
    manifest_yaml = _make_manifest_stub(
        runtime=inputs.runtime,
        board=inputs.board_id,
        device_family=inputs.device_family,
        design_ref=f"{inputs.design_name}:{inputs.version}",
        interfaces=_interfaces_to_manifest(inputs.hwh_info.interfaces),
        design_portability=inputs.hwh_info.design_portability,
    )
    index_yaml = _make_index_stub(
        design_name=inputs.design_name,
        version=inputs.version,
        summary=inputs.description,
        tags=inputs.tags,
        platform_id=inputs.platform_id,
    )
    return index_yaml, manifest_yaml


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command(name="build")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--namespace", default=None, help="Registry namespace (interactive if omitted).")
@click.option("--version", "semver", default=None, help="SemVer version (interactive if omitted).")
@click.option("--tags", default=None, help="Comma-separated tags (e.g. dma,axi,zynq).")
@click.option("--dry-run", is_flag=True, help="Preview output without writing files.")
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory (default: <design_slug>/ in current directory).",
)
@click.option(
    "--hwh",
    "hwh_path_opt",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Explicit path to .hwh file (required when multiple .hwh files are found).",
)
@click.option(
    "--npq",
    "npq_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Build a nanopynq platform from this .npq file and the bitstream it names (no HWH is read).",
)
def build_cmd(
    directory: Path,
    namespace: str | None,
    semver: str | None,
    tags: str | None,
    dry_run: bool,
    output_dir: Path | None,
    hwh_path_opt: Path | None,
    npq_path: Path | None,
) -> None:
    """Generate Platform Manifest and Design Index from a Vivado HWH file or a nanopynq .npq file."""
    if npq_path is not None:
        if hwh_path_opt is not None:
            raise click.UsageError("--npq and --hwh cannot be used together")
        _build_nanopynq(npq_path, namespace=namespace, semver=semver, tags=tags, output_dir=output_dir, dry_run=dry_run)
        return
    inputs = _collect_inputs(directory, hwh_path_opt, namespace=namespace, semver=semver, tags=tags)
    index_yaml, manifest_yaml = _render_stubs(inputs)
    _check_pushable(index_yaml, manifest_yaml)
    copies = {_HWH_FILE: inputs.hwh_path} if inputs.runtime == _HWH_RUNTIME else {}
    _write_outputs(
        inputs.design_name,
        inputs.platform_id,
        index_yaml,
        manifest_yaml,
        copies=copies,
        missing_bitstream=_BITSTREAM_FILE,
        output_dir=output_dir,
        dry_run=dry_run,
    )


def _build_nanopynq(
    npq_path: Path,
    *,
    namespace: str | None,
    semver: str | None,
    tags: str | None,
    output_dir: Path | None,
    dry_run: bool,
) -> None:
    """Build a nanopynq platform: board and bitstream come from the .npq, which sits next to the bitstream."""
    board, bitstream_file = _read_npq_target(npq_path)
    bitstream_path = npq_path.parent / bitstream_file
    if not bitstream_path.is_file():
        click.echo(f"Error: {npq_path.name} names bitstream {bitstream_file}, not found in {npq_path.parent}", err=True)
        exit_with(FailureKind.INVALID)
    try:
        check_npq(
            npq_path.read_bytes(),
            name=npq_path.name,
            board=board,
            bitstream_file=bitstream_file,
            bitstream_sha256=sha256_hex(bitstream_path),
        )
    except PublishError as exc:
        click.echo(f"Error: {exc}", err=True)
        exit_with(FailureKind.INVALID)
    try:
        entry = BoardDb.load().lookup(board)
    except Exception as exc:
        click.echo(f"Error: Board DB load failed: {exc}", err=True)
        exit_with(FailureKind.INFRA)
    click.echo(f"  board: {sanitize(board)}, bitstream: {bitstream_file}")

    click.echo("")
    design_name, version, runtime, description, tag_list = _prompt_design(namespace, semver, tags, runtime="nanopynq")
    manifest_yaml = _make_manifest_stub(
        runtime=runtime,
        board=board,
        device_family=entry.device_family if entry else None,
        design_ref=f"{design_name}:{version}",
        interfaces=[],
        design_portability=None,
        bitstream_file=bitstream_file,
        npq_file=npq_path.name,
    )
    platform_id = f"{board}/{runtime}"
    index_yaml = _make_index_stub(
        design_name=design_name, version=version, summary=description, tags=tag_list, platform_id=platform_id
    )
    _check_pushable(index_yaml, manifest_yaml)
    _write_outputs(
        design_name,
        platform_id,
        index_yaml,
        manifest_yaml,
        copies={bitstream_file: bitstream_path, npq_path.name: npq_path},
        missing_bitstream=None,
        output_dir=output_dir,
        dry_run=dry_run,
    )


def _read_npq_target(npq_path: Path) -> tuple[str, str]:
    """Return ``(board, bitstream file name)`` from the .npq, exiting 3 if they are unusable."""
    try:
        if npq_path.stat().st_size > MAX_NPQ_BYTES:
            raise ValueError(f"larger than {MAX_NPQ_BYTES} bytes")
        data = json.loads(npq_path.read_bytes())
    except (OSError, ValueError, RecursionError) as exc:
        click.echo(f"Error: Cannot read {npq_path}: {exc}", err=True)
        exit_with(FailureKind.INVALID)
    board = data.get("board") if isinstance(data, dict) else None
    bitstream = data.get("bitstream") if isinstance(data, dict) else None
    bitstream_file = bitstream.get("file") if isinstance(bitstream, dict) else None
    if not isinstance(board, str) or not isinstance(bitstream_file, str) or not is_single_filename(bitstream_file):
        click.echo(f"Error: {npq_path} needs a board and a bitstream.file (a file name next to it)", err=True)
        exit_with(FailureKind.INVALID)
    if _MANIFEST_FILE in (npq_path.name, bitstream_file):
        # Both are copied next to the generated manifest and would overwrite it.
        click.echo(f"Error: neither the .npq nor its bitstream can be named {_MANIFEST_FILE}", err=True)
        exit_with(FailureKind.INVALID)
    return board, bitstream_file


@dataclass(frozen=True)
class _BuildInputs:
    """What ``build`` reads from the HWH and asks the user, before generating YAML."""

    hwh_path: Path
    hwh_info: HwhInfo
    board_id: str
    device_family: str | None
    design_name: str
    version: str
    runtime: str
    description: str | None
    tags: list[str]

    @property
    def platform_id(self) -> str:
        return f"{self.board_id}/{self.runtime}"


def _collect_inputs(
    directory: Path,
    hwh_path_opt: Path | None,
    *,
    namespace: str | None,
    semver: str | None,
    tags: str | None,
) -> _BuildInputs:
    hwh_path = _select_hwh(directory, hwh_path_opt)
    hwh_info = _parse_hwh_and_report(hwh_path)
    board_id, device_family = _choose_board(hwh_info)

    # --- Interactive inputs ---
    click.echo("")
    design_name, version, runtime, description, tag_list = _prompt_design(namespace, semver, tags, runtime=None)

    return _BuildInputs(
        hwh_path=hwh_path,
        hwh_info=hwh_info,
        board_id=board_id,
        device_family=device_family,
        design_name=design_name,
        version=version,
        runtime=runtime,
        description=description,
        tags=tag_list,
    )


def _prompt_design(
    namespace: str | None, semver: str | None, tags: str | None, *, runtime: str | None
) -> tuple[str, str, str, str | None, list[str]]:
    """Ask for ``(design_name, version, runtime, description, tags)``; *runtime*, when given, is not asked."""
    ns_prompt = f"{namespace}/" if namespace else ""
    design_name: str = click.prompt(
        f"Design name [{ns_prompt}]",
        default=f"{ns_prompt}my-design",
        value_proc=lambda raw: _to_design_name(raw, namespace),
    )

    version: str = click.prompt("Version", default=semver or "0.1.0", value_proc=_to_version)

    if runtime is None:
        runtime = _prompt_runtime()

    description = click.prompt("Description", default="", show_default=False).strip() or None

    raw_tags = tags or click.prompt("Tags (comma-separated, optional)", default="", show_default=False)
    tag_list = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else []
    return design_name, version, runtime, description, tag_list


def _to_design_name(raw: str, namespace: str | None) -> str:
    """Return *raw* as ``<namespace>/<design>``, asking for the namespace when *raw* has none.

    push refuses an index whose name is not a :data:`DesignName`, so build checks it here;
    raising :class:`click.BadParameter` makes ``click.prompt`` ask again.
    """
    name = raw.strip()
    if "/" not in name:
        ns = namespace or click.prompt("Namespace").strip()
        name = f"{ns}/{name}"
    try:
        return _DESIGN_NAME_ADAPTER.validate_python(name)
    except ValidationError:
        msg = f"{name!r} is not <namespace>/<design> (lowercase letters, digits and hyphens)"
        raise click.BadParameter(msg) from None


def _to_version(raw: str) -> str:
    """Return *raw* stripped; push refuses an index whose version is not SemVer, so ask again."""
    version = raw.strip()
    try:
        return _SEMVER_ADAPTER.validate_python(version)
    except ValidationError:
        raise click.BadParameter(f"{version!r} is not a SemVer version (e.g. 1.0.0)") from None


def _check_pushable(index_yaml: str, manifest_yaml: str) -> None:
    """Run push's checks on the generated YAML, so build never writes what push rejects."""
    for name, check in (
        ("fabricgate-index.yaml", lambda: DesignIndex.model_validate(yaml.safe_load(index_yaml))),
        ("manifest.yaml", lambda: parse_platform_manifest(manifest_yaml)),
    ):
        try:
            check()
        except ValidationError as exc:
            click.echo(f"Error: The generated {name} would be rejected by push: {exc}", err=True)
            exit_with(FailureKind.INVALID)


def _select_hwh(directory: Path, hwh_path_opt: Path | None) -> Path:
    if hwh_path_opt is not None:
        # Used as given: the scan's skipping of earlier builds' copies does not apply.
        return hwh_path_opt
    click.echo("Scanning for HWH files ...")
    # An HWH next to a manifest.yaml is the copy an earlier build made, not a Vivado output.
    hwh_files = [f for f in find_hwh_files(directory) if not (f.parent / "manifest.yaml").exists()]
    if not hwh_files:
        click.echo(f"Error: No .hwh files found in {directory}", err=True)
        exit_with(FailureKind.GENERIC)
    for f in hwh_files:
        click.echo(f"  Found: {f.relative_to(directory.parent)}")

    if len(hwh_files) > 1:
        if not _stdin_is_tty():
            # Non-interactive: error — user must specify via --hwh
            names = ", ".join(f.name for f in hwh_files)
            click.echo(
                f"\nError: Multiple HWH files found ({names}). Use --hwh to specify one.",
                err=True,
            )
            exit_with(FailureKind.GENERIC)
        click.echo("\nMultiple HWH files found. Select one:")
        for i, f in enumerate(hwh_files):
            click.echo(f"  [{i + 1}] {f.relative_to(directory.parent)}")
        choice = click.prompt(
            "Select HWH file",
            type=click.IntRange(1, len(hwh_files)),
            default=1,
        )
        return hwh_files[choice - 1]
    return hwh_files[0]


def _parse_hwh_and_report(hwh_path: Path) -> HwhInfo:
    click.echo("\nParsing HWH ...")
    try:
        hwh_info = parse_hwh(hwh_path)
    except HwhParseError as exc:
        click.echo(f"Error: {exc}", err=True)
        exit_with(FailureKind.GENERIC)

    if hwh_info.boardpart:
        click.echo(f"  BOARDPART: {hwh_info.boardpart}")
    else:
        click.echo("  BOARDPART: (not found)")
    if hwh_info.interfaces:
        axi_count = sum(1 for i in hwh_info.interfaces if i.is_axi)
        bs_count = sum(1 for i in hwh_info.interfaces if i.is_board_specific)
        iface_summary = f"{len(hwh_info.interfaces)} interface(s)"
        if axi_count:
            iface_summary += f", {axi_count} AXI"
        if bs_count:
            bs_names = [i.vlnv.split(":")[-2] for i in hwh_info.interfaces if i.is_board_specific]
            iface_summary += f", board-specific: {', '.join(bs_names)}"
        click.echo(f"  EXTERNALINTERFACES: {iface_summary}")
    else:
        click.echo("  EXTERNALINTERFACES: (none)")

    click.echo(f"  → design_portability: {hwh_info.design_portability}")
    return hwh_info


def _choose_board(hwh_info: HwhInfo) -> tuple[str, str | None]:
    """Return ``(board_id, device_family)``, asking before using a board that is not in the Board DB."""
    try:
        db = BoardDb.load()
    except Exception as exc:
        click.echo(f"Error: Board DB load failed: {exc}", err=True)
        # INFRA: failed to load local Board DB (ADR-008 exit-code standard).
        exit_with(FailureKind.INFRA)

    if not hwh_info.boardpart:
        click.echo("  → No BOARDPART; cannot look up board")
        return _resolve_board_id(hwh_info, db)

    entry = db.lookup_by_boardpart(hwh_info.boardpart)
    if entry:
        click.echo(f"  → Board DB match: {entry.board_id} ({entry.display_name})")
        return entry.board_id, entry.device_family

    # Not in Board DB — propose virtual board
    click.echo("  → Board DB: not found")
    proposed_board_id, device_family = _resolve_board_id(hwh_info, db)
    if hwh_info.design_portability == "axi-only":
        click.echo(f"  → Suggest: {proposed_board_id}")
        if not click.confirm(
            f"\n  Board '{hwh_info.boardpart}' is not in the Board DB.\n"
            f"  Treat as {proposed_board_id} (AXI-only, any {device_family} board)?",
            default=True,
        ):
            click.echo("Aborted.", err=True)
            exit_with(FailureKind.GENERIC)
    else:
        click.echo(f"  → Suggest: {proposed_board_id}")
        if not click.confirm(
            f"\n  Platform will be registered as: {proposed_board_id}/<runtime>\n  Continue?",
            default=True,
        ):
            click.echo("Aborted.", err=True)
            exit_with(FailureKind.GENERIC)
    return proposed_board_id, device_family


def _write_outputs(
    design_name: str,
    platform_id: str,
    index_yaml: str,
    manifest_yaml: str,
    *,
    copies: dict[str, Path],
    missing_bitstream: str | None,
    output_dir: Path | None,
    dry_run: bool,
) -> None:
    """Write the index and manifest; *copies* maps a file name next to the manifest to its source."""
    design_slug = design_name.split("/")[-1] if "/" in design_name else design_name
    out_root = output_dir if output_dir is not None else Path(design_slug)
    # push reads <project>/<board_id>/<runtime>/manifest.yaml
    manifest_path = out_root / platform_id / "manifest.yaml"
    index_path = out_root / "fabricgate-index.yaml"
    copy_paths = [manifest_path.parent / name for name in copies]

    click.echo("\nGenerated:")
    if dry_run:
        click.echo(f"  (dry-run) {index_path}")
        click.echo(f"  (dry-run) {manifest_path}")
        for path in copy_paths:
            click.echo(f"  (dry-run) {path}")
        click.echo("\n--- fabricgate-index.yaml ---")
        click.echo(index_yaml)
        click.echo(f"--- {platform_id}/manifest.yaml ---")
        click.echo(manifest_yaml)
        return

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest_yaml, encoding="utf-8")
        # push uploads each artifact from next to the manifest, under the name the manifest declares.
        for source, dest in zip(copies.values(), copy_paths, strict=True):
            if source.resolve() != dest.resolve():
                shutil.copyfile(source, dest)
        index_path.write_text(index_yaml, encoding="utf-8")
    except OSError as exc:
        click.echo(f"Error: Write failed: {exc}", err=True)
        exit_with(FailureKind.INFRA)

    click.echo(f"  {index_path}  ✓")
    click.echo(f"  {manifest_path}  ✓")
    for path in copy_paths:
        click.echo(f"  {path}  ✓")
    if missing_bitstream is not None:
        click.echo(f"\nNote: Copy the bitstream to {manifest_path.parent / missing_bitstream} before publishing.")
    click.echo(f"Run `fabricgate push {out_root}/` to publish.")


def _prompt_runtime() -> str:
    result: str = click.prompt(
        "Runtime",
        # nanopynq is built from a .npq (--npq), not from an HWH.
        type=click.Choice(["pynq", "linux-fpgamgr"]),
        default="pynq",
    )
    return result
