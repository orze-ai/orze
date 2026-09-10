def _finalization_paths(tp, idea_dir: Path, cfg: Mapping) -> tuple[Path, ...]:
    receipt_dir = _receipt_dir(idea_dir, str(tp.attempt_id), create=False)
    state_root = Path(cfg.get("_orze_dir") or (
        Path(cfg.get("_project_root", ".")) / ".orze"))
    separation = cfg["data_separation"]
    return tuple(sorted({path.absolute() for path in (
        _idea_path(idea_dir, cfg["model_lineage"]["artifact"]),
        receipt_dir / "boundary.json", receipt_dir / "start.json",
        state_root / "state" / "data_separation.json",
        Path(separation["train_manifest"]), Path(separation["evaluation_manifest"]),
    )}, key=str))
