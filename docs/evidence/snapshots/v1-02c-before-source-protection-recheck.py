def _fs_lock(lock_dir: Path, stale_seconds: float = 600) -> bool:
    """Acquire a filesystem lock via atomic mkdir.
    Returns True if acquired, False if held by another.
    Auto-breaks stale locks older than stale_seconds using atomic rename
    to avoid TOCTOU races between nodes.  On the local host, also breaks
    locks whose owning PID has died (regardless of age)."""
    from orze.core.idea_source_lock import idea_source_lock_protected
    if idea_source_lock_protected(lock_dir):
        logger.debug("Refusing generic lock ownership in protected idea source namespace")
        return False
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
        meta = {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}
        (lock_dir / "lock.json").write_text(json.dumps(meta), encoding="utf-8")
        return True
    except FileExistsError:
        # A managed-role crash receipt is process authority, not a disposable
        # lock artifact. Only startup reconciliation on its owning host may
        # remove it after nonce-bound stable identities are proven stopped.
        role_receipt = lock_dir / "role-process.json"
        if role_receipt.exists() or role_receipt.is_symlink():
            logger.error(
                "Refusing lock takeover with unresolved managed role "
                "receipt: %s", lock_dir)
            return False
        # Check for stale lock
        try:
            meta_path = lock_dir / "lock.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    dir_age = time.time() - lock_dir.stat().st_mtime
                    if dir_age < 30:
                        return False
                    logger.warning(
                        "Breaking lock with unreadable metadata: %s (age %.0fs)",
                        lock_dir, dir_age)
                else:
                    lock_age = time.time() - meta.get("time", 0)
                    lock_host = meta.get("host", "")
                    lock_pid = meta.get("pid", 0)

                    pid_dead = (lock_host == socket.gethostname()
                                and lock_pid
                                and not _is_pid_alive(lock_host, lock_pid))
                    age_stale = lock_age > stale_seconds

                    if not (age_stale or pid_dead):
                        return False

                    if pid_dead:
                        logger.warning("Breaking dead-pid lock: %s (host=%s pid=%d)",
                                       lock_dir, lock_host, lock_pid)
                    else:
                        logger.warning("Breaking stale lock: %s (age %.0fs)",
                                       lock_dir, lock_age)
            else:
                # Orphaned lock dir with no lock.json — treat as stale
                # unless very recently created (< 30s grace period)
                dir_age = time.time() - lock_dir.stat().st_mtime
                if dir_age < 30:
                    return False
                logger.warning("Breaking orphaned lock (no lock.json): %s (age %.0fs)",
                               lock_dir, dir_age)

            # Atomic takeover: rename the stale lock dir to a unique name.
            # Only one node can succeed at this rename — the loser gets OSError.
            stale_name = lock_dir.with_name(
                f"{lock_dir.name}._stale_{uuid.uuid4().hex[:12]}"
            )
            try:
                os.rename(str(lock_dir), str(stale_name))
            except OSError:
                return False
            try:
                shutil.rmtree(stale_name)
            except OSError:
                pass
            try:
                lock_dir.mkdir(parents=True, exist_ok=False)
                new_meta = {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}
                (lock_dir / "lock.json").write_text(json.dumps(new_meta), encoding="utf-8")
                return True
            except FileExistsError:
                return False
        except Exception:
            pass
        return False

