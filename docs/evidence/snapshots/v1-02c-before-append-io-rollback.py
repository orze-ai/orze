def locked_append(path: Path, content: str, lock_dir: Path,
                  stale_seconds: float = 60, after_append=None) -> bool:
    """Append/finalize with source ownership, never an age-expiring lease.

    stale_seconds remains accepted for API compatibility but cannot authorize
    takeover of a possibly active or incompletely finalized source owner.
    """
    from orze.core.idea_source_lock import idea_source_lock, idea_source_lock_owned
    path = Path(path)
    lock_dir = Path(lock_dir)
    for candidate in (path, lock_dir):
        absolute = candidate.absolute()
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            if current.is_symlink():
                return False
    with idea_source_lock(lock_dir) as lease:
        if lease is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(path), flags, 0o644)
        try:
            metadata = os.fstat(fd)
            if (not statlib.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1):
                return False
            original_size = metadata.st_size
            encoded = content.encode("utf-8")
            if not idea_source_lock_owned(lease):
                raise OSError("idea_source_lock_ownership_lost")
            written = 0
            while written < len(encoded):
                count = os.write(fd, encoded[written:])
                if count <= 0:
                    raise OSError("locked_append_short_write")
                written += count
            os.fsync(fd)
            try:
                if after_append is not None:
                    if not callable(after_append):
                        raise TypeError("locked_append_finalizer_not_callable")
                    after_append()
                if not idea_source_lock_owned(lease):
                    raise OSError("idea_source_lock_ownership_lost")
            except BaseException:
                os.ftruncate(fd, original_size)
                os.fsync(fd)
                raise
        finally:
            os.close(fd)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return True


