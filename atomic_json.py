"""
Atomic JSON persistence shared by the credential file and the probe cache.

The two stores used to carry their own copy of "write a temporary file, then rename
it into place", and the copies had already drifted apart (only one of them pushed the
bytes to disk before the rename). One implementation keeps the durability guarantee
in one place instead of two.

原子写 JSON，供凭证文件与探测缓存共用。

两处存储各自抄了一份"写临时文件再改名"的实现，而两份已经分叉（只有一份在改名
之前把数据真正推到磁盘）。集中成一份实现，把持久化保证收在一处，而不是两处。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _fsync_directory(directory: Path) -> None:
    """
    Push the rename itself to disk where the platform allows it.

    On POSIX the directory entry is what makes the new file visible after a crash, so
    the directory is fsynced too; Windows cannot open a directory as a file, and a
    failure here is never worth failing the write over.

    在平台允许时把"改名"这一步本身也推到磁盘。

    POSIX 上崩溃后能否看到新文件取决于目录项，因此目录也要 fsync；Windows 无法把
    目录当文件打开，且这里的失败绝不值得让整个写入失败。
    """
    if os.name != "posix":
        return
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Any) -> None:
    """
    Write `payload` to `path` as JSON, atomically.

    The bytes land in a temporary sibling first, are flushed and fsynced, and only
    then is the file renamed into place. A crash or a power loss therefore leaves
    either the previous file or the completely written new one -- never a
    half-written file, and never a renamed-but-empty one (which is what a rename
    without the fsync can produce, since the name is durable before the data is).

    把 `payload` 以 JSON 形式原子写入 `path`。

    字节先落到旁边的临时文件，flush + fsync 之后才改名就位。因此崩溃或断电后留下
    的要么是旧文件、要么是完整写入的新文件——绝不会是半截文件，也绝不会是"改了名
    却内容为空"的文件（改名不做 fsync 就可能这样：名字先持久化，数据却没有）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
