#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-10 高速下载器：镜像校验 + 多连接分块 + 断点续传 + MD5 校验 + 自动解压。

为什么需要它
------------
官方源（``www.cs.toronto.edu/~kriz``）在部分网络环境下**单连接速率极低**：
本机实测仅 **0.11 MB/s**，完整下载 162 MB 需要约 **25 分钟** —— 这正是
"快速冒烟测试无法在 2 分钟内完成"的首要原因。

该源**支持 HTTP Range**，因此可以拆成 N 个分片并行下载，把聚合带宽提升一个量级：
本机 16 连接实测 **1.33 MB/s（约 12 倍）**，完整下载 + 校验 + 解压共 **123 s**。

一次真实故障与由此确立的设计约束
--------------------------------
Colab 上曾出现"整包进度显示 100% 却最终失败"的现象：分片 ``part_15`` 反复
``HTTPError``，重试 30 次后停在 ``10655754/10656121`` 字节。

根因**不是**网络抖动，而是所选镜像 ``cs231n.stanford.edu`` 上那份归档被**截断**了：

===============================================  ==========================
服务器                                          ``Content-Length``
===============================================  ==========================
``cs231n.stanford.edu``（速率最快，被选中）      ``170497704``（**短 367 字节**）
``data.brainchip.com``                           ``170498071``（正确）
===============================================  ==========================

于是 ::

    Range: bytes=159841950-170498070  ->  206  Content-Range: bytes 159841950-170497703/170497704
    Range: bytes=170497704-170498070  ->  416  Requested Range Not Satisfiable

服务器把请求**裁剪到自己的真实文件末尾**（因此 ``part_15`` 恰好停在 10655754），
续传时请求已位于其 EOF 之外的区间，被判定为**不可满足**并返回 **416** ——
这是一个**永久性**错误，重试 30 次当然还是 416。而进度按"已收字节 / 本地常量
``ARCHIVE_SIZE``"计算，``170497704/170498071 = 99.9998%`` 显示成了 ``100.0%``，
掩盖了真相。

由此确立四条硬约束（本脚本的实现依据）：

1. **先校验远端长度再下载**：用 ``HEAD`` 取 ``Content-Length``（失败则退回
   ``Range: bytes=0-0`` 解析 ``Content-Range``），与 ``ARCHIVE_SIZE`` 不符的镜像
   直接判定为**已损坏**并剔除 —— 它即使 16 个分片全部"成功"，MD5 也必然错。
2. **区分致命错误与可重试错误**：``416/404/403/...`` 属于请求本身不合法，
   立刻失败并**切换镜像**；只有超时、连接重置、``5xx``、``429`` 才退避重试。
3. **分块要小且可单独重下**：固定大小分块（默认 4 MiB）排队领取，
   单个分块失败只需要重下 4 MiB，而不是 10 MiB 的整段。
4. **最终以 MD5 为准，任何环节失败都要能回退**：长度 + MD5 双重校验，
   失败则清理残留 → 换下一个镜像 → 最后退回单连接。

脚本流程
--------
1. **镜像体检**：``HEAD`` 查长度（剔除损坏镜像）→ 有界读取采样 1 MB 测速 →
   探测 Range 支持；
2. **择优下载**：优先"长度正确 + 支持 Range + 速率最高"，用分块队列并发下载，
   分块落盘保留、断点续传、指数退避重试；
3. **回退链**：当前镜像多连接失败 → 尝试单连接 → 换下一个镜像（均自动清理残留）；
4. **完整性校验**：合并后比对文件大小 + MD5，期望值取自工程常量（避免硬编码漂移）；
5. **自动解压**：产出 torchvision 可直接读取的 ``cifar-10-batches-py`` 目录。

用法
----
    python scripts/fetch_cifar10.py                        # 下载到 03_code/data
    python scripts/fetch_cifar10.py --dest ../04_data/cifar10
    python scripts/fetch_cifar10.py --threads 8 --chunk-mb 4   # Colab 推荐值（默认）
    python scripts/fetch_cifar10.py --probe-only           # 只体检镜像、不下载
    python scripts/fetch_cifar10.py --force                # 忽略缓存重新下载
"""

from __future__ import annotations

import argparse
import hashlib
import os
import queue
import random
import re
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# 让脚本能直接以 `python scripts/fetch_cifar10.py` 运行（把工程根目录加入模块搜索路径）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10 import (  # noqa: E402  (必须在上面的 sys.path 处理之后导入)
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    ARCHIVE_SIZE,
    BASE_FOLDER,
)

#: 候选镜像（按经验排序；脚本会实测速率并校验长度后择优）
DEFAULT_MIRRORS: List[str] = [
    "https://www.cs.toronto.edu/~kriz/" + ARCHIVE_NAME,
    "http://cs231n.stanford.edu/" + ARCHIVE_NAME,
    "http://data.brainchip.com/dataset-mirror/cifar10/" + ARCHIVE_NAME,
]

#: 统一的 User-Agent（部分站点会拒绝 urllib 默认 UA）
USER_AGENT = "Mozilla/5.0 (compatible; cifar10-fetcher/2.0)"

#: 单次探测读取的字节数（1 MB，足以估算速率）
PROBE_BYTES = 1 << 20
#: 单个分块的读写块大小
CHUNK_SIZE = 1 << 18

#: 默认分块大小（4 MiB）：太小会放大请求开销，太大则单块失败的重下代价高
DEFAULT_CHUNK_SIZE = 4 << 20
#: 默认并发连接数：8 已接近多数镜像的饱和点，16 容易触发限流且收益递减
DEFAULT_THREADS = 8
#: 默认单分块最大重试次数（不含首次尝试）
DEFAULT_MAX_RETRIES = 8
#: 默认 socket 超时（秒）：作用于单次连接/读取操作
DEFAULT_TIMEOUT = 60.0
#: 退避参数：delay = U[0.5, 1.0] * min(CAP, BASE * FACTOR ** attempt)
BACKOFF_BASE = 1.0
BACKOFF_FACTOR = 2.0
BACKOFF_CAP = 30.0

#: 值得重试的 HTTP 状态码（临时性故障 / 限流）
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
#: 致命的 HTTP 状态码（重试无意义：请求本身不合法或资源不存在/越界）
FATAL_HTTP_STATUS = frozenset({400, 401, 403, 404, 405, 410, 416, 451})

_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
_log_lock = threading.Lock()


def log(message: str) -> None:
    """带时间戳的线程安全输出。"""
    with _log_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# 异常与工具
# --------------------------------------------------------------------------- #
class FatalDownloadError(RuntimeError):
    """不可重试的错误（区间越界 416、资源不存在 404、服务器忽略 Range 等）。

    这类错误重试多少次结果都一样，应当**立即**放弃当前镜像并切换到备用镜像。
    """


@dataclass
class Mirror:
    """一个候选镜像的体检结果。"""

    url: str
    speed: Optional[float]           # 采样下行速率（MB/s）；None 表示探测失败
    range_ok: bool                   # 是否支持 HTTP Range
    remote_size: Optional[int]       # 远端口径的文件大小；None 表示服务器未提供

    @property
    def size_ok(self) -> Optional[bool]:
        """远端大小是否与期望一致；None 表示无法判断（服务器未提供长度）。"""
        if self.remote_size is None:
            return None
        return self.remote_size == ARCHIVE_SIZE

    @property
    def usable(self) -> bool:
        """已知长度不符的镜像一定是坏的（MD5 必然失败），直接排除。"""
        return self.size_ok is not False


def md5_of(path: Path, chunk_size: int = 1 << 22) -> str:
    """计算文件 MD5。"""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def backoff_delay(attempt: int) -> float:
    """指数退避 + 抖动。

    纯指数退避会让 8 个并发分块在同一时刻一起醒来（惊群），
    因此在 ``[0.5, 1.0] * window`` 上取随机值把重试打散。
    """
    window = min(BACKOFF_CAP, BACKOFF_BASE * (BACKOFF_FACTOR ** attempt))
    return random.uniform(window * 0.5, window)


def parse_content_range(value: str) -> Tuple[Optional[int], Optional[int]]:
    """解析 ``Content-Range: bytes 0-99/170498071``，返回 ``(起始偏移, 总长度)``。"""
    match = _CONTENT_RANGE_RE.match((value or "").strip())
    if not match:
        return None, None
    total = None if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), total


def is_retryable(error: BaseException) -> bool:
    """判断异常是否值得重试（临时性故障）还是应立即失败（永久性故障）。"""
    if isinstance(error, FatalDownloadError):
        return False
    if isinstance(error, urllib.error.HTTPError):
        if error.code in FATAL_HTTP_STATUS:
            return False
        # 其余 4xx/5xx 中，仅重试已知的临时性状态码
        return error.code in RETRYABLE_HTTP_STATUS
    # 超时、连接重置、IncompleteRead、DNS 抖动等一律视为临时性
    return True


# --------------------------------------------------------------------------- #
# 镜像体检
# --------------------------------------------------------------------------- #
def remote_size(url: str, timeout: float = 15.0) -> Optional[int]:
    """获取远端文件大小：优先 ``HEAD``，不支持时退回 ``Range: bytes=0-0``。

    这是定位"镜像截断"故障的关键一步 —— 只有拿到服务器**自己的**文件长度，
    才能发现"这份归档比官方少 367 字节"。
    """
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length and length.strip().isdigit() and int(length) > 0:
                return int(length)
    except Exception:  # noqa: BLE001  某些服务器禁用 HEAD，退回 Range 探测
        pass

    try:
        request = urllib.request.Request(
            url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            _, total = parse_content_range(response.headers.get("Content-Range", ""))
            if total is not None:
                return total
            if response.status == 200:
                length = response.headers.get("Content-Length")
                if length and length.strip().isdigit():
                    return int(length)
    except Exception:  # noqa: BLE001
        pass
    return None


def probe_speed(url: str, nbytes: int = PROBE_BYTES, deadline: float = 8.0) -> Optional[float]:
    """采样测量单个连接的下行速率（MB/s）；不可用或过慢时返回 None。

    两个必须注意的点
    ----------------
    1. **有界读取**：若镜像忽略 Range 头而返回 200，
       无界的 ``resp.read()`` 会去读取整个归档（162 MB），导致探测长时间卡住。
    2. **硬性截止时间**：``urlopen`` 的 timeout 只作用于**单次 socket 操作**，
       慢速服务器每次都能在超时内返回少量数据，累加起来仍可能耗时数十秒
       （实测多伦多源探测耗时 42 s）。因此这里在读取循环内主动检查截止时间，
       一旦超时立即放弃该镜像。
    """
    start = time.time()
    try:
        request = urllib.request.Request(url, headers={
            "Range": f"bytes=0-{nbytes - 1}",
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
        })
        received = 0
        with urllib.request.urlopen(request, timeout=deadline) as response:
            while received < nbytes:
                if time.time() - start > deadline:     # 硬性截止，避免被慢速源拖住
                    break
                block = response.read(min(1 << 16, nbytes - received))
                if not block:
                    break
                received += len(block)
    except Exception:
        return None

    elapsed = max(1e-6, time.time() - start)
    if elapsed > deadline or received < nbytes // 4:   # 太慢或数据太少，视为不可用
        return None
    return received / (1 << 20) / elapsed


def supports_range(url: str, timeout: float = 8.0) -> bool:
    """判断镜像是否支持 HTTP Range（多连接下载的前提）。"""
    try:
        request = urllib.request.Request(
            url, headers={"Range": "bytes=0-99", "User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            data = response.read(100)                  # 有界读取
        return status == 206 and len(data) == 100
    except Exception:
        return False


def inspect_mirrors(mirrors: List[str]) -> List[Mirror]:
    """逐个体检候选镜像：远端长度 → 采样速率 → Range 支持。"""
    results: List[Mirror] = []
    for url in mirrors:
        size = remote_size(url)
        if size is None:
            log(f"  长度未知              <- {url}")
        elif size != ARCHIVE_SIZE:
            log(f"  长度不符 {size} (期望 {ARCHIVE_SIZE}，差 {ARCHIVE_SIZE - size} 字节) "
                f"-> 判定为损坏镜像 <- {url}")
        else:
            log(f"  长度正确 {size}       <- {url}")

        speed = probe_speed(url)
        if speed is None:
            log(f"  不可用（探测超时/过慢）<- {url}")
            continue

        range_ok = supports_range(url)
        mirror = Mirror(url, speed, range_ok, size)
        results.append(mirror)
        log(f"  可用 {speed:6.2f} MB/s | Range={str(range_ok):5s} | "
            f"长度={'正确' if mirror.size_ok else ('未知' if mirror.size_ok is None else '不符')}"
            f" <- {url}")
    return results


# --------------------------------------------------------------------------- #
# 分块下载
# --------------------------------------------------------------------------- #
def is_archive_valid(path: Path) -> bool:
    """判断归档文件是否完整（大小 + MD5 双重校验）。"""
    try:
        if not path.is_file() or path.stat().st_size != ARCHIVE_SIZE:
            return False
        return md5_of(path) == ARCHIVE_MD5
    except Exception:
        return False


def is_extracted(dest: Path) -> bool:
    """判断是否已解压就绪。"""
    return (dest / BASE_FOLDER / "test_batch").is_file()


def plan_chunks(expect_size: int, chunk_size: int) -> List[Tuple[int, int]]:
    """把 ``[0, expect_size)`` 切成固定大小的闭区间列表 ``[(start, end), ...]``。"""
    return [
        (start, min(expect_size - 1, start + chunk_size - 1))
        for start in range(0, expect_size, chunk_size)
    ]


def download_chunk(
    url: str,
    start: int,
    end: int,
    part: Path,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """下载字节区间 ``[start, end]`` 到分块文件，支持断点续传 + 指数退避重试。

    关键约束
    --------
    * 服务器**必须**回 ``206``；若回 ``200``（忽略了 Range），绝不能按偏移追加，
      否则数据会整体错位 —— 这里直接判定为致命错误。
    * 校验 ``Content-Range`` 的起始偏移与总长度，防止"服务器裁剪区间"被静默接受
      （这正是 ``part_15`` 停在 10655754 字节的直接原因）。
    * 致命错误（416 等）立即抛出，交由上层切换镜像，不做无谓重试。
    """
    want = end - start + 1
    last_error: Optional[BaseException] = None

    for attempt in range(max_retries + 1):
        have = part.stat().st_size if part.is_file() else 0
        if have == want:
            return
        if have > want:
            # 上一次响应给了超出区间的数据（Range 被忽略 / 并发写入），只能整块重下
            part.unlink(missing_ok=True)
            have = 0

        try:
            request = urllib.request.Request(url, headers={
                "Range": f"bytes={start + have}-{end}",
                "User-Agent": USER_AGENT,
                # 必须禁用压缩：经过 gzip 传输后 Range 语义与 Content-Length 都会错位
                "Accept-Encoding": "identity",
            })
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise FatalDownloadError(
                        f"服务器忽略了 Range（HTTP {response.status}，期望 206）"
                    )
                got_start, total = parse_content_range(response.headers.get("Content-Range", ""))
                if got_start is None:
                    raise FatalDownloadError("响应缺少合法的 Content-Range 头")
                if got_start != start + have:
                    raise FatalDownloadError(
                        f"Content-Range 起始偏移不符（收到 {got_start}，期望 {start + have}）"
                    )
                if total is not None and total != ARCHIVE_SIZE:
                    raise FatalDownloadError(
                        f"远端文件长度为 {total}，与期望 {ARCHIVE_SIZE} 不符（镜像已损坏）"
                    )

                with open(part, "ab") as out:
                    while True:
                        block = response.read(CHUNK_SIZE)
                        if not block:
                            break
                        out.write(block)
        except BaseException as error:  # noqa: BLE001
            if isinstance(error, FatalDownloadError) or not is_retryable(error):
                raise FatalDownloadError(
                    f"分片 [{start}-{end}] 遇到不可重试错误：{type(error).__name__}: {error}"
                ) from error
            last_error = error
            if attempt >= max_retries:
                break
            delay = backoff_delay(attempt)
            if attempt < 3 or attempt % 2 == 0:
                log(f"  分片 [{start}-{end}] 第 {attempt + 1}/{max_retries + 1} 次尝试失败"
                    f"（{type(error).__name__}: {error}），{delay:.1f}s 后重试")
            time.sleep(delay)
        # 循环回到顶部：重新读取落盘大小，已收满则返回

    have = part.stat().st_size if part.is_file() else 0
    raise RuntimeError(
        f"分片 [{start}-{end}] 下载不完整（{have}/{want} 字节，共 {max_retries + 1} 次尝试）"
        + (f"，最后错误：{type(last_error).__name__}: {last_error}" if last_error else "")
    )


def cleanup_partials(dest: Path) -> None:
    """清理分块目录与临时合并文件（切换镜像前必须调用，避免脏数据被复用）。"""
    part_dir = dest / "_parts"
    if part_dir.is_dir():
        shutil.rmtree(part_dir, ignore_errors=True)
    (dest / "_archive_merging.bin").unlink(missing_ok=True)


def multi_connection_download(
    url: str,
    dest: Path,
    threads: int = DEFAULT_THREADS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path:
    """多连接分块下载并合并，返回归档文件路径。

    实现要点
    --------
    * **工作队列**：分块索引放进队列，由 ``threads`` 个工作线程领取 ——
      先完成的分块会自动去领下一个，避免"慢分块拖住线程池"；
    * **分块独立重下**：每个分块单独落盘、单独续传、单独重试，
      失败只会重下该分块（默认 4 MiB）；
    * **快速熔断**：任一分块遇到致命错误（如 416）立即置位 ``abort`` 事件，
      其余线程尽快退出，由上层切换镜像。
    """
    part_dir = dest / "_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    # 分块大小是分块文件偏移的语义来源：与上次不一致则必须整体作废（否则数据错位）
    marker = part_dir / ".chunk_size"
    previous = marker.read_text().strip() if marker.is_file() else None
    if previous is not None and previous != str(chunk_size):
        log(f"  分块大小由 {previous} 变为 {chunk_size}，丢弃旧分块重新开始")
        cleanup_partials(dest)
        part_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(chunk_size))

    chunks = plan_chunks(ARCHIVE_SIZE, chunk_size)
    part_path = lambda index: part_dir / f"part_{index:04d}"  # noqa: E731

    # 断点续传：统计已有且未越界的分块进度
    resumed = sum(
        1 for index, (s, e) in enumerate(chunks)
        if part_path(index).is_file() and 0 < part_path(index).stat().st_size <= e - s + 1
    )
    log(f"启用 {len(chunks)} 个分块 × {chunk_size / (1 << 20):.1f} MiB，"
        f"并发 {threads} 连接"
        + (f"（复用 {resumed} 个已有分块）" if resumed else ""))

    start_time = time.time()
    stop_event = threading.Event()
    abort_event = threading.Event()
    pending: "queue.Queue[int]" = queue.Queue()
    for index in range(len(chunks)):
        pending.put(index)
    progress = {"done": 0}
    progress_lock = threading.Lock()

    def reporter() -> None:
        while not stop_event.wait(20):
            got = sum(
                part_path(i).stat().st_size for i in range(len(chunks)) if part_path(i).is_file()
            )
            elapsed = max(1e-6, time.time() - start_time)
            with progress_lock:
                done = progress["done"]
            log(f"  进度 {got / (1 << 20):7.1f}/{ARCHIVE_SIZE / (1 << 20):.1f} MB "
                f"({100 * got / ARCHIVE_SIZE:5.1f}%) 分块 {done}/{len(chunks)} "
                f"平均 {got / (1 << 20) / elapsed:.2f} MB/s 已用 {elapsed:.0f}s")

    def worker() -> None:
        while not abort_event.is_set():
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return
            start, end = chunks[index]
            try:
                download_chunk(url, start, end, part_path(index), max_retries, timeout)
            except BaseException as error:  # noqa: BLE001
                failures.append((index, start, end, error))
                if isinstance(error, FatalDownloadError):
                    abort_event.set()       # 致命错误：熔断，其余线程尽快退出
                return
            else:
                with progress_lock:
                    progress["done"] += 1

    failures: List[Tuple[int, int, int, BaseException]] = []
    reporter_thread = threading.Thread(target=reporter, daemon=True)
    reporter_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            list(pool.map(lambda _: worker(), range(max(1, threads))))
    finally:
        stop_event.set()

    if failures:
        fatal = next(
            (item for item in failures if isinstance(item[3], FatalDownloadError)), failures[0]
        )
        raise fatal[3]

    elapsed = max(1e-6, time.time() - start_time)
    got = sum(part_path(i).stat().st_size for i in range(len(chunks)))
    log(f"  分块下载完成：{got / (1 << 20):.1f} MB / {elapsed:.0f}s = "
        f"{got / (1 << 20) / elapsed:.2f} MB/s")

    return merge_and_verify(chunks, part_path, dest)


def merge_and_verify(chunks: List[Tuple[int, int]], part_path, dest: Path) -> Path:
    """按序合并分块，并做「文件大小 + MD5」双重校验。"""
    for index, (start, end) in enumerate(chunks):
        path = part_path(index)
        size = path.stat().st_size if path.is_file() else 0
        if size != end - start + 1:
            raise RuntimeError(
                f"分片 part_{index:04d} 长度异常（{size}/{end - start + 1} 字节）"
            )

    merge_tmp = dest / "_archive_merging.bin"
    with open(merge_tmp, "wb") as out:
        for index in range(len(chunks)):
            with open(part_path(index), "rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    out.write(block)

    actual_size = merge_tmp.stat().st_size
    if actual_size != ARCHIVE_SIZE:
        merge_tmp.unlink(missing_ok=True)
        raise RuntimeError(f"合并后文件大小异常：{actual_size}/{ARCHIVE_SIZE} 字节")

    log(f"  校验 MD5（{ARCHIVE_SIZE / (1 << 20):.1f} MB）...")
    digest = md5_of(merge_tmp)
    if digest != ARCHIVE_MD5:
        merge_tmp.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 校验失败：期望 {ARCHIVE_MD5}，实际 {digest}")
    log(f"  校验通过：大小 {ARCHIVE_SIZE} 字节，MD5 {digest}")

    archive = dest / ARCHIVE_NAME
    os.replace(merge_tmp, archive)
    shutil.rmtree(dest / "_parts", ignore_errors=True)
    return archive


def single_connection_download(
    url: str,
    dest: Path,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path:
    """单连接流式下载（最后的回退方案），支持断点续传 + 重试。"""
    archive = dest / ARCHIVE_NAME
    part = dest / (ARCHIVE_NAME + ".partial")

    for attempt in range(max_retries + 1):
        have = part.stat().st_size if part.is_file() else 0
        if have > ARCHIVE_SIZE:
            part.unlink(missing_ok=True)
            have = 0
        if have == ARCHIVE_SIZE:
            break

        log(f"单连接下载（第 {attempt + 1}/{max_retries + 1} 次尝试，从 "
            f"{have / (1 << 20):.1f} MB 处续传）...")
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        start_time = time.time()
        last_report = 0.0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                # 有断点却收到 200，说明服务器忽略了 Range：必须从头覆盖，不能追加
                mode = "ab"
                if have and response.status != 206:
                    log("  服务器忽略了 Range，从头重新下载")
                    have, mode = 0, "wb"
                with open(part, mode) as out:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        if time.time() - last_report > 15:
                            last_report = time.time()
                            got = out.tell()
                            elapsed = max(1e-6, time.time() - start_time)
                            log(f"  进度 {got / (1 << 20):7.1f}/{ARCHIVE_SIZE / (1 << 20):.1f} MB "
                                f"({100 * got / ARCHIVE_SIZE:5.1f}%) "
                                f"平均 {got / (1 << 20) / elapsed:.2f} MB/s")
        except BaseException as error:  # noqa: BLE001
            if isinstance(error, urllib.error.HTTPError) and error.code in FATAL_HTTP_STATUS:
                raise FatalDownloadError(
                    f"单连接遇到不可重试错误：HTTP {error.code} {error.reason}"
                ) from error
            if attempt >= max_retries:
                raise
            delay = backoff_delay(attempt)
            log(f"  单连接中断（{type(error).__name__}: {error}），{delay:.1f}s 后重试")
            time.sleep(delay)
        # 循环回到顶部：检查落盘大小，收满则结束

    size = part.stat().st_size if part.is_file() else 0
    if size != ARCHIVE_SIZE:
        raise RuntimeError(f"下载不完整：{size}/{ARCHIVE_SIZE} 字节")

    log(f"  校验 MD5（{size / (1 << 20):.1f} MB）...")
    digest = md5_of(part)
    if digest != ARCHIVE_MD5:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 校验失败：期望 {ARCHIVE_MD5}，实际 {digest}")
    log(f"  校验通过：大小 {size} 字节，MD5 {digest}")
    os.replace(part, archive)
    return archive


def download_with_fallback(
    mirrors: List[Mirror],
    dest: Path,
    threads: int,
    chunk_size: int,
    max_retries: int,
    timeout: float,
) -> Path:
    """按优先级依次尝试各镜像；每个镜像内部先多连接、再单连接。

    任何一次尝试失败都会**清理残留**后再进入下一档，保证不会把
    上一个镜像的半成品当成断点续传的基础（脏数据会导致 MD5 静默错误）。
    """
    failures: List[str] = []
    for rank, mirror in enumerate(mirrors, start=1):
        modes: List[Tuple[str, bool]] = []
        if mirror.range_ok:
            modes.append(("多连接分块", True))
        modes.append(("单连接", False))

        for label, use_multi in modes:
            log(f"[尝试 {rank}/{len(mirrors)}] 镜像 {mirror.url} —— {label}")
            try:
                if use_multi:
                    archive = multi_connection_download(
                        mirror.url, dest, threads, chunk_size, max_retries, timeout
                    )
                else:
                    archive = single_connection_download(mirror.url, dest, max_retries, timeout)
                log(f"[成功] {label} 完成：{archive}")
                return archive
            except BaseException as error:  # noqa: BLE001
                log(f"[回退] {label} 失败：{type(error).__name__}: {error}")
                failures.append(f"{mirror.url} / {label} / {type(error).__name__}: {error}")
                cleanup_partials(dest)
                (dest / (ARCHIVE_NAME + ".partial")).unlink(missing_ok=True)

    raise RuntimeError("所有镜像与下载方式均失败：\n  - " + "\n  - ".join(failures))


# --------------------------------------------------------------------------- #
# 解压
# --------------------------------------------------------------------------- #
def extract(archive: Path, dest: Path) -> None:
    """解压归档，产出 ``<dest>/cifar-10-batches-py``。"""
    if is_extracted(dest):
        log("已存在解压结果，跳过解压")
        return
    log(f"解压 {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    log(f"解压完成：{dest / BASE_FOLDER}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 高速下载器（镜像体检 + 多连接分块 + MD5 校验 + 解压）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dest", type=str, default=str(PROJECT_ROOT / "data"),
                        help="数据目录（与 train.py 的 --data-root 保持一致）")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="并发连接数")
    parser.add_argument("--chunk-mb", type=float, default=DEFAULT_CHUNK_SIZE / (1 << 20),
                        help="单个分块大小（MiB）")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="单个分块的最大重试次数（不含首次尝试）")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="单次 socket 操作超时（秒）")
    parser.add_argument("--mirrors", type=str, nargs="*", default=None, help="自定义镜像 URL 列表")
    parser.add_argument("--probe-only", action="store_true", help="只体检镜像，不下载")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存，强制重新下载")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ARCHIVE_NAME
    mirrors = [str(item) for item in (args.mirrors or DEFAULT_MIRRORS)]
    chunk_size = max(1 << 20, int(args.chunk_mb * (1 << 20)))
    threads = max(1, int(args.threads))

    log("=" * 78)
    log(f"目标目录：{dest}")
    log(f"归档期望：{ARCHIVE_NAME}  {ARCHIVE_SIZE} 字节 "
        f"({ARCHIVE_SIZE / (1 << 20):.1f} MB)  MD5={ARCHIVE_MD5}")
    log("=" * 78)

    # ---------------- 0. 缓存命中则直接复用 ---------------- #
    if not args.force and is_archive_valid(archive):
        log("归档已存在且 MD5 校验通过，跳过下载")
    elif not args.force and is_extracted(dest):
        log("数据已解压就绪（未保留归档文件），跳过下载")
        _report(dest)
        return 0
    else:
        # ---------------- 1. 镜像体检 ---------------- #
        log("体检候选镜像：远端长度 -> 采样 1 MB 测速 -> Range 支持 ...")
        inspected = inspect_mirrors(mirrors)
        for mirror in inspected:
            if not mirror.usable:
                log(f"[剔除] 远端长度为 {mirror.remote_size}，与期望 {ARCHIVE_SIZE} 不符"
                    f"（镜像已损坏）<- {mirror.url}")

        candidates = [mirror for mirror in inspected if mirror.usable]
        if not candidates:
            log("失败：所有镜像均不可用或长度不符。请检查网络，或改用 "
                "--dataset synthetic 走零下载冒烟测试。")
            return 1

        if args.probe_only:
            log("仅体检（--probe-only），结束。")
            return 0

        # 排序：先看是否支持 Range（决定能否多连接加速），再看速率
        candidates.sort(key=lambda item: (item.range_ok, item.speed or 0.0), reverse=True)
        log(f"镜像尝试顺序：{', '.join(item.url for item in candidates)}")

        # ---------------- 2. 下载（含回退链） ---------------- #
        try:
            archive = download_with_fallback(
                candidates, dest, threads, chunk_size, args.max_retries, args.timeout
            )
        except Exception as error:  # noqa: BLE001
            log(f"下载失败：{error}")
            return 1

    # ---------------- 3. 解压 ---------------- #
    extract(archive, dest)
    _report(dest)
    return 0


def _report(dest: Path) -> None:
    """打印最终目录结构与关键文件大小。"""
    log("-" * 78)
    log("数据就绪：")
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            log(f"  {path.relative_to(dest)}  ({path.stat().st_size / (1 << 20):.1f} MB)")
    log("提示：把它交给训练脚本 —— python train.py --data-root " + str(dest))
    log("-" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
