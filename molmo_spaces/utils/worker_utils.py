# Shared utilities for distributed workers (datagen and eval).
#
# This module provides common functionality used by both data generation
# and evaluation workers, including:
# - Resource usage logging (CPU, memory, GPU)
# - Background memory monitoring
# - GPU memory utilities
# - Process exit code interpretation

import contextlib
import os
import threading
from typing import Any

import psutil
import torch

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def log_resource_usage(logger: Any, prefix: str = "") -> None:
    """
    Log current process and system resource usage.

    Logs process memory, system memory, CPU usage, and GPU memory (if available).
    Includes Beaker allocation info when running on Beaker.

    Args:
        logger: Logger instance (anything with .info() and .debug() methods)
        prefix: Prefix string for log message
    """

    try:
        process = psutil.Process()
        process_info = process.memory_info()
        process_mem_mb = process_info.rss / 1024 / 1024

        system_mem = psutil.virtual_memory()
        system_mem_used_gb = system_mem.used / 1024 / 1024 / 1024

        cpu_percent = process.cpu_percent(interval=0.1)

        # Get system total memory
        system_mem_total_gb = system_mem.total / 1024 / 1024 / 1024
        system_mem_percent = system_mem.percent

        # Build memory usage message
        log_msg = (
            f"{prefix}Resource usage: "
            f"Process={process_mem_mb:.1f}MB, "
            f"System={system_mem_used_gb:.1f}/{system_mem_total_gb:.1f}GB ({system_mem_percent:.1f}%)"
        )

        # Add Beaker allocation as separate readout if available
        beaker_memory_bytes = os.getenv("BEAKER_ASSIGNED_MEMORY_BYTES")
        if beaker_memory_bytes:
            allocated_mem_gb = int(beaker_memory_bytes) / 1024**3
            allocated_mem_percent = (system_mem_used_gb / allocated_mem_gb) * 100
            log_msg += f", Allocated={system_mem_used_gb:.1f}/{allocated_mem_gb:.1f}GB ({allocated_mem_percent:.1f}%)"

        log_msg += f", CPU={cpu_percent:.1f}%"

        # Add GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_mem_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_mem_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                gpu_mem_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                log_msg += f", GPU={gpu_mem_allocated_mb:.1f}/{gpu_mem_reserved_mb:.1f}/{gpu_mem_total_mb:.1f}MB"
            except Exception as gpu_e:
                logger.debug(f"Failed to get GPU memory: {gpu_e}")

        logger.info(log_msg)
    except Exception as e:
        logger.warning(f"Failed to log resource usage: {e}")


def get_actual_gpu_memory_usage() -> tuple[int | None, int | None]:
    """
    Get actual GPU memory usage across all processes using NVIDIA Management Library.

    This uses pynvml to get the true GPU memory usage (not just PyTorch's view),
    which is critical for determining how many workers can fit on a GPU.

    Returns:
        Tuple of (used_bytes, total_bytes), or (None, None) if unavailable
    """
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        return None, None

    try:
        # Initialize NVML if not already done
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlInit()

        # Get memory info for device 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        return mem_info.used, mem_info.total
    except Exception:
        return None, None


def interpret_exit_code(exit_code: int | None) -> str:
    """
    Interpret process exit code to provide helpful debugging information.

    Common exit codes:
    - 0: Success
    - 1: General error
    - -9 (SIGKILL): Often indicates OOM kill
    - -11 (SIGSEGV): Segmentation fault
    - -15 (SIGTERM): Process was terminated

    Args:
        exit_code: Process exit code (negative for signals)

    Returns:
        Human-readable interpretation of the exit code
    """
    if exit_code == 0:
        return "success"
    elif exit_code == 1:
        return "general error"
    elif exit_code == -9:
        return "SIGKILL - likely OOM (out of memory) kill"
    elif exit_code == -11:
        return "SIGSEGV - segmentation fault"
    elif exit_code == -15:
        return "SIGTERM - terminated"
    elif exit_code == -6:
        return "SIGABRT - abort signal"
    elif exit_code is None:
        return "still running"
    else:
        return f"unknown (signal {-exit_code} or error {exit_code})"


class BackgroundMemoryMonitor:
    """
    Background thread that periodically logs memory usage.

    Useful for monitoring worker resource usage during long-running jobs.
    Especially important for eval workers where GPU memory is the bottleneck.

    Example usage:
        monitor = BackgroundMemoryMonitor(logger, worker_id=0, interval_seconds=30)
        monitor.start()
        # ... do work ...
        monitor.stop()
    """

    def __init__(
        self,
        logger: Any,
        worker_id: int,
        interval_seconds: float = 30.0,
        prefix_template: str = "[Worker {worker_id}] Monitor: ",
    ):
        """
        Initialize the background memory monitor.

        Args:
            logger: Logger instance
            worker_id: Worker ID for log messages
            interval_seconds: How often to log resource usage
            prefix_template: Template for log prefix (can use {worker_id})
        """
        self.logger = logger
        self.worker_id = worker_id
        self.interval_seconds = interval_seconds
        self.prefix = prefix_template.format(worker_id=worker_id)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.debug(
            f"Worker {self.worker_id} started background memory monitor "
            f"(interval: {self.interval_seconds}s)"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the background monitoring thread.

        Args:
            timeout: How long to wait for thread to stop
        """
        if self._thread is None:
            return

        self.logger.debug(f"Worker {self.worker_id} stopping background memory monitor...")
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def _monitor_loop(self) -> None:
        """Background thread loop that logs memory usage at regular intervals."""
        while not self._stop_event.is_set():
            log_resource_usage(self.logger, self.prefix)
            self._stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "BackgroundMemoryMonitor":
        """Context manager entry - starts monitoring."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops monitoring."""
        self.stop()
