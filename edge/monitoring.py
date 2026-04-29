"""System monitoring — thermal zones, CMA, RAM for Jetson."""

import os
import glob
from datetime import datetime


def read_thermal_zones():
    """Read all thermal zones from /sys/class/thermal/ (no root needed).

    Returns:
        Dict mapping zone name (e.g. "gpu-thermal") to temperature in Celsius.
    """
    temps = {}
    for tz_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(os.path.join(tz_path, "type"), "rb") as f:
                raw = f.read()
            if not raw:
                continue
            name = raw.decode().strip()

            with open(os.path.join(tz_path, "temp"), "rb") as f:
                raw = f.read()
            if not raw:
                continue
            val = raw.decode().strip()
            if val and val != "0":
                temps[name] = round(int(val) / 1000.0, 1)
        except (OSError, ValueError):
            pass
    return temps


def get_thermal():
    """Get GPU, CPU, and SoC temperatures.

    Returns:
        Dict with gpu_temp_c, cpu_temp_c, soc_temp_c keys.
    """
    temps = read_thermal_zones()
    return {
        "gpu_temp_c": temps.get("gpu-thermal"),
        "cpu_temp_c": temps.get("cpu-thermal"),
        "soc_temp_c": temps.get("tj-thermal"),
        "all_zones": temps,
    }


def get_memory_info():
    """Read memory info from /proc/meminfo.

    Returns:
        Dict with ram_total_mb, ram_available_mb, ram_used_mb,
        swap_total_mb, swap_used_mb, swap_free_mb.
    """
    result = {}
    try:
        with open("/proc/meminfo", "rb") as f:
            raw = f.read().decode()
        for line in raw.split("\n"):
            parts = line.split(":")
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            val_kb = int(parts[1].strip().split()[0])
            if key == "MemTotal":
                result["ram_total_mb"] = val_kb // 1024
            elif key == "MemAvailable":
                result["ram_available_mb"] = val_kb // 1024
            elif key == "SwapTotal":
                result["swap_total_mb"] = val_kb // 1024
            elif key == "SwapFree":
                result["swap_free_mb"] = val_kb // 1024

        if "ram_total_mb" in result and "ram_available_mb" in result:
            result["ram_used_mb"] = result["ram_total_mb"] - result["ram_available_mb"]
        if "swap_total_mb" in result and "swap_free_mb" in result:
            result["swap_used_mb"] = result["swap_total_mb"] - result["swap_free_mb"]
    except Exception:
        pass
    return result


def get_cma():
    """Read CMA (Contiguous Memory Allocator) stats.

    Returns:
        Dict with cma_total_mb, cma_free_mb, cma_used_mb, cma_used_pct.
    """
    result = {}
    try:
        with open("/proc/meminfo", "rb") as f:
            raw = f.read().decode()
        for line in raw.split("\n"):
            parts = line.split(":")
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            val_kb = int(parts[1].strip().split()[0])
            if "CmaTotal" in key:
                result["cma_total_mb"] = val_kb // 1024
            elif "CmaFree" in key:
                result["cma_free_mb"] = val_kb // 1024

        if "cma_total_mb" in result and "cma_free_mb" in result:
            t = result["cma_total_mb"]
            f = result["cma_free_mb"]
            result["cma_used_mb"] = t - f
            result["cma_used_pct"] = round((1 - f / t) * 100, 1) if t > 0 else 0
    except Exception:
        pass
    return result


def get_snapshot():
    """Get a full system snapshot combining all monitoring data.

    Returns:
        Dict with temps, memory, CMA, and timestamp.
    """
    data = {}
    data.update(get_thermal())
    data.update(get_memory_info())
    data.update(get_cma())
    data["device"] = "Jetson Orin Nano 8GB"
    data["timestamp"] = datetime.now().isoformat()
    return data
