#!/usr/bin/env python3
"""
jetson-monitor.py — Real-time Jetson Orin Nano system monitor.

Parses tegrastats output for CPU/GPU temps, frequencies, power, RAM.
Outputs JSON or formatted text.

Usage:
  python3 jetson-monitor.py           # One snapshot
  python3 jetson-monitor.py --loop    # Continuous (update every 1s)
  python3 jetson-monitor.py --json    # JSON output
  python3 jetson-monitor.py --stress  # Stress test (Ollama + monitor)
  python3 jetson-monitor.py --duration 60  # Monitor for N seconds
"""

import subprocess
import re
import json
import time
import sys
import os
import glob
from datetime import datetime

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
RESULTS_DIR = os.path.join(WORKSPACE, "memory", "gpu-benchmarks")


def parse_tegrastats_line(line):
    """Parse a single tegrastats output line."""
    data = {}

    # RAM
    m = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
    if m:
        data["ram_used_mb"] = int(m.group(1))
        data["ram_total_mb"] = int(m.group(2))

    # SWAP
    m = re.search(r'SWAP\s+(\d+)/(\d+)MB', line)
    if m:
        data["swap_used_mb"] = int(m.group(1))
        data["swap_total_mb"] = int(m.group(2))

    # CPU utilization per core
    m = re.search(r'CPU\s+\[([^\]]+)\]', line)
    if m:
        cores = m.group(1).split(',')
        data["cpu_cores"] = []
        for c in cores:
            parts = c.strip().split('@')
            data["cpu_cores"].append({
                "util_pct": int(parts[0].replace('%', '')),
                "freq_mhz": int(parts[1]) if len(parts) > 1 else 0,
            })

    # GPU frequency
    m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if m:
        data["gpu_freq_pct"] = int(m.group(1))

    # Temperatures
    for sensor in ["cpu", "gpu", "soc0", "soc1", "soc2", "tj"]:
        m = re.search(rf'{sensor}@([\d.]+)C', line)
        if m:
            data[f"temp_{sensor}_c"] = round(float(m.group(1)), 1)

    # Power
    for rail in ["VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"]:
        m = re.search(rf'{rail}\s+(\d+)mW', line)
        if m:
            data[f"power_{rail.lower()}_mw"] = int(m.group(1))

    return data


def read_tegrastats():
    """Read a single sample from tegrastats."""
    try:
        result = subprocess.run(
            ["timeout", "2", "tegrastats", "--interval", "1000"],
            capture_output=True, text=True, timeout=4
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            return parse_tegrastats_line(lines[-1])
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return {}


def read_thermal_zones():
    """Read thermal zones from /sys/class/thermal/ (no root needed)."""
    temps = {}
    for tz_path in sorted(glob.glob('/sys/class/thermal/thermal_zone*')):
        try:
            temp_raw = ""
            with open(os.path.join(tz_path, 'type'), 'rb') as f:
                raw = f.read()
            if not raw:
                continue
            name = raw.decode().strip()

            with open(os.path.join(tz_path, 'temp'), 'rb') as f:
                raw = f.read()
            if not raw:
                continue
            temp_raw = raw.decode().strip()
            if temp_raw and temp_raw != '0':
                temps[name] = round(int(temp_raw) / 1000.0, 1)
        except (OSError, ValueError):
            pass
    return temps


def read_cma():
    """Read CMA and general memory info from /proc/meminfo."""
    result = {}
    try:
        with open('/proc/meminfo', 'rb') as f:
            raw = f.read().decode()
        for line in raw.split('\n'):
            parts = line.split(':')
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            val = int(parts[1].strip().split()[0])  # kB
            if 'CmaTotal' in key:
                result['cma_total_kb'] = val
            elif 'CmaFree' in key:
                result['cma_free_kb'] = val
            elif key == 'MemTotal':
                result['ram_total_mb'] = val // 1024
            elif key == 'MemAvailable':
                result['ram_available_mb'] = val // 1024
            elif key == 'SwapTotal':
                result['swap_total_mb'] = val // 1024
            elif key == 'SwapFree':
                result['swap_free_mb'] = val // 1024
        if 'cma_total_kb' in result and 'cma_free_kb' in result:
            t = result['cma_total_kb']
            f = result['cma_free_kb']
            result['cma_used_pct'] = round((1 - f / t) * 100, 1)
        if 'ram_total_mb' in result and 'ram_available_mb' in result:
            result['ram_used_mb'] = result['ram_total_mb'] - result['ram_available_mb']
        if 'swap_total_mb' in result and 'swap_free_mb' in result:
            result['swap_used_mb'] = result['swap_total_mb'] - result['swap_free_mb']
    except Exception:
        pass
    return result


def get_snapshot():
    """Get a full system snapshot."""
    data = read_tegrastats()
    data.update(read_cma())
    # Use thermal zones as fallback if tegrastats didn't get temps
    if not data.get("temp_gpu_c"):
        temps = read_thermal_zones()
        if "gpu-thermal" in temps:
            data["temp_gpu_c"] = temps["gpu-thermal"]
        if "cpu-thermal" in temps:
            data["temp_cpu_c"] = temps["cpu-thermal"]
        if "tj-thermal" in temps:
            data["temp_tj_c"] = temps["tj-thermal"]
    data["timestamp"] = datetime.now().isoformat()
    return data


def stress_test(duration=60, model="deepseek-r1:1.5b"):
    """Run Ollama inference while monitoring GPU stats."""
    print(f"🔥 Jetson Stress Test — {duration}s with {model}")
    print("=" * 60)

    # Start Ollama inference in background
    ollama_proc = subprocess.Popen(
        ["ollama", "run", model, "Write a long detailed essay about the history of computing, covering at least 500 words."],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    samples = []
    start = time.time()
    peak_gpu_temp = 0
    peak_power = 0
    peak_cpu = 0

    try:
        while time.time() - start < duration:
            data = get_snapshot()
            samples.append(data)

            gpu_temp = data.get("temp_gpu_c", 0)
            power = data.get("power_vdd_in_mw", 0)
            cpu_max = max((c["util_pct"] for c in data.get("cpu_cores", [])), default=0)

            peak_gpu_temp = max(peak_gpu_temp, gpu_temp)
            peak_power = max(peak_power, power)
            peak_cpu = max(peak_cpu, cpu_max)

            elapsed = time.time() - start
            bar_len = 40
            filled = int(bar_len * elapsed / duration)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(f"\r  [{bar}] {elapsed:.0f}/{duration}s  "
                  f"GPU:{gpu_temp:.1f}°C  CPU:{cpu_max}%  "
                  f"Power:{power}mW  RAM:{data.get('ram_used_mb', '?')}/{data.get('ram_total_mb', '?')}MB",
                  end="", flush=True)

            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ollama_proc.terminate()
        ollama_proc.wait(timeout=5)

    print(f"\n\n📊 Stress Test Results ({duration}s)")
    print("-" * 40)
    print(f"  Peak GPU Temp:  {peak_gpu_temp:.1f}°C")
    print(f"  Peak Power:     {peak_power}mW ({peak_power/1000:.1f}W)")
    print(f"  Peak CPU Util:  {peak_cpu}%")
    print(f"  Samples:        {len(samples)}")

    # Check if thermal throttling occurred
    if peak_gpu_temp > 80:
        print(f"  ⚠️  THERMAL THROTTLING LIKELY (>80°C)")
    elif peak_gpu_temp > 60:
        print(f"  ⚠️  Elevated temps (>60°C)")
    else:
        print(f"  ✅ Thermal headroom OK")

    results = {
        "timestamp": datetime.now().isoformat(),
        "duration_s": duration,
        "model": model,
        "peak_gpu_temp_c": peak_gpu_temp,
        "peak_power_mw": peak_power,
        "peak_cpu_pct": peak_cpu,
        "samples": len(samples),
        "final_state": samples[-1] if samples else {},
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  💾 Saved to {path}")

    return results


def monitor_loop(duration=None, json_output=False):
    """Continuous monitoring."""
    start = time.time()
    samples = []

    try:
        while True:
            data = get_snapshot()
            samples.append(data)

            if json_output:
                print(json.dumps(data))
            else:
                gpu_temp = data.get("temp_gpu_c", "?")
                cpu_temp = data.get("temp_cpu_c", "?")
                power = data.get("power_vdd_in_mw", "?")
                ram = f"{data.get('ram_used_mb', '?')}/{data.get('ram_total_mb', '?')}MB"
                cma_free = f"{data.get('cma_free_kb', 0)//1024}MB"
                gpu_freq = f"{data.get('gpu_freq_pct', 0)}%"
                print(f"{datetime.now().strftime('%H:%M:%S')}  "
                      f"GPU:{gpu_temp}°C  CPU:{cpu_temp}°C  "
                      f"Pwr:{power}mW  RAM:{ram}  "
                      f"CMA:{cma_free}  GR3D:{gpu_freq}",
                      flush=True)

            if duration and time.time() - start >= duration:
                break
            time.sleep(1)

    except KeyboardInterrupt:
        pass

    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Jetson Orin Nano Monitor")
    parser.add_argument("--loop", action="store_true", help="Continuous monitoring")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--stress", action="store_true", help="Run stress test with Ollama")
    parser.add_argument("--duration", type=int, default=None, help="Duration in seconds")
    parser.add_argument("--model", type=str, default="deepseek-r1:1.5b", help="Ollama model for stress test")
    args = parser.parse_args()

    if args.stress:
        stress_test(duration=args.duration or 60, model=args.model)
    elif args.loop:
        monitor_loop(duration=args.duration, json_output=args.json)
    else:
        data = get_snapshot()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"🔧 Jetson Orin Nano — {data.get('timestamp', 'now')}")
            print(f"  GPU Temp:   {data.get('temp_gpu_c', '?')}°C")
            print(f"  CPU Temp:   {data.get('temp_cpu_c', '?')}°C")
            print(f"  SoC Temp:   {data.get('temp_tj_c', '?')}°C")
            print(f"  Power:      {data.get('power_vdd_in_mw', '?')}mW total")
            print(f"  RAM:        {data.get('ram_used_mb', '?')}/{data.get('ram_total_mb', '?')}MB")
            print(f"  Swap:       {data.get('swap_used_mb', '?')}/{data.get('swap_total_mb', '?')}MB")
            print(f"  CMA:        {data.get('cma_free_kb', 0)//1024}MB free / {data.get('cma_total_kb', 0)//1024}MB total ({data.get('cma_used_pct', '?')}% used)")
            print(f"  GPU Freq:   {data.get('gpu_freq_pct', 0)}%")
            cores = data.get("cpu_cores", [])
            if cores:
                utils = [f"{c['util_pct']}%@{c['freq_mhz']}MHz" for c in cores]
                print(f"  CPU Cores:  {'  '.join(utils)}")
