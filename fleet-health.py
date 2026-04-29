"""
fleet-health.py — Fleet health monitoring for the JC1 vessel.

Checks system health, service status, disk space, memory usage,
Plato status, and reports to HEARTBEAT.md.

Usage:
  python3 fleet-health.py              # Full health check
  python3 fleet-health.py services     # Service status only
  python3 fleet-health.py hardware     # Hardware metrics only
  python3 fleet-health.py plato        # Plato MUD status
  python3 fleet-health.py report       # Write report to HEARTBEAT.md
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path


WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
HEARTBEAT = os.path.join(WORKSPACE, "HEARTBEAT.md")


def run_cmd(cmd, timeout=10):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 1
    except Exception as e:
        return str(e), 1


def check_hardware():
    """Check Jetson hardware metrics."""
    metrics = {}

    # CPU temp
    out, _ = run_cmd("cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | tail -1")
    if out and out != "TIMEOUT":
        temp_c = int(out) / 1000
        metrics["cpu_temp_c"] = round(temp_c, 1)
        metrics["thermal_status"] = "OK" if temp_c < 60 else ("WARN" if temp_c < 80 else "CRIT")

    # Memory
    out, _ = run_cmd("free -m | grep Mem")
    if out:
        parts = out.split()
        total = int(parts[1])
        used = int(parts[2])
        available = int(parts[6])
        metrics["ram_total_mb"] = total
        metrics["ram_used_mb"] = used
        metrics["ram_available_mb"] = available
        metrics["ram_pct"] = round(used / total * 100, 1)

    # Disk
    out, _ = run_cmd("df -h / | tail -1")
    if out:
        parts = out.split()
        metrics["disk_total"] = parts[1]
        metrics["disk_used"] = parts[2]
        metrics["disk_pct"] = parts[4].replace("%", "")

    # Uptime
    out, _ = run_cmd("uptime -p")
    metrics["uptime"] = out

    # GPU
    out, _ = run_cmd("tegrastats --interval 1000 --stop 2>/dev/null | tail -1")
    if out and "TIMEOUT" not in out and out:
        metrics["gpu_stats"] = out.strip()

    return metrics


def check_services():
    """Check systemd user services."""
    services = {}
    service_list = [
        "openclaw-gateway",
        "evennia-plato",
        "plato",
        "hardware-mud",
    ]

    for svc in service_list:
        out, rc = run_cmd(f"systemctl --user is-active {svc}.service 2>/dev/null")
        if rc == 0:
            services[svc] = "running"
        else:
            services[svc] = "stopped"

    return services


def check_plato():
    """Check Plato Evennia MUD status."""
    plato = {}

    # Evennia status
    out, _ = run_cmd("cd /home/lucineer/plato-jetson && /home/lucineer/.local/bin/evennia status 2>&1")
    if "NOT RUNNING" in out:
        plato["evennia"] = "stopped"
    elif "running" in out.lower():
        plato["evennia"] = "running"
    else:
        plato["evennia"] = "unknown"
        plato["status_output"] = out[:200]

    # Port check
    for port, name in [(4000, "telnet"), (4001, "web")]:
        out, _ = run_cmd(f"ss -tlnp | grep {port}")
        plato[f"{name}_port"] = "open" if str(port) in out else "closed"

    # Tile count
    tiles_dir = os.path.join(WORKSPACE, "memory", "tiles")
    tiles = list(Path(tiles_dir).glob("*.md")) if os.path.exists(tiles_dir) else []
    plato["tile_count"] = len(tiles)

    # Skill tree
    skills_file = os.path.join(WORKSPACE, "memory", "skills", "_tree_state.json")
    if os.path.exists(skills_file):
        with open(skills_file) as f:
            state = json.load(f)
        plato["skill_count"] = len(state.get("skills", {}))

    return plato


def check_git():
    """Check git repos status."""
    git_info = {}

    # Main workspace
    out, _ = run_cmd(f"cd {WORKSPACE} && git log --oneline -1 2>&1")
    git_info["workspace_last_commit"] = out.split("\n")[0] if out else "error"

    out, _ = run_cmd(f"cd {WORKSPACE} && git status --short 2>&1 | wc -l")
    git_info["workspace_uncommitted"] = out.strip()

    # Branch
    out, _ = run_cmd(f"cd {WORKSPACE} && git branch --show-current 2>&1")
    git_info["workspace_branch"] = out.strip()

    return git_info


def check_network():
    """Check network connectivity."""
    net = {}

    # Gateway
    out, _ = run_cmd("ss -tlnp | grep 18789")
    net["gateway_port"] = "open" if "18789" in out else "closed"

    # Internet
    out, rc = run_cmd("curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 https://api.github.com 2>&1")
    net["github_api"] = out if rc == 0 else "unreachable"

    return net


def full_check():
    """Run all health checks."""
    return {
        "timestamp": datetime.now().isoformat(),
        "hardware": check_hardware(),
        "services": check_services(),
        "plato": check_plato(),
        "git": check_git(),
        "network": check_network(),
    }


def format_report(check):
    """Format health check as readable report."""
    lines = [
        f"🏥 JC1 Fleet Health — {check['timestamp']}",
        "=" * 50,
    ]

    # Hardware
    h = check["hardware"]
    lines.append("\n🔧 Hardware")
    temp = h.get("cpu_temp_c", "?")
    lines.append(f"  CPU Temp: {temp}°C [{h.get('thermal_status', '?')}]")
    if "ram_used_mb" in h:
        lines.append(f"  RAM: {h['ram_used_mb']}MB / {h['ram_total_mb']}MB ({h['ram_pct']}%)")
    if "disk_used" in h:
        lines.append(f"  Disk: {h['disk_used']} / {h['disk_total']} ({h['disk_pct']}%)")
    lines.append(f"  Uptime: {h.get('uptime', '?')}")

    # Services
    lines.append("\n⚙️  Services")
    for svc, status in check["services"].items():
        icon = "✅" if status == "running" else "❌"
        lines.append(f"  {icon} {svc}: {status}")

    # Plato
    lines.append("\n🏛️  Plato")
    p = check["plato"]
    icon = "✅" if p.get("evennia") == "running" else "❌"
    lines.append(f"  {icon} Evennia: {p.get('evennia', '?')}")
    lines.append(f"  Telnet (4000): {p.get('telnet_port', '?')}")
    lines.append(f"  Web (4001): {p.get('web_port', '?')}")
    lines.append(f"  Tiles: {p.get('tile_count', 0)}")
    lines.append(f"  Skills: {p.get('skill_count', 0)}")

    # Git
    lines.append("\n📂 Git")
    g = check["git"]
    lines.append(f"  Last: {g.get('workspace_last_commit', '?')}")
    lines.append(f"  Branch: {g.get('workspace_branch', '?')}")
    lines.append(f"  Uncommitted: {g.get('workspace_uncommitted', '?')} files")

    # Network
    lines.append("\n🌐 Network")
    n = check["network"]
    icon = "✅" if n.get("gateway_port") == "open" else "❌"
    lines.append(f"  {icon} Gateway (18789): {n.get('gateway_port', '?')}")
    lines.append(f"  GitHub API: {n.get('github_api', '?')}")

    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"

    if cmd == "check" or cmd == "full":
        check = full_check()
        print(format_report(check))

    elif cmd == "services":
        for svc, status in check_services().items():
            icon = "✅" if status == "running" else "❌"
            print(f"  {icon} {svc}: {status}")

    elif cmd == "hardware":
        h = check_hardware()
        for k, v in h.items():
            print(f"  {k}: {v}")

    elif cmd == "plato":
        p = check_plato()
        for k, v in p.items():
            print(f"  {k}: {v}")

    elif cmd == "report":
        check = full_check()
        report = format_report(check)
        print(report)
        # Append to heartbeat
        with open(HEARTBEAT, "a") as f:
            f.write(f"\n\n### Health Report ({check['timestamp']})\n```\n{report}\n```\n")

    elif cmd == "json":
        print(json.dumps(full_check(), indent=2))

    else:
        print(__doc__)
