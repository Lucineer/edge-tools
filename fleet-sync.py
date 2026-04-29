#!/usr/bin/env python3
"""
fleet-sync.py — Automatic fleet bottle synchronization.

Pulls bottles from Forgemaster and Oracle1, indexes them, and sends responses.
Part of the JC1 edge toolkit.

Usage:
  python3 fleet-sync.py inbox           # Check for new bottles addressed to JC1
  python3 fleet-sync.py send <message>  # Send message to Oracle1
  python3 fleet-sync.py status          # Show fleet connection status
  python3 fleet-sync.py digest          # Summarize all recent fleet messages
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")

# Fleet endpoints
FM_REPO = "/tmp/forgemaster"
ORACLE1_SHELL = "http://147.224.38.131:8848"
JC1_AGENT = "jc1"


def read_bottle(path):
    """Read a bottle markdown file and extract metadata."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if not raw:
            return None
        text = raw.decode("utf-8", errors="replace")
        lines = text.strip().split("\n")

        meta = {"path": path, "filename": os.path.basename(path), "size": len(raw)}
        meta["lines"] = len(lines)

        # Extract sender/from
        for line in lines:
            if line.startswith("## From:") or line.startswith("## from:"):
                meta["from"] = line.split(":", 1)[1].strip()
            elif line.startswith("## Date:") or line.startswith("## date:"):
                meta["date"] = line.split(":", 1)[1].strip()
            elif line.startswith("## Status:") or line.startswith("[I2I:"):
                meta["status"] = line.strip()

        meta["preview"] = lines[0][:120] if lines else "(empty)"
        meta["text"] = text
        return meta
    except Exception as e:
        return {"path": path, "error": str(e)}


def check_forgemaster():
    """Pull and check Forgemaster bottles."""
    print("⚒️  Forgemaster")

    # Pull latest
    try:
        subprocess.run(["git", "pull", "-q"], cwd=FM_REPO, capture_output=True, timeout=15)
        print("   ✅ Repository updated")
    except Exception as e:
        print(f"   ⚠️  Pull failed: {e}")

    # Find bottles addressed to JC1
    inbox_dir = os.path.join(FM_REPO, "for-fleet")
    if not os.path.exists(inbox_dir):
        print("   ⚠️  No inbox directory")
        return []

    bottles = []
    for fname in sorted(os.listdir(inbox_dir)):
        if "TO-JC1" in fname.upper() or "TO-JETSONCLAW1" in fname.upper():
            path = os.path.join(inbox_dir, fname)
            bottle = read_bottle(path)
            if bottle:
                bottles.append(bottle)
                age = ""
                if bottle.get("date"):
                    age = f" ({bottle['date']})"
                print(f"   📨 {fname}{age}")
                print(f"      {bottle['preview']}")

    if not bottles:
        print("   (no new bottles)")
    return bottles


def check_oracle1():
    """Check Oracle1 connection and recent messages."""
    print(f"🌐 Oracle1 ({ORACLE1_SHELL})")

    try:
        # Check connection
        resp = subprocess.run(
            ["curl", "-s", "-m", "5", f"{ORACLE1_SHELL}/connect?agent={JC1_AGENT}"],
            capture_output=True, text=True, timeout=10
        )
        if resp.returncode != 0:
            print("   ❌ Connection failed")
            return None

        data = json.loads(resp.stdout)
        print(f"   ✅ Connected — room: {data.get('room', '?')}")
        print(f"   Rooms: {', '.join(data.get('rooms', []))}")
        return data

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def send_to_oracle1(message):
    """Send a message to Oracle1 via the shell endpoint."""
    print(f"📤 Sending to Oracle1...")

    try:
        resp = subprocess.run(
            ["curl", "-s", "-m", "10", "-X", "POST",
             f"{ORACLE1_SHELL}/cmd/shell",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"agent": JC1_AGENT, "command": f"echo {message}"})],
            capture_output=True, text=True, timeout=15
        )

        if resp.returncode != 0:
            print(f"   ❌ Failed: {resp.stderr}")
            return False

        data = json.loads(resp.stdout)
        if data.get("exit_code") == 0:
            print(f"   ✅ Sent (container: {data.get('container', '?')})")
            return True
        else:
            print(f"   ❌ Error: {data.get('stderr', '?')}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def show_status():
    """Show full fleet status."""
    print("📡 Fleet Status — JetsonClaw1\n")

    # Forgemaster
    fm_exists = os.path.exists(FM_REPO)
    print(f"⚒️  Forgemaster: {'cloned' if fm_exists else 'not found'}")
    if fm_exists:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-3"],
                cwd=FM_REPO, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                commits = result.stdout.strip().split("\n")
                print("   Recent:")
                for c in commits:
                    print(f"   {c}")
        except:
            pass

    # Oracle1
    print(f"\n🌐 Oracle1: {ORACLE1_SHELL}")
    try:
        resp = subprocess.run(
            ["curl", "-s", "-m", "5", f"{ORACLE1_SHELL}/connect?agent={JC1_AGENT}"],
            capture_output=True, text=True, timeout=10
        )
        if resp.returncode == 0:
            data = json.loads(resp.stdout)
            print(f"   ✅ Connected — room: {data.get('room')}")
        else:
            print("   ❌ Unreachable")
    except:
        print("   ❌ Unreachable")

    # Local Plato
    plato_dir = "/home/lucineer/plato-jetson"
    print(f"\n🏛️  Local Plato (Evennia): {plato_dir}")
    if os.path.exists(plato_dir):
        try:
            result = subprocess.run(
                ["evennia", "status"],
                cwd=plato_dir, capture_output=True, text=True, timeout=5
            )
            print(f"   {result.stdout.strip()}")
        except:
            print("   (status check failed)")

    # System
    print(f"\n🔧 System:")
    try:
        with open("/sys/class/thermal/thermal_zone1/temp", "rb") as f:
            gpu_temp = int(f.read()) / 1000
        print(f"   GPU: {gpu_temp:.1f}°C")
    except:
        pass
    try:
        with open("/proc/meminfo", "rb") as f:
            for line in f.read().decode().split("\n"):
                if "CmaFree" in line:
                    cma_free = int(line.split()[1]) // 1024
                elif "CmaTotal" in line:
                    cma_total = int(line.split()[1]) // 1024
        print(f"   CMA: {cma_free}MB / {cma_total}MB")
    except:
        pass


def digest_fleet():
    """Create a digest of all fleet activity."""
    print("📋 Fleet Digest\n")
    print("=" * 50)

    # Count FM bottles
    inbox_dir = os.path.join(FM_REPO, "for-fleet")
    if os.path.exists(inbox_dir):
        all_bottles = [f for f in os.listdir(inbox_dir) if f.startswith("BOTTLE")]
        to_jc1 = [f for f in all_bottles if "TO-JC1" in f.upper() or "TO-JETSONCLAW1" in f.upper()]
        from_jc1 = [f for f in all_bottles if "FROM-JC1" in f.upper() or "FROM-JETSONCLAW1" in f.upper()]
        print(f"⚒️  Forgemaster:")
        print(f"   Total bottles: {len(all_bottles)}")
        print(f"   To JC1: {len(to_jc1)}")
        print(f"   From JC1: {len(from_jc1)}")

        # Show recent TO-JC1
        if to_jc1:
            print(f"   Latest inbox:")
            for fname in sorted(to_jc1)[-3:]:
                path = os.path.join(inbox_dir, fname)
                bottle = read_bottle(path)
                if bottle:
                    print(f"   📨 {fname}")
                    print(f"      {bottle.get('preview', '')}")

    # Oracle1 status
    print(f"\n🌐 Oracle1: {ORACLE1_SHELL}")
    try:
        resp = subprocess.run(
            ["curl", "-s", "-m", "5", f"{ORACLE1_SHELL}/connect?agent={JC1_AGENT}"],
            capture_output=True, text=True, timeout=10
        )
        if resp.returncode == 0:
            data = json.loads(resp.stdout)
            print(f"   ✅ Connected — room: {data.get('room')}")
    except:
        print("   ❌ Unreachable")

    print(f"\n{'=' * 50}")
    print(f"Generated: {datetime.now().isoformat()}")


def main():
    if len(sys.argv) < 2:
        print("fleet-sync.py — Fleet bottle synchronization")
        print("Usage: fleet-sync.py [inbox|send|status|digest]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "inbox":
        check_forgemaster()
        print()
        check_oracle1()

    elif cmd == "send":
        if len(sys.argv) < 3:
            print("Usage: fleet-sync.py send <message>")
            sys.exit(1)
        message = " ".join(sys.argv[2:])
        send_to_oracle1(message)

    elif cmd == "status":
        show_status()

    elif cmd == "digest":
        digest_fleet()

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
