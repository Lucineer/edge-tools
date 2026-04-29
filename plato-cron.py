"""
plato-cron.py — Scheduled task system for Plato fleet.

Manages recurring tasks like health checks, bottle syncs,
tile indexing, and fleet pings. Cron-based scheduling.

Inspired by: GenericAgent (8K⭐) + fleet patterns from trending research

Usage:
  python3 plato-cron.py schedule <name> <interval> <command>
  python3 plato-cron.py list
  python3 plato-cron.py run [name]     # Run task(s) now
  python3 plato-cron.py remove <name>
  python3 plato-cron.py status         # Show last run times
  python3 plato-cron.py tick           # Process all due tasks
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
CRON_FILE = os.path.join(WORKSPACE, "memory", "plato-cron.json")


@dataclass
class CronTask:
    name: str
    command: str
    interval_min: int = 60  # minutes between runs
    last_run: str = ""  # ISO timestamp
    last_status: str = ""  # ok / error / skipped
    last_output: str = ""
    next_run: str = ""  # ISO timestamp
    enabled: bool = True
    created: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_due(self):
        """Check if task is due to run."""
        if not self.last_run or not self.enabled:
            return True
        last = datetime.fromisoformat(self.last_run)
        next_due = last + timedelta(minutes=self.interval_min)
        return datetime.now() >= next_due

    def mark_run(self, status, output=""):
        """Mark task as run."""
        now = datetime.now()
        self.last_run = now.isoformat()
        self.last_status = status
        self.last_output = output[:500]
        self.next_run = (now + timedelta(minutes=self.interval_min)).isoformat()


class CronManager:
    def __init__(self):
        self.tasks = {}
        self.load()

    def load(self):
        if os.path.exists(CRON_FILE):
            with open(CRON_FILE) as f:
                data = json.load(f)
            self.tasks = {k: CronTask.from_dict(v) for k, v in data.get("tasks", {}).items()}

    def save(self):
        os.makedirs(os.path.dirname(CRON_FILE), exist_ok=True)
        data = {
            "updated": datetime.now().isoformat(),
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
        }
        with open(CRON_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def schedule(self, name, interval, command):
        """Add or update a scheduled task."""
        now = datetime.now().isoformat()
        task = CronTask(
            name=name,
            command=command,
            interval_min=interval,
            next_run=(datetime.now() + timedelta(minutes=interval)).isoformat(),
            created=now,
        )
        self.tasks[name] = task
        self.save()
        return task

    def remove(self, name):
        """Remove a task."""
        if name in self.tasks:
            del self.tasks[name]
            self.save()
            return True
        return False

    def run(self, name=None):
        """Run task(s) now."""
        if name:
            if name not in self.tasks:
                return f"❌ Unknown task: {name}"
            return self._run_task(self.tasks[name])

        results = []
        for task in self.tasks.values():
            if task.enabled and task.is_due():
                result = self._run_task(task)
                results.append(result)
        self.save()

        if not results:
            return "No tasks due."
        return "\n".join(results)

    def _run_task(self, task):
        """Execute a single task."""
        start = time.time()
        try:
            result = subprocess.run(
                task.command, shell=True, capture_output=True, text=True, timeout=300
            )
            elapsed = time.time() - start
            if result.returncode == 0:
                task.mark_run("ok", result.stdout)
                return f"✅ {task.name} ({elapsed:.1f}s)\n{result.stdout[:200]}"
            else:
                task.mark_run("error", result.stderr)
                return f"❌ {task.name} ({elapsed:.1f}s)\n{result.stderr[:200]}"
        except subprocess.TimeoutExpired:
            task.mark_run("error", "TIMEOUT")
            return f"⏰ {task.name}: timed out after 300s"
        except Exception as e:
            task.mark_run("error", str(e))
            return f"❌ {task.name}: {e}"

    def tick(self):
        """Process all due tasks (for daemon mode)."""
        return self.run()

    def list(self):
        """List all scheduled tasks."""
        lines = ["⏰ Plato Cron — Scheduled Tasks", "=" * 50]
        for name, task in sorted(self.tasks.items()):
            status_icon = "🟢" if task.enabled else "🔴"
            due = "DUE" if task.is_due() else f"in {self._time_until(task.next_run)}"
            last = task.last_run[:16] if task.last_run else "never"
            last_s = f" [{task.last_status}]" if task.last_status else ""
            lines.append(f"  {status_icon} {name:25s} every {task.interval_min:4d}m  last: {last}{last_s}  next: {due}")

        if not self.tasks:
            lines.append("  (no tasks scheduled)")
        return "\n".join(lines)

    def status(self):
        """Show detailed status of all tasks."""
        lines = ["📊 Task Status Report", "=" * 50]
        for name, task in sorted(self.tasks.items()):
            lines.append(f"\n  📋 {name}")
            lines.append(f"     Command: {task.command[:60]}")
            lines.append(f"     Interval: {task.interval_min} minutes")
            lines.append(f"     Last run: {task.last_run or 'never'}")
            lines.append(f"     Last status: {task.last_status or 'pending'}")
            lines.append(f"     Next due: {task.next_run}")
            if task.last_output:
                lines.append(f"     Last output: {task.last_output[:100]}")
        return "\n".join(lines)

    def _time_until(self, iso_str):
        """Human-readable time until a timestamp."""
        try:
            target = datetime.fromisoformat(iso_str)
            diff = target - datetime.now()
            if diff.total_seconds() < 0:
                return "OVERDUE"
            minutes = int(diff.total_seconds() / 60)
            if minutes < 60:
                return f"{minutes}m"
            hours = minutes // 60
            if hours < 24:
                return f"{hours}h {minutes % 60}m"
            return f"{hours // 24}d {hours % 24}h"
        except:
            return "unknown"

    def seed_defaults(self):
        """Seed with recommended default tasks."""
        defaults = [
            ("fleet-health", 30, "python3 ~/.openclaw/workspace/tools/fleet-health.py report"),
            ("tile-graph-build", 60, "python3 ~/.openclaw/workspace/tools/tile-graph.py build"),
            ("git-push", 15, "cd ~/.openclaw/workspace && git add -A && git commit -m 'cron: auto-push' --allow-empty && git push 2>&1"),
            ("memory-compact", 120, "cd ~/.openclaw/workspace && find memory/ -name '*.md' -size +100k -exec truncate -s 100k {} \\; 2>/dev/null; echo 'done'"),
            ("evennia-check", 10, "cd /home/lucineer/plato-jetson && ~/.local/bin/evennia status 2>&1 | head -3"),
        ]
        for name, interval, cmd in defaults:
            if name not in self.tasks:
                self.schedule(name, interval, cmd)
                print(f"  ⏰ Scheduled: {name} (every {interval}m)")
            else:
                print(f"  ⏭️  Already exists: {name}")
        print(f"\n  {len(defaults)} tasks ready")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cron = CronManager()
    cmd = sys.argv[1]

    if cmd == "schedule":
        if len(sys.argv) < 5:
            print("Usage: schedule <name> <interval_min> <command>")
            return
        name = sys.argv[2]
        interval = int(sys.argv[3])
        command = " ".join(sys.argv[4:])
        task = cron.schedule(name, interval, command)
        print(f"✅ Scheduled: {name} every {interval}m")
        print(f"   Command: {command}")
        print(f"   Next run: {task.next_run}")

    elif cmd == "remove":
        name = sys.argv[2] if len(sys.argv) > 2 else ""
        if cron.remove(name):
            print(f"🗑️ Removed: {name}")
        else:
            print(f"❌ Not found: {name}")

    elif cmd == "run":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        print(cron.run(name))

    elif cmd == "tick":
        print(cron.tick())

    elif cmd == "list":
        print(cron.list())

    elif cmd == "status":
        print(cron.status())

    elif cmd == "seed":
        print("🌱 Seeding default fleet tasks:")
        cron.seed_defaults()

    elif cmd == "daemon":
        """Run as a simple daemon — tick every minute."""
        print(f"🔄 Plato Cron daemon starting... (Ctrl+C to stop)")
        try:
            while True:
                output = cron.tick()
                if output and "No tasks due" not in output:
                    print(f"[{datetime.now().strftime('%H:%M')}] {output[:200]}")
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nDaemon stopped.")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
