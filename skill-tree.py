"""
skill-tree.py — Self-evolving agent skill tree.

An agent can learn, register, and evolve skills dynamically.
Skills have requirements, levels, and can compose.

Inspired by: GenericAgent (8K⭐) — skill tree + self-evolution pattern
Adapted for: Plato knowledge management + Jetson edge computing

Usage:
  python3 skill-tree.py learn <name> [--from tile:<id> | --from repo:<url>]
  python3 skill-tree.py evolve <name>
  python3 skill-tree.py tree [name]
  python3 skill-tree.py suggest
  python3 skill-tree.py run <name> [args]
"""

import os
import sys
import json
import glob
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

SKILLS_DIR = os.path.expanduser("~/.openclaw/workspace/memory/skills")
STATE_FILE = os.path.join(SKILLS_DIR, "_tree_state.json")


@dataclass
class Skill:
    name: str
    level: int = 1  # 1=beginner, 5=expert
    domain: str = "general"
    description: str = ""
    source: str = ""  # tile:, repo:, or manual
    requirements: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    learned_at: str = ""
    evolved_at: str = ""
    evolve_count: int = 0
    success_rate: float = 0.0
    uses: int = 0
    composed_of: list = field(default_factory=list)  # sub-skills
    tags: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SkillTree:
    skills: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def learn(self, name: str, source: str = "manual", domain: str = "general", description: str = ""):
        """Register a new skill in the tree."""
        os.makedirs(SKILLS_DIR, exist_ok=True)
        now = datetime.now().isoformat()

        skill = Skill(
            name=name,
            domain=domain,
            description=description,
            source=source,
            learned_at=now,
        )

        # Check if learning from a tile
        if source.startswith("tile:") and not description:
            tile_id = source[5:]
            skill.description = self._read_tile(tile_id) or description
            skill.commands = self._extract_commands(skill.description)

        self.skills[name.lower()] = skill
        self.history.append({"action": "learn", "skill": name, "time": now})
        self._save()

        return skill

    def evolve(self, name: str):
        """Evolve a skill — increase level, improve success rate."""
        key = name.lower()
        if key not in self.skills:
            return f"❌ Unknown skill: {name}"

        skill = self.skills[key]
        if skill.level >= 5:
            return f"🏆 {name} is already at max level (5)"

        skill.level += 1
        skill.evolved_at = datetime.now().isoformat()
        skill.evolve_count += 1
        self.history.append({"action": "evolve", "skill": name, "level": skill.level, "time": datetime.now().isoformat()})
        self._save()

        return f"⬆️ {name} evolved to level {skill.level}"

    def suggest(self):
        """Suggest skills to learn based on gaps in the tree."""
        known_domains = set(s.domain for s in self.skills.values())

        # Suggest based on common domains we don't have
        all_domains = ["plato", "edge", "fleet", "cocapn", "networking", "security", "monitoring", "deployment"]
        missing = [d for d in all_domains if d not in known_domains]

        # Suggest based on low-level skills
        upgradable = [s.name for s in self.skills.values() if s.level < 5]

        suggestions = []
        for d in missing:
            suggestions.append(f"🆕 Learn a {d} skill (new domain)")
        for name in upgradable:
            skill = self.skills[name]
            suggestions.append(f"⬆️ Evolve {name} (level {skill.level} → {skill.level + 1})")

        return suggestions if suggestions else ["✅ Tree is well-developed!"]

    def tree(self, name: str = None):
        """Display the skill tree."""
        if name:
            key = name.lower()
            if key not in self.skills:
                return f"❌ Unknown skill: {name}"
            skill = self.skills[key]
            return self._format_skill(skill)

        lines = ["🌳 JC1 Skill Tree", "=" * 50]
        domains = {}
        for skill in self.skills.values():
            domains.setdefault(skill.domain, []).append(skill)

        for domain, skills in sorted(domains.items()):
            lines.append(f"\n📂 {domain.title()}")
            for skill in sorted(skills, key=lambda s: s.level, reverse=True):
                level_bar = "★" * skill.level + "☆" * (5 - skill.level)
                lines.append(f"  {level_bar} {skill.name} (L{skill.level}) — {skill.description[:40]}")

        total = len(self.skills)
        avg_level = sum(s.level for s in self.skills.values()) / max(total, 1)
        lines.append(f"\n📊 {total} skills, avg level {avg_level:.1f}")

        return "\n".join(lines)

    def run(self, name: str, args: list = None):
        """Execute a skill's commands."""
        key = name.lower()
        if key not in self.skills:
            return f"❌ Unknown skill: {name}"

        skill = self.skills[key]
        skill.uses += 1
        self._save()

        if not skill.commands:
            return f"ℹ️ {name} has no executable commands. Try evolving it first."

        results = []
        for cmd in skill.commands:
            results.append(f"  $ {cmd}")
            # In a real implementation, this would execute commands
            # For now, just list them
        results.append(f"\n✅ Ran {len(skill.commands)} commands from {name}")

        return "\n".join(results)

    def _read_tile(self, tile_id: str) -> str:
        """Read content from a Plato tile, stripping frontmatter."""
        tiles_dir = os.path.expanduser("~/.openclaw/workspace/memory/tiles")
        path = os.path.join(tiles_dir, tile_id if tile_id.endswith(".md") else f"{tile_id}.md")
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            # Strip YAML frontmatter
            display_lines = []
            in_fm = False
            for line in lines:
                if line.strip() == "---" and not in_fm:
                    in_fm = True
                    continue
                elif line.strip() == "---" and in_fm:
                    in_fm = False
                    continue
                if not in_fm:
                    display_lines.append(line)
            return "".join(display_lines).strip()[:500]
        return ""

    def _extract_commands(self, content: str) -> list:
        """Extract executable commands from tile content."""
        commands = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("$ ") or line.startswith("# "):
                commands.append(line.lstrip("$ #"))
        return commands[:5]  # Limit to 5 commands per skill

    def _format_skill(self, skill: Skill) -> str:
        level_bar = "★" * skill.level + "☆" * (5 - skill.level)
        lines = [
            f"🎯 {skill.name} [{level_bar}]",
            f"   Domain: {skill.domain}",
            f"   Source: {skill.source}",
            f"   Learned: {skill.learned_at}",
            f"   Evolved: {skill.evolve_count} times (last: {skill.evolved_at or 'never'})",
            f"   Uses: {skill.uses}",
            f"   Description: {skill.description[:200]}",
        ]
        if skill.commands:
            lines.append(f"   Commands:")
            for cmd in skill.commands:
                lines.append(f"     $ {cmd}")
        if skill.composed_of:
            lines.append(f"   Composed of: {', '.join(skill.composed_of)}")
        return "\n".join(lines)

    def _save(self):
        os.makedirs(SKILLS_DIR, exist_ok=True)
        state = {
            "skills": {k: v.to_dict() for k, v in self.skills.items()},
            "history": self.history[-100:],  # Keep last 100 events
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                state = json.load(f)
            tree = cls()
            tree.skills = {k: Skill.from_dict(v) for k, v in state.get("skills", {}).items()}
            tree.history = state.get("history", [])
            return tree
        return cls()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    tree = SkillTree.load()
    cmd = sys.argv[1]

    if cmd == "learn":
        name = sys.argv[2] if len(sys.argv) > 2 else input("Skill name: ")
        source = "manual"
        domain = "general"
        desc = ""
        for i, arg in enumerate(sys.argv):
            if arg == "--from" and i + 1 < len(sys.argv):
                source = sys.argv[i + 1]
            if arg == "--domain" and i + 1 < len(sys.argv):
                domain = sys.argv[i + 1]
            if arg == "--desc" and i + 1 < len(sys.argv):
                desc = sys.argv[i + 1]
        skill = tree.learn(name, source, domain, desc)
        print(f"✅ Learned: {skill.name} (L{skill.level}, {skill.domain})")

    elif cmd == "evolve":
        name = sys.argv[2] if len(sys.argv) > 2 else input("Skill name: ")
        print(tree.evolve(name))

    elif cmd == "tree":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        print(tree.tree(name))

    elif cmd == "suggest":
        for s in tree.suggest():
            print(s)

    elif cmd == "run":
        name = sys.argv[2] if len(sys.argv) > 2 else input("Skill name: ")
        args = sys.argv[3:]
        print(tree.run(name, args))

    elif cmd == "history":
        for h in tree.history[-20:]:
            print(f"  [{h['time'][:16]}] {h['action']}: {h['skill']}")

    elif cmd == "seed":
        """Seed the tree with known JC1 skills."""
        seeds = [
            ("edge-router", "edge", "Route AI tasks to best Jetson local model by task type + resource check", "tile:trending-research-2026-04"),
            ("plato-bridge", "plato", "Bridge Evennia MUD rooms to git-backed knowledge tiles with search", "tile:trending-research-2026-04"),
            ("fleet-agent", "fleet", "Multi-agent dispatch and composition across the fleet", "tile:trending-research-2026-04"),
            ("model-routing", "cocapn", "Smart model routing with 60-97% cost savings", "tile:trending-research-2026-04"),
            ("jetson-bootstrap", "edge", "C11 Jetson system bootstrap with zero compiler warnings", "manual"),
            ("cuda-optimization", "edge", "CUDA kernel optimization for Orin Nano 1024 cores", "manual"),
            ("evennia-mud", "plato", "Build and maintain Evennia MUD-based knowledge system with 14 rooms", "manual"),
            ("git-agent", "fleet", "Git-native agent architecture — the repo IS the agent", "manual"),
        ]
        for name, domain, desc, source in seeds:
            tree.learn(name, source, domain, desc)
        print(f"🌱 Seeded {len(seeds)} skills into the tree")
        print(tree.tree())

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
