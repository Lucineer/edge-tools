"""
tile-graph.py — Lightweight graph knowledge index for Plato tiles.

Builds a relationship graph between tiles based on shared tags,
content references, and explicit links. Enables fast traversal.

Inspired by: FalkorDB (4.3K⭐, C-based graph DB) + codebase-memory-mcp (1.9K⭐)
Adapted for: Plato knowledge tiles — no external DB needed

Usage:
  python3 tile-graph.py build          # Build graph from all tiles
  python3 tile-graph.py search <query> # Graph-aware search
  python3 tile-graph.py related <name> # Find related tiles
  python3 tile-graph.py graph [name]   # Show adjacency graph
  python3 tile-graph.py clusters       # Show tile clusters/groups
"""

import os
import re
import json
import glob
from collections import defaultdict
from datetime import datetime


TILES_DIR = os.path.expanduser("~/.openclaw/workspace/memory/tiles")
GRAPH_FILE = os.path.expanduser("~/.openclaw/workspace/memory/tiles/_graph.json")


def parse_frontmatter(content):
    """Extract YAML frontmatter from a tile."""
    fm = {}
    in_fm = False
    fm_lines = []
    for line in content.split("\n"):
        if line.strip() == "---" and not in_fm:
            in_fm = True
            continue
        elif line.strip() == "---" and in_fm:
            in_fm = False
            break
        if in_fm:
            fm_lines.append(line)
    for line in fm_lines:
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # Parse lists like [tag1, tag2]
            if val.startswith("[") and val.endswith("]"):
                val = [t.strip() for t in val[1:-1].split(",") if t.strip()]
            fm[key] = val
    return fm


def build_graph():
    """Build a relationship graph from all tiles."""
    tiles = glob.glob(os.path.join(TILES_DIR, "*.md"))
    nodes = {}
    edges = []

    for path in tiles:
        name = os.path.basename(path)
        with open(path) as f:
            content = f.read()

        fm = parse_frontmatter(content)
        body = content.split("---", 2)[-1].strip() if content.count("---") >= 2 else content

        nodes[name] = {
            "id": fm.get("id", name.replace(".md", "")),
            "domain": fm.get("domain", "general"),
            "tags": fm.get("tags", []) if isinstance(fm.get("tags"), list) else [],
            "created": fm.get("created", ""),
            "updated": fm.get("updated", ""),
            "title": body.split("\n")[0].strip("# ").strip() if body else name,
            "word_count": len(body.split()),
            "path": path,
        }

    # Build edges: shared tags
    tag_to_tiles = defaultdict(list)
    for name, node in nodes.items():
        for tag in node["tags"]:
            tag_to_tiles[tag].append(name)

    seen_edges = set()
    for tag, tile_names in tag_to_tiles.items():
        for i in range(len(tile_names)):
            for j in range(i + 1, len(tile_names)):
                edge = tuple(sorted([tile_names[i], tile_names[j]]))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append({
                        "source": edge[0],
                        "target": edge[1],
                        "type": "shared_tag",
                        "tag": tag,
                        "weight": 1,
                    })

    # Build edges: content references (mentions of other tile names)
    for name, node in nodes.items():
        with open(node["path"]) as f:
            content = f.read().lower()
        for other_name in nodes:
            if other_name != name:
                other_id = nodes[other_name]["id"]
                if other_id and other_id in content:
                    edge = tuple(sorted([name, other_name]))
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        edges.append({
                            "source": edge[0],
                            "target": edge[1],
                            "type": "content_reference",
                            "weight": 2,
                        })

    graph = {
        "built_at": datetime.now().isoformat(),
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "tiles": len(nodes),
            "connections": len(edges),
            "tags": len(tag_to_tiles),
        }
    }

    with open(GRAPH_FILE, "w") as f:
        json.dump(graph, f, indent=2)

    return graph


def load_graph():
    """Load the graph from disk."""
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE) as f:
            return json.load(f)
    return None


def search_graph(query, graph=None):
    """Graph-aware search: find tiles matching query + their neighbors."""
    if not graph:
        graph = load_graph()
    if not graph:
        return "No graph built. Run 'build' first."

    query = query.lower()
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Direct matches
    direct = []
    for name, node in nodes.items():
        if (query in node["title"].lower() or
            query in name.lower() or
            any(query in str(t).lower() for t in node["tags"]) or
            any(query in str(t).lower() for t in [node["domain"]])):
            direct.append(name)

    # Neighbor matches (connected to direct matches)
    neighbors = set()
    for edge in edges:
        if edge["source"] in direct:
            neighbors.add(edge["target"])
        elif edge["target"] in direct:
            neighbors.add(edge["source"])

    # Sort: direct first, then neighbors by connection weight
    def score(name):
        s = 0
        if name in direct:
            s += 10
        for e in edges:
            if (e["source"] == name or e["target"] == name):
                if name in direct or e["source"] in direct or e["target"] in direct:
                    s += e["weight"]
        return s

    all_results = list(set(direct) | neighbors)
    all_results.sort(key=score, reverse=True)

    lines = [f"🔍 Graph search: '{query}' — {len(all_results)} results\n{'='*50}"]
    for name in all_results[:15]:
        node = nodes[name]
        marker = "●" if name in direct else "○"
        conn_count = sum(1 for e in edges if e["source"] == name or e["target"] == name)
        lines.append(f"  {marker} {name:40s} [{', '.join(node['tags'][:3])}] ({conn_count} connections)")

    return "\n".join(lines)


def related(name, graph=None):
    """Find tiles related to a given tile."""
    if not graph:
        graph = load_graph()
    if not graph:
        return "No graph built."

    nodes = graph["nodes"]
    edges = graph["edges"]

    # Fuzzy match name
    matches = [n for n in nodes if name.lower() in n.lower()]
    if not matches:
        return f"No tile found matching '{name}'"

    target = matches[0]
    neighbors = []
    for edge in edges:
        if edge["source"] == target:
            neighbor = edge["target"]
            neighbors.append((neighbor, edge["type"], edge.get("tag", "")))
        elif edge["target"] == target:
            neighbor = edge["source"]
            neighbors.append((neighbor, edge["type"], edge.get("tag", "")))

    if not neighbors:
        return f"🔗 {target} has no connections"

    lines = [f"🔗 Related to {target} ({len(neighbors)} connections):\n"]
    for neighbor, etype, tag in sorted(neighbors, key=lambda x: x[0]):
        node = nodes[neighbor]
        reason = f"via {tag}" if tag else f"via {etype}"
        lines.append(f"  → {neighbor:40s} {reason}")
        lines.append(f"    {node['title'][:60]}")

    return "\n".join(lines)


def show_graph(name=None, graph=None):
    """Display the tile graph as ASCII art."""
    if not graph:
        graph = load_graph()
    if not graph:
        return "No graph built."

    nodes = graph["nodes"]
    edges = graph["edges"]

    if name:
        return related(name, graph)

    lines = [f"🕸️  Plato Knowledge Graph — {graph['stats']['tiles']} tiles, {graph['stats']['connections']} edges\n{'='*50}"]

    # Show by clusters (connected components)
    adjacency = defaultdict(set)
    for edge in edges:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])

    visited = set()
    cluster_id = 0
    for node in nodes:
        if node not in visited:
            cluster_id += 1
            # BFS
            queue = [node]
            cluster = []
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(cluster) > 1:
                lines.append(f"\n  Cluster {cluster_id} ({len(cluster)} tiles):")
                for tile in cluster:
                    node_info = nodes[tile]
                    conn = len(adjacency[tile])
                    lines.append(f"    📄 {tile:40s} ({conn} edges) {node_info['title'][:30]}")
            else:
                lines.append(f"  📄 {cluster[0]:40s} (isolated)")

    return "\n".join(lines)


def show_clusters(graph=None):
    """Show tile clusters with statistics."""
    if not graph:
        graph = load_graph()
    if not graph:
        return "No graph built."

    adjacency = defaultdict(set)
    for edge in graph["edges"]:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])

    visited = set()
    clusters = []
    for node in graph["nodes"]:
        if node not in visited:
            queue = [node]
            cluster = []
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            clusters.append(cluster)

    clusters.sort(key=len, reverse=True)

    lines = [f"📊 Tile Clusters — {len(clusters)} groups\n{'='*50}"]
    for i, cluster in enumerate(clusters[:10]):
        if len(cluster) == 1:
            lines.append(f"  📄 {cluster[0]} (isolated)")
        else:
            # Find common tags
            all_tags = defaultdict(int)
            for tile in cluster:
                for tag in graph["nodes"][tile].get("tags", []):
                    all_tags[tag] += 1
            top_tags = sorted(all_tags.items(), key=lambda x: -x[1])[:3]
            tag_str = ", ".join(f"{t}({c})" for t, c in top_tags)
            lines.append(f"  📦 Cluster {i+1}: {len(cluster)} tiles — tags: {tag_str}")
            for tile in cluster:
                lines.append(f"    • {tile}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "graph"

    if cmd == "build":
        g = build_graph()
        print(f"✅ Graph built: {g['stats']['tiles']} tiles, {g['stats']['connections']} edges, {g['stats']['tags']} tags")

    elif cmd == "search":
        query = sys.argv[2] if len(sys.argv) > 2 else input("Search: ")
        print(search_graph(query))

    elif cmd == "related":
        name = sys.argv[2] if len(sys.argv) > 2 else input("Tile: ")
        print(related(name))

    elif cmd == "graph":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        print(show_graph(name))

    elif cmd == "clusters":
        print(show_clusters())

    else:
        print(__doc__)
