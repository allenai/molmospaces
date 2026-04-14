#!/usr/bin/env python3
"""
Failure Video Viewer — scans house_* directories inside an eval output folder,
finds failed episodes based on the h5 success flag at the last timestep, and
serves a card-based HTML viewer.

Usage:
    python scripts/view_failures.py <eval_folder> [--port 9123] [--plan-step-threshold 52]
"""

import argparse
import glob
import http.server
import json
import re
import socketserver
import webbrowser
from datetime import datetime
from pathlib import Path

import h5py


def scan_failures(base_dir: Path):
    failures = []
    h5_files = sorted(glob.glob(str(base_dir / "house_*" / "trajectories*.h5")))

    _dt_re = re.compile(r"(\d{8}_\d{6})")

    for h5_path in h5_files:
        house_dir = Path(h5_path).parent
        house_name = house_dir.name

        # Extract run datetime from path (e.g. "20260410_150954")
        run_dt_compact = None
        run_dt_iso = None
        for part in house_dir.parts:
            m = _dt_re.fullmatch(part)
            if m:
                run_dt_compact = m.group(1)
                try:
                    run_dt_iso = datetime.strptime(run_dt_compact, "%Y%m%d_%H%M%S").strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    pass
                break

        with h5py.File(h5_path, "r") as f:
            for traj_key in sorted(f.keys()):
                traj_idx = int(traj_key.split("_")[1])
                success_arr = f[traj_key]["success"][:]
                if success_arr[-1]:
                    continue

                scene_raw = f[traj_key]["obs_scene"][()]
                if isinstance(scene_raw, bytes):
                    scene_raw = scene_raw.decode()
                scene = json.loads(scene_raw)

                task_desc = scene.get("task_description", "N/A")
                prompt = scene.get("prompt", task_desc)
                task_type = scene.get("task_type", "unknown")
                obj_name = scene.get("object_name", "N/A")
                policy = scene.get("policy_name", "N/A")
                time_spent = scene.get("time_spent", 0)
                num_steps = len(success_arr)

                ep_prefix = f"episode_{traj_idx:08d}"
                wrist_videos = sorted(
                    house_dir.glob(f"{ep_prefix}_wrist_camera_batch_*.mp4")
                )
                exo_videos = [
                    v
                    for v in sorted(
                        house_dir.glob(f"{ep_prefix}_exo_camera_*_batch_*.mp4")
                    )
                    if "depth" not in v.name
                ]

                if not wrist_videos:
                    continue

                failures.append(
                    {
                        "house": house_name,
                        "traj_key": traj_key,
                        "output_folder": str(house_dir),
                        "run_dt_compact": run_dt_compact or "",
                        "run_dt_iso": run_dt_iso or "",
                        "task_description": task_desc,
                        "prompt": prompt,
                        "task_type": task_type,
                        "object_name": obj_name,
                        "policy": policy,
                        "time_spent": round(time_spent, 1),
                        "num_steps": num_steps,
                        "wrist_video": str(wrist_videos[0].relative_to(base_dir)),
                        "exo_video": str(exo_videos[0].relative_to(base_dir))
                        if exo_videos
                        else None,
                    }
                )

    return failures


def count_total_episodes(base_dir: Path) -> int:
    total = 0
    for h5_path in glob.glob(str(base_dir / "house_*" / "trajectories*.h5")):
        with h5py.File(h5_path, "r") as f:
            total += len(f.keys())
    return total


def build_html(failures, base_dir: Path, plan_step_threshold: int) -> str:
    total_episodes = count_total_episodes(base_dir)
    plan_fails = sum(1 for f in failures if f["num_steps"] <= plan_step_threshold)
    exec_fails = len(failures) - plan_fails
    success_rate = (
        round(100 * (total_episodes - len(failures)) / total_episodes, 1)
        if total_episodes
        else 0.0
    )

    # Subtitle: show last two path segments (e.g. "TiptopPolicyEvalConfig/20260413_171256")
    parts = base_dir.resolve().parts
    subtitle = "/".join(parts[-2:]) if len(parts) >= 2 else base_dir.name

    cards_html = ""
    for fail in failures:
        exo_section = ""
        if fail["exo_video"]:
            exo_section = f"""
                <div class="video-container">
                    <span class="video-label">Exo Camera</span>
                    <video controls preload="metadata">
                        <source src="/{fail['exo_video']}" type="video/mp4">
                    </video>
                </div>"""

        is_plan_fail = fail["num_steps"] <= plan_step_threshold
        category_badge = (
            '<span class="category-badge plan-fail">Fail to Plan</span>'
            if is_plan_fail
            else '<span class="category-badge exec-fail">Execution Failure</span>'
        )
        collapsed_class = " collapsed" if is_plan_fail else ""
        category = "plan-fail" if is_plan_fail else "exec-fail"

        cards_html += f"""
        <div class="card{collapsed_class}" data-category="{category}">
            <div class="card-header" onclick="toggleCard(this)">
                <span class="toggle-icon">▾</span>
                <span class="house-badge">{fail['house']}</span>
                <span class="traj-badge">{fail['traj_key']}</span>
                <span class="type-badge">{fail['task_type']}</span>
                {category_badge}
                <span class="header-preview">{fail['prompt']}</span>
                <span class="header-steps">{fail['num_steps']} steps</span>
            </div>
            <div class="card-body">
                <div class="instruction">
                    <span class="label">Instruction:</span> {fail['prompt']}
                </div>
                <div class="meta-row">
                    <div class="meta-item"><span class="label">Object:</span> {fail['object_name']}</div>
                    <div class="meta-item"><span class="label">Policy:</span> {fail['policy']}</div>
                    <div class="meta-item"><span class="label">Steps:</span> {fail['num_steps']}</div>
                    <div class="meta-item"><span class="label">Time:</span> {fail['time_spent']}s</div>
                </div>
                <div class="meta-row">
                    <div class="meta-item"><span class="label">Run Time:</span> {fail['run_dt_iso'] or 'N/A'}<span class="hidden-search">{fail['run_dt_compact']}</span></div>
                    <div class="meta-item output-folder"><span class="label">Output Folder:</span> <code>{fail['output_folder']}</code></div>
                </div>
                <div class="videos">
                    <div class="video-container">
                        <span class="video-label">Wrist Camera</span>
                        <video controls preload="metadata">
                            <source src="/{fail['wrist_video']}" type="video/mp4">
                        </video>
                    </div>
                    {exo_section}
                </div>
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Failure Video Viewer</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e1e4e8; padding: 24px; }}
    .header {{ max-width: 1200px; margin: 0 auto 32px; }}
    .header h1 {{ font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 8px; }}
    .header .subtitle {{ color: #8b949e; font-size: 15px; }}
    .stats {{ display: flex; gap: 24px; margin-top: 16px; flex-wrap: wrap; }}
    .stat-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px 24px; min-width: 140px; }}
    .stat-box .stat-value {{ font-size: 32px; font-weight: 700; color: #ff6b6b; }}
    .stat-box .stat-value.success {{ color: #3fb950; }}
    .stat-box .stat-value.total {{ color: #58a6ff; }}
    .stat-box .stat-label {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
    .controls {{ max-width: 1200px; margin: 0 auto 24px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    .controls input {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 16px; color: #e1e4e8; font-size: 14px; flex: 1; min-width: 200px; }}
    .controls input::placeholder {{ color: #484f58; }}
    .controls input:focus {{ outline: none; border-color: #58a6ff; }}
    .controls button {{ background: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 10px 16px; color: #e1e4e8; cursor: pointer; font-size: 13px; white-space: nowrap; }}
    .controls button:hover {{ background: #30363d; }}
    .controls button.active {{ background: #1f6feb; border-color: #1f6feb; }}
    .cards {{ max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; overflow: hidden; transition: border-color 0.2s; }}
    .card:hover {{ border-color: #484f58; }}
    .card-header {{ padding: 16px 20px; display: flex; gap: 10px; align-items: center; border-bottom: 1px solid #21262d; flex-wrap: wrap; cursor: pointer; user-select: none; }}
    .card-header:hover {{ background: #1c2128; }}
    .toggle-icon {{ font-size: 14px; color: #8b949e; transition: transform 0.2s; width: 14px; display: inline-block; }}
    .card.collapsed .toggle-icon {{ transform: rotate(-90deg); }}
    .card.collapsed .card-body {{ display: none; }}
    .card.collapsed .card-header {{ border-bottom: none; }}
    .header-preview {{ color: #c9d1d9; font-size: 14px; flex: 1; min-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .header-steps {{ color: #8b949e; font-size: 12px; margin-left: auto; }}
    .category-badge {{ padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; }}
    .category-badge.plan-fail {{ background: #3d2e10; color: #f0b93a; }}
    .category-badge.exec-fail {{ background: #2d1b3d; color: #c77dff; }}
    .house-badge {{ background: #1f2937; color: #58a6ff; padding: 4px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; }}
    .traj-badge {{ background: #1f2937; color: #8b949e; padding: 4px 12px; border-radius: 6px; font-size: 13px; }}
    .type-badge {{ background: #3b1f23; color: #ff6b6b; padding: 4px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; text-transform: uppercase; }}
    .card-body {{ padding: 20px; }}
    .instruction {{ font-size: 18px; font-weight: 500; color: #f0f3f6; margin-bottom: 16px; line-height: 1.4; }}
    .instruction .label {{ color: #8b949e; font-size: 13px; font-weight: 400; display: block; margin-bottom: 4px; }}
    .meta-row {{ display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }}
    .meta-item {{ font-size: 13px; color: #8b949e; }}
    .meta-item .label {{ font-weight: 600; color: #c9d1d9; }}
    .hidden-search {{ display: none; }}
    .output-folder code {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 12px; background: #0d1117; padding: 2px 6px; border-radius: 4px; color: #79c0ff; word-break: break-all; }}
    .videos {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .video-container {{ flex: 1; min-width: 280px; }}
    .video-label {{ display: block; font-size: 12px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
    video {{ width: 100%; border-radius: 8px; background: #0d1117; }}
    .empty {{ text-align: center; padding: 60px 20px; color: #484f58; font-size: 16px; }}
    .count-display {{ color: #8b949e; font-size: 14px; margin-left: auto; }}
</style>
</head>
<body>
    <div class="header">
        <h1>Failure Video Viewer</h1>
        <p class="subtitle">{subtitle}</p>
        <div class="stats">
            <div class="stat-box"><div class="stat-value total">{total_episodes}</div><div class="stat-label">Total Episodes</div></div>
            <div class="stat-box"><div class="stat-value">{len(failures)}</div><div class="stat-label">Failures</div></div>
            <div class="stat-box"><div class="stat-value" style="color: #f0b93a;">{plan_fails}</div><div class="stat-label">Fail to Plan (≤{plan_step_threshold} steps)</div></div>
            <div class="stat-box"><div class="stat-value" style="color: #c77dff;">{exec_fails}</div><div class="stat-label">Execution Failures</div></div>
            <div class="stat-box"><div class="stat-value success">{total_episodes - len(failures)}</div><div class="stat-label">Successes</div></div>
            <div class="stat-box"><div class="stat-value total">{success_rate}%</div><div class="stat-label">Success Rate</div></div>
        </div>
    </div>

    <div class="controls">
        <input type="text" id="search" placeholder="Filter by house, instruction, object..." oninput="filterCards()">
        <button class="category-btn active" onclick="setCategory('all', this)">All</button>
        <button class="category-btn" onclick="setCategory('exec-fail', this)">Execution</button>
        <button class="category-btn" onclick="setCategory('plan-fail', this)">Fail to Plan</button>
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
        <button onclick="playAll()">Play Visible</button>
        <button onclick="pauseAll()">Pause All</button>
        <span class="count-display" id="count-display"></span>
    </div>

    <div class="cards" id="cards">
        {cards_html if cards_html else '<div class="empty">No failures found!</div>'}
    </div>

<script>
let activeCategory = 'all';

function filterCards() {{
    const query = document.getElementById('search').value.toLowerCase();
    const cards = document.querySelectorAll('.card');
    let visible = 0;
    cards.forEach(card => {{
        const text = card.textContent.toLowerCase();
        const matchesQuery = !query || text.includes(query);
        const matchesCategory = activeCategory === 'all' || card.dataset.category === activeCategory;
        const show = matchesQuery && matchesCategory;
        card.style.display = show ? '' : 'none';
        if (show) visible++;
    }});
    document.getElementById('count-display').textContent = visible + ' of ' + cards.length + ' shown';
}}

function setCategory(cat, btn) {{
    activeCategory = cat;
    document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filterCards();
}}

function toggleCard(header) {{ header.parentElement.classList.toggle('collapsed'); }}

function expandAll() {{
    document.querySelectorAll('.card').forEach(c => {{
        if (c.style.display !== 'none') c.classList.remove('collapsed');
    }});
}}

function collapseAll() {{
    document.querySelectorAll('.card').forEach(c => c.classList.add('collapsed'));
}}

function playAll() {{
    document.querySelectorAll('.card').forEach(card => {{
        if (card.style.display !== 'none' && !card.classList.contains('collapsed')) {{
            card.querySelectorAll('video').forEach(v => v.play());
        }}
    }});
}}

function pauseAll() {{ document.querySelectorAll('video').forEach(v => v.pause()); }}

filterCards();
</script>
</body>
</html>"""


def make_handler(base_dir: Path, html_content: str):
    class VideoHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(base_dir), **kwargs)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                content = html_content.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                super().do_GET()

        def log_message(self, format, *args):
            pass

    return VideoHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_folder",
        type=str,
        help="Path to an eval output folder containing house_* subdirectories",
    )
    parser.add_argument("--port", type=int, default=9123, help="Port to serve on")
    parser.add_argument(
        "--plan-step-threshold",
        type=int,
        default=52,
        help="Episodes with num_steps <= this are classified as 'Fail to Plan'",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open a browser window",
    )
    args = parser.parse_args()

    base_dir = Path(args.eval_folder).resolve()
    if not base_dir.is_dir():
        raise SystemExit(f"Not a directory: {base_dir}")
    if not list(base_dir.glob("house_*")):
        raise SystemExit(f"No house_* subdirectories found in: {base_dir}")

    print(f"Scanning {base_dir} for failures...")
    failures = scan_failures(base_dir)
    print(f"Found {len(failures)} failed episodes.")
    html = build_html(failures, base_dir, args.plan_step_threshold)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), make_handler(base_dir, html)) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"Serving at {url}")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
