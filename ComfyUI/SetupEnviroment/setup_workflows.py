#!/usr/bin/env python3
"""
setup_workflows.py — daily setup for an ephemeral ComfyUI pod.

DESIGN: static JSON maps as the primary, fast, network-independent source of
truth (this is the proven-working approach), with live resolution as a
fallback ONLY for things not already in those maps — and anything resolved
that way gets written back into the same JSON files, so they keep growing
and there's a single source of truth to look at (not a separate cache you
have to remember exists).

RESOLUTION ORDER per item:
  1. model_map.json / node_map.json (next to this script) — instant, no
     network call, this is what you've already verified and trust.
  2. Live lookup:
       - nodes: Comfy Registry API (api.comfy.org) exact class-name match,
         same source ComfyUI's own "Install Missing Nodes" uses.
       - models: no universal registry exists, so this step is skipped;
         models always fall through to step 3 if not already mapped.
  3. Interactive prompt (only if running in a real terminal) — asks once,
     writes the answer straight into model_map.json / node_map.json.
  4. Still unresolved -> reported clearly at the end, nothing crashes.

ALWAYS-DOWNLOAD OVERRIDES (independent of what's in today's workflows):
  always_nodes.json   - flat list of repo URLs to always clone
  always_models.json  - {filename: {url, dest}} to always download

Every run only downloads/clones what's actually MISSING on disk — already
present files/repos are skipped instantly. Re-running after adding or
editing a workflow only pulls the delta for that day.

Failures are logged separately to errors.log with the exact command that
failed and its output, so you can see every problem in one place.

Usage:
  python3 setup_workflows.py /path/to/workflows
  python3 setup_workflows.py /path/to/workflows --comfyui-dir /workspace/x/ComfyUI
  python3 setup_workflows.py /path/to/workflows --non-interactive
  python3 setup_workflows.py /path/to/workflows --connections 24 --parallel 4
  HF_TOKEN=hf_xxx python3 setup_workflows.py /path/to/workflows
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import glob
import traceback
import urllib.request
import urllib.error
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_MAP_PATH = os.path.join(SCRIPT_DIR, "model_map.json")
NODE_MAP_PATH = os.path.join(SCRIPT_DIR, "node_map.json")
ALWAYS_NODES_FILE = os.path.join(SCRIPT_DIR, "always_nodes.json")
ALWAYS_MODELS_FILE = os.path.join(SCRIPT_DIR, "always_models.json")
ERROR_LOG = os.path.join(SCRIPT_DIR, "errors.log")
RUN_LOG = os.path.join(SCRIPT_DIR, "setup.log")

COMFY_API_BASE = "https://api.comfy.org"

MODEL_EXTS = re.compile(r'\.(safetensors|ckpt|pt|pth|bin|onnx|gguf)$', re.I)

CORE_NODES = {
    "KSampler", "KSamplerAdvanced", "CLIPTextEncode", "CLIPSetLastLayer", "VAEDecode",
    "VAEEncode", "VAEEncodeForInpaint", "CheckpointLoaderSimple", "LoraLoader",
    "VAELoader", "ControlNetLoader", "ControlNetApply", "ControlNetApplyAdvanced",
    "EmptyLatentImage", "LatentUpscale", "LatentUpscaleBy", "LatentComposite",
    "ImageScale", "ImageScaleBy", "SaveImage", "LoadImage", "PreviewImage",
    "CLIPLoader", "UNETLoader", "DualCLIPLoader", "UpscaleModelLoader",
    "ImageUpscaleWithModel", "ConditioningCombine", "ConditioningSetArea",
    "ConditioningZeroOut", "CLIPVisionLoader", "CLIPVisionEncode", "StyleModelLoader",
    "unCLIPConditioning", "GLIGENLoader", "DiffControlNetLoader", "Note", "Reroute",
    "PrimitiveNode", "MarkdownNote",
    "BasicScheduler", "CFGGuider", "CreateVideo", "KSamplerSelect", "RandomNoise",
    "SamplerCustomAdvanced", "SaveVideo", "LoadVideo", "LoadAudio", "SaveAudio",
    "EmptySD3LatentImage", "ModelSamplingSD3", "ModelSamplingFlux",
}

DEST_GUESS_RULES = [
    (re.compile(r'vae', re.I), "vae"),
    (re.compile(r'upscal', re.I), "latent_upscale_models"),
    (re.compile(r'(text_encoder|t5|clip|gemma)', re.I), "text_encoders"),
    (re.compile(r'lora', re.I), "loras"),
    (re.compile(r'controlnet', re.I), "controlnet"),
    (re.compile(r'\.gguf$', re.I), "unet"),
]


def guess_dest(filename):
    for pattern, dest in DEST_GUESS_RULES:
        if pattern.search(filename):
            return dest
    return "checkpoints"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(item, cmd, stderr_text):
    if isinstance(cmd, (list, tuple)):
        cmd_str = " ".join(str(c) for c in cmd)
    else:
        cmd_str = str(cmd)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"=== {ts} | {item} ===\n")
        f.write(f"COMMAND: {cmd_str}\n")
        if stderr_text:
            f.write(f"OUTPUT:\n{stderr_text.strip()}\n")
        f.write("\n")
    log(f"[FAIL] {item} — see errors.log for the exact command + output")


# ---------------------------------------------------------------------------
# Workflow parsing
# ---------------------------------------------------------------------------
def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nodes(data):
    if isinstance(data, dict) and "nodes" in data:
        return data["nodes"]
    elif isinstance(data, dict):
        return list(data.values())
    return []


def extract(path):
    data = load_json_file(path)
    nodes = get_nodes(data)
    models, node_types = set(), set()

    for node in nodes:
        if not isinstance(node, dict):
            continue
        ntype = node.get("type") or node.get("class_type")
        if ntype:
            node_types.add(ntype)

        widgets = node.get("widgets_values")
        if isinstance(widgets, list):
            for w in widgets:
                if isinstance(w, str) and MODEL_EXTS.search(w):
                    models.add(w)

        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, str) and MODEL_EXTS.search(v):
                    models.add(v)

    return models, (node_types - CORE_NODES)


def find_workflow_files(target):
    if os.path.isdir(target):
        return sorted(glob.glob(os.path.join(target, "**", "*.json"), recursive=True))
    elif any(ch in target for ch in "*?["):
        return sorted(glob.glob(target, recursive=True))
    return [target]


# ---------------------------------------------------------------------------
# Static maps — PRIMARY source of truth. Loaded once, written back to
# whenever something new gets resolved (live API or interactive prompt).
# ---------------------------------------------------------------------------
def load_map(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
        log(f"Created empty {os.path.basename(path)}.")
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if type(data) != type(default):
            log(f"[WARN] {path} has unexpected structure, treating as empty for this run.")
            return default
        return data
    except json.JSONDecodeError as e:
        log(f"[WARN] {path} has invalid JSON ({e}) — treating as empty for this run. Fix the file to keep its entries.")
        return default


def save_map(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_always_nodes():
    return load_map(ALWAYS_NODES_FILE, [])


def load_always_models():
    return load_map(ALWAYS_MODELS_FILE, {})


# ---------------------------------------------------------------------------
# Live fallback resolution (nodes only — models have no universal registry)
# ---------------------------------------------------------------------------
def http_get_json(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "setup_workflows.py"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def comfy_registry_lookup_by_node_name(node_name):
    data = http_get_json(f"{COMFY_API_BASE}/comfy-nodes/{node_name}/node")
    if data and data.get("repository"):
        return data["repository"]
    return None


def resolve_node(node_type, node_map, interactive):
    # 1. static map — primary, trusted, instant
    if node_type in node_map:
        return node_map[node_type]

    # 2. live Comfy Registry API — trusted, exact match
    repo = comfy_registry_lookup_by_node_name(node_type)
    if repo:
        node_map[node_type] = repo
        log(f"[NODE] {node_type} -> {repo}  (via Comfy Registry API, saved to node_map.json)")
        return repo

    # 3. interactive prompt, written back to the static map
    if interactive:
        choice = input(f"[NODE] '{node_type}' — not found anywhere. Paste its GitHub repo URL (or Enter to skip): ").strip()
        if choice.startswith("http"):
            node_map[node_type] = choice
            return choice

    return None


def resolve_model(filename, model_map, interactive):
    # 1. static map — primary, trusted, instant
    if filename in model_map:
        return model_map[filename]

    # 2. no universal model registry exists — go straight to interactive
    if interactive:
        print(f"\n[MODEL] '{filename}' — no known source yet.")
        url = input("        Paste a direct download URL (or Enter to skip): ").strip()
        if url.startswith("http"):
            default_dest = guess_dest(filename)
            dest = input(f"        Destination subfolder under models/ [{default_dest}]: ").strip() or default_dest
            info = {"url": url, "dest": dest}
            model_map[filename] = info
            return info

    return None


# ---------------------------------------------------------------------------
# Actions: clone / download
# ---------------------------------------------------------------------------
def have_aria2():
    if shutil.which("aria2c"):
        return True
    log("aria2c not found, attempting install...")
    install_cmd = ["apt-get", "install", "-y", "-qq", "aria2"]
    if shutil.which("apt-get"):
        subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = subprocess.run(install_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            log_error("install aria2", install_cmd, result.stderr)
    return shutil.which("aria2c") is not None


def download_model_file(filename, info, models_dir, connections, hf_token):
    dest_dir = os.path.join(models_dir, info["dest"])
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.isfile(dest_path):
        log(f"[SKIP] model exists: {filename}")
        return True

    log(f"[DL]   {filename}  ->  models/{info['dest']}/")
    cmd = [
        "aria2c",
        "-x", str(connections), "-s", str(connections),
        "-k", "1M",
        "--file-allocation=none",
        "--summary-interval=30",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "-d", dest_dir,
        "-o", filename + ".part",
        info["url"],
    ]
    real_cmd = cmd
    if hf_token:
        real_cmd = cmd.copy()
        real_cmd.insert(1, f"--header=Authorization: Bearer {hf_token}")

    result = subprocess.run(real_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        os.replace(dest_path + ".part", dest_path)
        log(f"[OK]   {filename}")
        return True
    else:
        log_error(f"download model: {filename}", cmd, result.stderr)  # token redacted in logged cmd
        return False


def clone_node_repo(repo_url, custom_nodes_dir):
    dir_name = repo_url.rstrip("/").split("/")[-1]
    dest = os.path.join(custom_nodes_dir, dir_name)

    if os.path.isdir(dest):
        log(f"[SKIP] node exists: {dir_name}")
        return True

    log(f"[CLONE] {dir_name}")
    cmd = ["git", "clone", "--depth", "1", repo_url, dest]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log_error(f"clone node: {dir_name}", cmd, result.stderr)
        return False

    req = os.path.join(dest, "requirements.txt")
    if os.path.isfile(req):
        log(f"[PIP]  installing requirements for {dir_name}")
        pip_cmd = [sys.executable, "-m", "pip", "install", "-r", req, "--break-system-packages"]
        result = subprocess.run(pip_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            log_error(f"pip install requirements for {dir_name}", pip_cmd, result.stderr)
    log(f"[OK]   {dir_name}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflows", help="Folder (or glob) of workflow .json files")
    ap.add_argument("--comfyui-dir", default=os.environ.get("COMFYUI_DIR", "/workspace/runpod-slim/ComfyUI"))
    ap.add_argument("--connections", type=int, default=16, help="aria2c connections per file")
    ap.add_argument("--parallel", type=int, default=3, help="how many models to download at once")
    ap.add_argument("--non-interactive", action="store_true", help="never prompt; just report unresolved items")
    args = ap.parse_args()

    interactive = (not args.non_interactive) and sys.stdin.isatty()

    models_dir = os.path.join(args.comfyui_dir, "models")
    custom_nodes_dir = os.path.join(args.comfyui_dir, "custom_nodes")
    hf_token = os.environ.get("HF_TOKEN", "")

    files = find_workflow_files(args.workflows)
    if not files:
        log(f"No workflow files found at: {args.workflows}")
        sys.exit(1)

    all_models, all_nodes = set(), set()
    for f in files:
        try:
            m, n = extract(f)
        except Exception:
            log_error(f"parse workflow: {f}", f"extract('{f}')", traceback.format_exc())
            continue
        all_models |= m
        all_nodes |= n

    log(f"=== Scanned {len(files)} workflow file(s) ===")
    log(f"Found {len(all_models)} model reference(s), {len(all_nodes)} custom node type(s) in today's workflows")

    always_node_repos = load_always_nodes()
    always_models = load_always_models()
    log(f"Always-download list: {len(always_node_repos)} node repo(s), {len(always_models)} model(s)")
    log(f"Interactive mode: {'ON' if interactive else 'OFF'}\n")

    node_map = load_map(NODE_MAP_PATH, {})
    model_map = load_map(MODEL_MAP_PATH, {})

    # ---- resolve + install nodes ----
    log("\n=== Resolving & installing custom nodes ===")
    unresolved_nodes = set()
    resolved_repos = set(always_node_repos)
    for node_type in sorted(all_nodes):
        repo = resolve_node(node_type, node_map, interactive)
        if repo:
            resolved_repos.add(repo)
        else:
            unresolved_nodes.add(node_type)
    save_map(NODE_MAP_PATH, node_map)  # write back anything newly resolved

    for repo in sorted(resolved_repos):
        clone_node_repo(repo, custom_nodes_dir)

    # ---- resolve + download models ----
    log("\n=== Resolving & downloading models ===")
    aria2_ok = have_aria2()
    if not aria2_ok:
        log("[WARN] aria2c unavailable — downloads will be skipped.")

    unresolved_models = set()
    to_download = dict(always_models)
    for filename in sorted(all_models):
        if filename in to_download:
            continue
        info = resolve_model(filename, model_map, interactive)
        if info:
            to_download[filename] = info
        else:
            unresolved_models.add(filename)
    save_map(MODEL_MAP_PATH, model_map)  # write back anything newly resolved

    if aria2_ok and to_download:
        items = list(to_download.items())
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            list(ex.map(
                lambda item: download_model_file(item[0], item[1], models_dir, args.connections, hf_token),
                items
            ))

    # ---- summary ----
    log("\n=== Summary ===")
    log(f"Custom node repos installed (workflow + always-list): {len(resolved_repos)}")
    log(f"Models downloaded or already present (workflow + always-list): {len(to_download)}")

    if unresolved_nodes:
        log(f"\n⚠ Still unresolved — {len(unresolved_nodes)} node type(s):")
        for n in sorted(unresolved_nodes):
            log(f"   {n}")
        log("   Re-run interactively (remove --non-interactive) to resolve and save into node_map.json.")

    if unresolved_models:
        log(f"\n⚠ Still unresolved — {len(unresolved_models)} model file(s):")
        for m in sorted(unresolved_models):
            log(f"   {m}")
        log("   Re-run interactively to provide a URL once and save into model_map.json.")

    if os.path.exists(ERROR_LOG):
        log(f"\n⚠ One or more failures occurred — see {ERROR_LOG} for exact commands + output.")
    elif not unresolved_nodes and not unresolved_models:
        log("\nEverything resolved and downloaded. ✅")

    log(f"\nRun log:   {RUN_LOG}")
    log(f"Error log: {ERROR_LOG} (only created if something failed)")


if __name__ == "__main__":
    main()
