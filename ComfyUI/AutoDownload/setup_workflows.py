#!/usr/bin/env python3
"""
setup_workflows.py — one-shot setup for a folder of ComfyUI workflows.

What it does:
  1. Scans every .json workflow in a folder (recursively).
  2. Extracts every model filename referenced and every non-core node type used.
  3. Looks each one up in model_map.json / node_map.json (sitting next to this script).
  4. Downloads models with aria2c (multi-connection) and clones custom node repos —
     skipping anything already present.
  5. Prints a clear "NOT MAPPED — add manually" list for anything it doesn't
     recognize, so you can extend the JSON maps once and reuse them forever.

Usage:
  python3 setup_workflows.py /path/to/workflows --comfyui-dir /workspace/runpod-slim/ComfyUI

  # tune download concurrency
  python3 setup_workflows.py /path/to/workflows --connections 16 --parallel 3

  # if a model source needs a HuggingFace token
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
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_MAP_PATH = os.path.join(SCRIPT_DIR, "model_map.json")
NODE_MAP_PATH = os.path.join(SCRIPT_DIR, "node_map.json")

MODEL_EXTS = re.compile(r'\.(safetensors|ckpt|pt|pth|bin|onnx|gguf)$', re.I)

# Stock ComfyUI node types — not exhaustive, but covers common core + advanced
# sampling/video nodes added over time. Anything not in here is treated as a
# candidate custom node.
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


# ---------------------------------------------------------------------------
# Workflow parsing
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nodes(data):
    if isinstance(data, dict) and "nodes" in data:
        return data["nodes"]
    elif isinstance(data, dict):
        return list(data.values())
    return []


def extract(path):
    data = load_json(path)
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
# Maps
# ---------------------------------------------------------------------------
def load_map(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def have_aria2():
    if shutil.which("aria2c"):
        return True
    print("[setup] aria2c not found, attempting install...")
    if shutil.which("apt-get"):
        subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return shutil.which("aria2c") is not None


def download_model(filename, info, models_dir, connections, hf_token):
    dest_dir = os.path.join(models_dir, info["dest"])
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.isfile(dest_path):
        print(f"[SKIP] model exists: {filename}")
        return True

    print(f"[DL]   {filename}  ->  models/{info['dest']}/")
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
    if hf_token:
        cmd.insert(1, f"--header=Authorization: Bearer {hf_token}")

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        os.replace(dest_path + ".part", dest_path)
        print(f"[OK]   {filename}")
        return True
    else:
        print(f"[FAIL] {filename}: {result.stderr.strip()[-300:]}")
        return False


def clone_node(repo_url, custom_nodes_dir):
    dir_name = repo_url.rstrip("/").split("/")[-1]
    dest = os.path.join(custom_nodes_dir, dir_name)

    if os.path.isdir(dest):
        print(f"[SKIP] node exists: {dir_name}")
        return True

    print(f"[CLONE] {dir_name}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, dest],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(f"[FAIL] clone {dir_name}: {result.stderr.strip()[-300:]}")
        return False

    req = os.path.join(dest, "requirements.txt")
    if os.path.isfile(req):
        print(f"[PIP]  installing requirements for {dir_name}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req, "--break-system-packages"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    print(f"[OK]   {dir_name}")
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
    args = ap.parse_args()

    models_dir = os.path.join(args.comfyui_dir, "models")
    custom_nodes_dir = os.path.join(args.comfyui_dir, "custom_nodes")
    hf_token = os.environ.get("HF_TOKEN", "")

    files = find_workflow_files(args.workflows)
    if not files:
        print(f"No workflow files found at: {args.workflows}", file=sys.stderr)
        sys.exit(1)

    all_models, all_nodes = set(), set()
    for f in files:
        try:
            m, n = extract(f)
        except Exception as e:
            print(f"[WARN] failed to parse {f}: {e}", file=sys.stderr)
            continue
        all_models |= m
        all_nodes |= n

    print(f"=== Scanned {len(files)} workflow file(s) ===")
    print(f"Found {len(all_models)} unique model reference(s), {len(all_nodes)} unique custom node type(s)\n")

    model_map = load_map(MODEL_MAP_PATH)
    node_map = load_map(NODE_MAP_PATH)

    # ---- resolve nodes -> repos (clone sequentially, order doesn't matter much) ----
    resolved_repos = set()
    unmapped_nodes = set()
    for node_type in sorted(all_nodes):
        repo = node_map.get(node_type)
        if repo:
            resolved_repos.add(repo)
        else:
            unmapped_nodes.add(node_type)

    print("=== Installing custom nodes ===")
    if not have_aria2():
        print("[WARN] aria2c unavailable — model downloads will be skipped. Install aria2 and re-run.")
        aria2_ok = False
    else:
        aria2_ok = True

    for repo in sorted(resolved_repos):
        clone_node(repo, custom_nodes_dir)

    # ---- resolve models -> urls (download in parallel, throttled) ----
    print("\n=== Downloading models ===")
    unmapped_models = set()
    to_download = []
    for filename in sorted(all_models):
        info = model_map.get(filename)
        if info:
            to_download.append((filename, info))
        else:
            unmapped_models.add(filename)

    if aria2_ok and to_download:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            list(ex.map(
                lambda item: download_model(item[0], item[1], models_dir, args.connections, hf_token),
                to_download
            ))

    # ---- report ----
    print("\n=== Summary ===")
    print(f"Custom node repos resolved & installed: {len(resolved_repos)}")
    print(f"Models resolved & downloaded (or already present): {len(to_download)}")

    if unmapped_nodes:
        print(f"\n⚠ NOT MAPPED — {len(unmapped_nodes)} node type(s) with no known repo:")
        for n in sorted(unmapped_nodes):
            print(f"   {n}")
        print(f"   -> find the repo (check the workflow's missing-node banner in ComfyUI, or search")
        print(f"      GitHub/ComfyUI-Manager) and add an entry to: {NODE_MAP_PATH}")

    if unmapped_models:
        print(f"\n⚠ NOT MAPPED — {len(unmapped_models)} model file(s) with no known source:")
        for m in sorted(unmapped_models):
            print(f"   {m}")
        print(f"   -> find the download URL (HuggingFace/Civitai) and add an entry to: {MODEL_MAP_PATH}")

    if not unmapped_nodes and not unmapped_models:
        print("\nEverything resolved. ✅")


if __name__ == "__main__":
    main()
