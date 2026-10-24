"""Serve OpenVLA-7B actions for a SimplerEnv client.

The request body is JSON: {"image": "<base64 uint8 bytes>", "shape": [H, W, 3], "instruction": "..."}.
The response is {"action": [x, y, z, roll, pitch, yaw, gripper]}.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
UNNORM_KEY = "bridge_orig"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "openvla-7b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_policy(model_dir: Path, device: str):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return processor, model


def predict(processor, model, image: np.ndarray, instruction: str, device: str) -> list[float]:
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(device, dtype=torch.bfloat16)
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    return np.asarray(action, dtype=np.float64).reshape(-1).tolist()


def make_handler(processor, model, device: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                image = np.frombuffer(base64.b64decode(payload["image"]), dtype=np.uint8)
                image = image.reshape(payload["shape"])
                action = predict(processor, model, image, payload["instruction"], device)
                body = json.dumps({"action": action}).encode()
                self.send_response(200)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    args = parse_args()
    print(f"loading {args.model}", flush=True)
    processor, model = load_policy(args.model, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(processor, model, args.device))
    print(f"openvla server http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
