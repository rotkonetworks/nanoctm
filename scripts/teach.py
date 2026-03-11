#!/usr/bin/env python3
"""Interactive teaching harness for CTM neuroplasticity.

Chat with the model, teach it facts, compact memories, log everything.
Designed to be driven by Claude or a human via stdin.

Commands:
  /teach <text>     - Add text to teaching buffer
  /compact          - Run compact_memory() on teaching buffer
  /recall [prompt]  - Test recall with optional prompt
  /clear            - Clear teaching buffer
  /save             - Save current CTM weights
  /load <path>      - Load CTM weights from checkpoint
  /dream            - Run convergence diagnostics
  /status           - Show model state, VRAM, teaching buffer
  /export           - Export session log to jsonl
  /quit             - Exit

Usage:
  NANOCHAT_NO_COMPILE=1 python3 -m scripts.teach [--checkpoint PATH]
"""

import os
os.environ.setdefault("NANOCHAT_NO_COMPILE", "1")

import sys
import json
import time
import argparse
import torch
from datetime import datetime
from pathlib import Path

from nanochat.common import compute_init, autodetect_device_type
from nanochat.qwen_backbone import QwenBackboneGPT, QwenTokenizer
from nanochat.engine import Engine
from nanochat.gpt import CTMCache


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive CTM teaching")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to CTM checkpoint (.pt)")
    parser.add_argument("--ctm-iterations", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="data/teaching_logs")
    parser.add_argument("--compact-lr", type=float, default=3e-4)
    parser.add_argument("--compact-steps", type=int, default=30)
    parser.add_argument("--gen-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


class TeachingSession:
    def __init__(self, args):
        self.args = args
        self.teaching_buffer = []
        self.session_log = []
        self.compact_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup log dir
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"session_{self.session_id}.jsonl"

        # Load model
        device_type = autodetect_device_type()
        _, _, _, _, self.device = compute_init(device_type)

        n_synch = 448 if "2.5" in args.backbone else 512
        self.model = QwenBackboneGPT.from_pretrained(
            args.backbone,
            ctm_kwargs={
                "ctm_iterations": args.ctm_iterations,
                "ctm_n_synch": n_synch,
                "ctm_memory_length": 16,
                "ctm_memory_hidden": 32,
                "ctm_synapse_depth": 32,
            },
        )
        self.model = self.model.to(self.device)

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=self.device, weights_only=True)
            self.model.load_ctm_state_dict(ckpt["ctm_state_dict"])
            print(f"Loaded checkpoint: {args.checkpoint}")
            del ckpt
            torch.cuda.empty_cache()
        else:
            self.model.init_ctm_weights()
            print("Initialized fresh CTM weights")

        self.model.eval()
        self.tokenizer = QwenTokenizer.from_pretrained(args.backbone)
        self.engine = Engine(self.model, self.tokenizer)

        vram = torch.cuda.memory_allocated() // 1024**2
        print(f"Model ready on {self.device} ({vram}MB VRAM)")
        print(f"Session log: {self.log_file}")
        print(f"Type /help for commands, or just chat.\n")

    def log_event(self, event_type, data):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session": self.session_id,
            "type": event_type,
            "compact_count": self.compact_count,
            **data,
        }
        self.session_log.append(entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def generate(self, prompt, max_tokens=None, temperature=None):
        max_tokens = max_tokens or self.args.gen_tokens
        temperature = temperature or self.args.temperature
        tokens = self.tokenizer.encode(prompt)
        self.model.reset_cache()
        results, _ = self.engine.generate_batch(
            tokens, num_samples=1, max_tokens=max_tokens,
            temperature=temperature, seed=int(time.time()) % 10000,
        )
        generated = self.tokenizer.decode(results[0][len(tokens):])
        return generated.strip()

    def cmd_teach(self, text):
        if not text:
            print("Usage: /teach <text to teach>")
            return
        self.teaching_buffer.append(text)
        print(f"  Added to buffer ({len(self.teaching_buffer)} items, "
              f"{sum(len(self.tokenizer.encode(t)) for t in self.teaching_buffer)} tokens)")
        self.log_event("teach", {"text": text})

    def cmd_compact(self):
        if not self.teaching_buffer:
            print("  Teaching buffer empty. Use /teach first.")
            return

        teaching_text = " ".join(self.teaching_buffer)
        tokens = self.tokenizer.encode(teaching_text)
        print(f"  Compacting {len(tokens)} tokens ({len(self.teaching_buffer)} items)...")

        teaching_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        target_ids = teaching_ids.clone()
        target_ids[:, :-1] = teaching_ids[:, 1:]
        target_ids[:, -1] = -1

        result = self.model.compact_memory(
            teaching_ids, target_ids,
            lr=self.args.compact_lr,
            steps=self.args.compact_steps,
        )

        self.compact_count += 1
        self.model.eval()

        print(f"  Loss: {result['losses'][0]:.3f} -> {result['losses'][-1]:.3f}")
        print(f"  Delta: {result.get('total_delta', 0):.4f}")
        print(f"  Dopamine: mean={result.get('dopamine_mean', 0):.3f}")
        print(f"  Compactions so far: {self.compact_count}")

        self.log_event("compact", {
            "teaching_text": teaching_text,
            "n_tokens": len(tokens),
            "loss_start": result["losses"][0],
            "loss_end": result["losses"][-1],
            "total_delta": result.get("total_delta", 0),
            "dopamine_mean": result.get("dopamine_mean", 0),
            "dopamine_std": result.get("dopamine_std", 0),
            "dS_out_norm": result.get("dS_out_norm", 0),
            "all_losses": result["losses"],
        })

    def cmd_recall(self, prompt=None):
        prompts = [prompt] if prompt else [
            "My name is",
            "What is my name?",
            "I am from",
            "What do you know about me?",
        ]
        print("  [Recall test]")
        for p in prompts:
            response = self.generate(p)
            print(f'    "{p}" -> "{response[:100]}"')
            self.log_event("recall", {"prompt": p, "response": response})

    def cmd_dream(self):
        results = self.model.dream(device=self.device)
        for layer_idx, d in results.items():
            print(f"  Layer {layer_idx}: converged={d['converged']}, "
                  f"final_delta={d['final_distance']:.4f}")
        self.log_event("dream", {
            str(k): {"converged": v["converged"], "final_delta": v["final_distance"]}
            for k, v in results.items()
        })

    def cmd_save(self, path=None):
        if path is None:
            path = self.log_dir / f"ctm_session_{self.session_id}_c{self.compact_count}.pt"
        state = {
            "ctm_state_dict": self.model.get_ctm_state_dict(),
            "compact_count": self.compact_count,
            "session_id": self.session_id,
        }
        torch.save(state, path)
        print(f"  Saved: {path}")
        self.log_event("save", {"path": str(path)})

    def cmd_load(self, path):
        if not path or not os.path.exists(path):
            print(f"  File not found: {path}")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_ctm_state_dict(ckpt["ctm_state_dict"])
        self.compact_count = ckpt.get("compact_count", 0)
        print(f"  Loaded: {path} (compactions: {self.compact_count})")
        del ckpt
        torch.cuda.empty_cache()
        self.log_event("load", {"path": path})

    def cmd_status(self):
        vram = torch.cuda.memory_allocated() // 1024**2
        peak = torch.cuda.max_memory_allocated() // 1024**2
        buf_tokens = sum(len(self.tokenizer.encode(t)) for t in self.teaching_buffer)
        print(f"  Backbone: {self.args.backbone}")
        print(f"  Checkpoint: {self.args.checkpoint or 'fresh'}")
        print(f"  VRAM: {vram}MB (peak: {peak}MB)")
        print(f"  Compactions: {self.compact_count}")
        print(f"  Teaching buffer: {len(self.teaching_buffer)} items, {buf_tokens} tokens")
        print(f"  Session log: {self.log_file}")
        print(f"  Log entries: {len(self.session_log)}")

    def cmd_clear(self):
        n = len(self.teaching_buffer)
        self.teaching_buffer.clear()
        print(f"  Cleared {n} items from teaching buffer")

    def cmd_export(self):
        print(f"  Session log: {self.log_file}")
        print(f"  {len(self.session_log)} entries")

    def cmd_refine(self, arg=""):
        """Self-refinement: model finds its own weak spots and compacts them.

        Generates text from diverse prompts, identifies highest prediction error
        sequences, and runs compact_memory on them. The model teaches itself.
        """
        n_rounds = int(arg) if arg.strip().isdigit() else 3
        refine_prompts = [
            "The history of", "In mathematics,", "A common mistake is",
            "The difference between", "One important fact is",
            "Scientists have discovered", "The process of",
            "According to research,", "A fundamental principle",
            "The relationship between", "In everyday life,",
            "The most common", "An interesting property of",
            "When we consider", "The key insight is",
        ]

        import random
        random.shuffle(refine_prompts)

        print(f"  [Self-refine: {n_rounds} rounds]")
        for round_i in range(n_rounds):
            # Generate from random prompts, measure prediction error
            candidates = []
            for prompt in refine_prompts[:5]:
                response = self.generate(prompt, max_tokens=40)
                full_text = prompt + " " + response

                # Measure prediction error on the generated text
                tokens = self.tokenizer.encode(full_text)
                ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                targets = ids.clone()
                targets[:, :-1] = ids[:, 1:]
                targets[:, -1] = -1
                with torch.no_grad():
                    _, loss = self.model.forward(ids, targets=targets)
                candidates.append((loss.item(), full_text, prompt))

            # Pick the worst (highest loss = most confused)
            candidates.sort(key=lambda x: -x[0])
            worst_loss, worst_text, worst_prompt = candidates[0]
            best_loss, _, best_prompt = candidates[-1]

            print(f"  Round {round_i+1}: worst={worst_loss:.3f} ({worst_prompt}), "
                  f"best={best_loss:.3f} ({best_prompt})")

            # Compact on the weakest area
            tokens = self.tokenizer.encode(worst_text)
            teaching_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            target_ids = teaching_ids.clone()
            target_ids[:, :-1] = teaching_ids[:, 1:]
            target_ids[:, -1] = -1

            result = self.model.compact_memory(
                teaching_ids, target_ids,
                lr=self.args.compact_lr * 0.5,  # gentler for self-refine
                steps=self.args.compact_steps // 2,
            )
            self.compact_count += 1
            self.model.eval()

            print(f"    Compacted: loss {result['losses'][0]:.3f} -> {result['losses'][-1]:.3f}, "
                  f"delta={result.get('total_delta', 0):.3f}")

            self.log_event("self_refine", {
                "round": round_i + 1,
                "worst_prompt": worst_prompt,
                "worst_text": worst_text,
                "worst_loss": worst_loss,
                "best_prompt": best_prompt,
                "best_loss": best_loss,
                "compact_loss_start": result["losses"][0],
                "compact_loss_end": result["losses"][-1],
                "compact_delta": result.get("total_delta", 0),
            })

            random.shuffle(refine_prompts)

        print(f"  Self-refine complete. Compactions: {self.compact_count}")

    def cmd_help(self):
        print("  Commands:")
        print("    /teach <text>   - Add to teaching buffer")
        print("    /compact        - Run compact_memory()")
        print("    /recall [prompt]- Test recall")
        print("    /refine [N]     - Self-refine N rounds (default 3)")
        print("    /clear          - Clear teaching buffer")
        print("    /save [path]    - Save CTM weights")
        print("    /load <path>    - Load CTM weights")
        print("    /dream          - Convergence diagnostics")
        print("    /status         - Model state & VRAM")
        print("    /export         - Show log path")
        print("    /quit           - Exit")
        print("  Or just type to chat with the model.")

    def chat(self, text):
        response = self.generate(text)
        print(f"  > {response[:200]}")
        self.log_event("chat", {"prompt": text, "response": response})

    def run(self):
        while True:
            try:
                line = input("you: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not line:
                continue

            if line.startswith("/"):
                parts = line.split(None, 1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/quit" or cmd == "/exit":
                    break
                elif cmd == "/teach":
                    self.cmd_teach(arg)
                elif cmd == "/compact":
                    self.cmd_compact()
                elif cmd == "/recall":
                    self.cmd_recall(arg if arg else None)
                elif cmd == "/refine":
                    self.cmd_refine(arg)
                elif cmd == "/clear":
                    self.cmd_clear()
                elif cmd == "/save":
                    self.cmd_save(arg if arg else None)
                elif cmd == "/load":
                    self.cmd_load(arg)
                elif cmd == "/dream":
                    self.cmd_dream()
                elif cmd == "/status":
                    self.cmd_status()
                elif cmd == "/export":
                    self.cmd_export()
                elif cmd == "/help":
                    self.cmd_help()
                else:
                    print(f"  Unknown command: {cmd}. Type /help")
            else:
                self.chat(line)


if __name__ == "__main__":
    args = parse_args()
    session = TeachingSession(args)
    session.run()
