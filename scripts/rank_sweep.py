"""B4 bonus: controlled rank sweep, position fixed at text-linear, r in {8, 16, 64}.

Run from the repo root in Colab, after NB1-NB5 have completed (needs adapters/correct/,
data/split/train.jsonl, results/baselines_frozen.json, results/runs.csv all present).

    !python scripts/rank_sweep.py

r=16 is not retrained -- it IS `correct` (same target=text-linear, same LR=LORA_LR),
already trained and scored. Only r=8 and r=64 are new runs.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from labkit import data, evaluate as ev, generate, modeling, report, train
from labkit.config import LORA_LR, SPECS, get_tier, training_epochs

TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


train_rows = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
target = load_jsonl(ROOT / "data" / "eval_target.jsonl")

EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
if EVAL_LIMIT:
    target = target[:EVAL_LIMIT]


def make_spec(r: int):
    from labkit.config import LoraSpec
    return LoraSpec(
        key=f"rank_r{r}", r=r, alpha=2 * r, target="text-linear", lr=LORA_LR,
        load_in_4bit=False,
        label=f"all-linear · r={r} · LR 10x · 16-bit",
        teaches=f"B4 rank sweep: position fixed at text-linear, r={r}.",
    )


def train_rank(r: int) -> pathlib.Path:
    spec = make_spec(r)
    out = ROOT / "adapters" / spec.key
    if (out / "adapter_model.safetensors").exists() and not os.environ.get("FORCE_RETRAIN"):
        print(f"skip r={r}: {out} already trained")
        return out

    model, tok = generate.load_base(TIER, load_in_4bit=spec.load_in_4bit)
    train_ds = Dataset.from_list(
        data.to_training_dataset(tok, train_rows, max_length=TIER.max_length,
                                 mask_mode=os.environ.get("MASK_MODE", "assistant-only")))
    targets = modeling.resolve_target_modules(model, spec.target)
    trainable = modeling.count_lora_params(model, targets, spec.r)
    max_steps = train.planned_steps(len(train_ds), TIER, training_epochs())

    want = train.sft_config_kwargs(TIER, spec, str(out), max_steps=max_steps)
    sft_kwargs, _ = train.filter_kwargs(SFTConfig, want, label=f"SFTConfig[r{r}]")
    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[r{r}]")

    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                         train_dataset=train_ds, processing_class=tok,
                         peft_config=LoraConfig(**lora_kwargs))
    train.align_trainable_precision(trainer.model)

    t0 = time.perf_counter()
    res = trainer.train()
    elapsed = time.perf_counter() - t0
    trainer.model.save_pretrained(out)

    row = train.summarize_run(spec, TIER, targets, trainable, elapsed, generate.peak_vram_gb())
    row["final_loss"] = round(res.training_loss, 4)
    row["max_steps"] = max_steps
    row["teaches"] = spec.teaches
    report.append_row(row, results_dir=ROOT / "results")

    del trainer, model
    generate.free_memory()
    return out


def score(adapter_dir: pathlib.Path, r: int) -> dict:
    model, tok = generate.load_base(TIER, load_in_4bit=False)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    preds, lat = generate.generate_batch(model, tok, [row["input"] for row in target],
                                         system=None, label=f"rank_r{r}/target")
    tgt = sum(ev.triage_field_accuracy(p, row["label"]) for p, row in zip(preds, target)) / len(target)
    fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)
    del model
    generate.free_memory()
    return {"r": r, "target": round(tgt, 4), "format": round(fmt, 4), "n": len(target)}


print("=" * 70)
print("B4 — controlled rank sweep, position fixed at text-linear")
print("=" * 70)

sweep = []
for r in (8, 64):
    print(f"\n--- r={r} ---")
    adir = train_rank(r)
    sweep.append(score(adir, r))

# r=16 reuses `correct` -- same target=text-linear, same LR, already trained/scored.
correct_rows = [row for row in report.read_rows("runs.csv", results_dir=ROOT / "results")
                if row.get("run") == "correct"]
if correct_rows:
    r16_row = correct_rows[-1]
    sweep.append({"r": 16, "target": None, "format": None, "n": None,
                  "note": "reused from `correct` (results/autopsy.json / verdict.json comparison[c])"})

sweep.sort(key=lambda x: x["r"])
print("\n--- Rank sweep result ---")
print(report.markdown_table(sweep, ["r", "target", "format", "n"]))
report.write_json(sweep, "rank_sweep.json", results_dir=ROOT / "results")
print("\nwrote results/rank_sweep.json")
