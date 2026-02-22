# app_gradio.py
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml

try:
    import jsonpatch
except Exception:
    jsonpatch = None

from openai import OpenAI

from msm_agent.pipeline import run_mvp


# ----------------------------
# Helpers
# ----------------------------
def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_yaml_load(s: str) -> Dict[str, Any]:
    obj = yaml.safe_load(s)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return obj


def yaml_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False)


def read_text(p: Path, default: str = "") -> str:
    return p.read_text() if p.exists() else default


def read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    g = metrics.get("grade", {})
    d = metrics.get("data", {})
    occ = metrics.get("occupancy", {})
    plat = metrics.get("plateau", {})
    lines = [
        f"Grade: {g.get('label','?')}",
        f"Total time (ns): {d.get('total_time_ns','NA')}",
        f"dt_ns_effective: {d.get('dt_ns_effective','NA')}",
        f"n_trajs: {d.get('n_trajs','NA')}",
        f"n_used_states: {occ.get('n_used','NA')} / {occ.get('n_clusters','NA')}",
        f"ITS rel_std_max(top-k): {plat.get('rel_std_max','NA')}",
    ]
    if g.get("fail_reasons"):
        lines.append("Fail reasons: " + "; ".join(g["fail_reasons"]))
    if g.get("warn_reasons"):
        lines.append("Warnings: " + "; ".join(g["warn_reasons"]))
    return "\n".join(lines)


def apply_patch(base_cfg: Dict[str, Any], patch_ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    if jsonpatch is None:
        # fallback: only support replace
        cfg = copy.deepcopy(base_cfg)
        for op in patch_ops:
            if op.get("op") != "replace":
                raise ValueError("jsonpatch not installed; only 'replace' supported.")
            path = op["path"].strip("/")
            keys = path.split("/") if path else []
            cur = cfg
            for k in keys[:-1]:
                cur = cur[k]
            cur[keys[-1]] = op["value"]
        return cfg

    patch = jsonpatch.JsonPatch(patch_ops)
    return patch.apply(copy.deepcopy(base_cfg), in_place=False)


def init_default_config() -> str:
    example_path = Path("examples/fs_peptide_mvp.yaml")
    if example_path.exists():
        return example_path.read_text()
    # fallback minimal
    return yaml_dump({
        "run": {"output_dir": "runs", "run_name": "fs_peptide_mvp", "seed": 42},
        "data": {"kind": "fs_peptide", "stride": 10, "dt_ps_per_frame": 50, "topology": None, "xtc_glob": None},
        "features": {"type": "dihedral", "dihedral_types": ["phi", "psi"]},
        "preprocess": {"scaler": "robust"},
        "tica": {"lag_time_frames": 2, "n_components": 4},
        "clustering": {"method": "kmeans", "n_clusters": 100},
        "msm": {"lag_time_frames_list": [1, 2, 5, 10, 20], "n_timescales": 10, "ergodic_cutoff": 0.5},
        "plots": {"bins": 90, "gridsize": 120},
        "gates": {"min_occupancy": 10, "max_tiny_state_frac": 0.30, "min_avg_out_degree": 3, "plateau_k": 3, "plateau_rel_var": 0.30},
    })


# ----------------------------
# OpenAI LLM Advisor
# ----------------------------
class OpenAIAdvisor:
    def __init__(self, model: str = "gpt-5.2"):
        self.client = OpenAI()
        self.model = model

    def analyze_one_run(
        self,
        base_cfg: Dict[str, Any],
        run_metrics: Dict[str, Any],
        n_patches: int = 6,
    ) -> Dict[str, Any]:
        """
        Returns dict with keys: analysis (str), patches (list[{name, ops}])
        Uses Structured Outputs (JSON Schema).
        """
        # Keep the payload small: send summary + key numeric signals, not full arrays.
        grade = run_metrics.get("grade", {})
        plateau = run_metrics.get("plateau", {})
        occupancy = run_metrics.get("occupancy", {})
        sparsity = run_metrics.get("sparsity", {})

        # Also include the current core hyperparams so patches are grounded.
        cur = {
            "tica": base_cfg.get("tica", {}),
            "clustering": base_cfg.get("clustering", {}),
            "msm": base_cfg.get("msm", {}),
            "gates": base_cfg.get("gates", {}),
        }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "analysis": {"type": "string"},
                "patches": {
                "type": "array",
                "minItems": 1,
                "maxItems": n_patches,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                    "name": {"type": "string"},
                    "ops": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "op": {"type": "string", "enum": ["replace"]},
                            "path": {
                            "type": "string",
                            "enum": [
                                "/tica/lag_time_frames",
                                "/tica/n_components",
                                "/clustering/n_clusters",
                                "/msm/lag_time_frames_list",
                                "/msm/n_timescales",
                            ],
                            },
                            "value_kind": {"type": "string", "enum": ["int", "int_list"]},
                            "value_int": {"type": "integer"},
                            "value_int_list": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["op", "path", "value_kind", "value_int", "value_int_list"],
                        },
                    },
                    },
                    "required": ["name", "ops"],
                },
                },
            },
            "required": ["analysis", "patches"],
            }

        prompt = {
            "role": "user",
            "content": (
                "You are an MSM validation assistant. Your job:\n"
                "1) Analyze ONE MSM run summary.\n"
                "2) Propose a batch of config JSON-patches (exactly N patches if possible).\n\n"
                "Constraints:\n"
                "- Only modify these paths: /tica/lag_time_frames, /tica/n_components, "
                "/clustering/n_clusters, /msm/lag_time_frames_list, /msm/n_timescales.\n"
                "- Keep changes minimal and targeted for diagnosing Markovianity/stability.\n"
                "- Prefer diversity: vary one factor per patch when possible.\n"
                "- No new keys, no deleting keys.\n\n"
                f"N = {n_patches}\n\n"
                "Current hyperparams:\n"
                f"{json.dumps(cur, indent=2)}\n\n"
                "Run signals:\n"
                f"{json.dumps({'grade': grade, 'plateau': plateau, 'occupancy': occupancy, 'sparsity': sparsity}, indent=2)}\n"
            )
        }

        response = self.client.responses.create(
            model=self.model,
            input=[prompt],
            # Structured Outputs (JSON schema)
            text={
                    "format": {
                        "type": "json_schema",
                        "name": "msm_patch_proposal",
                        "strict": True,
                        "schema": schema,
                    }
            }
        )

        # For Structured Outputs, output_text will be a JSON string conforming to schema.
        out = json.loads(response.output_text)
        print("\n[OpenAI raw output_text]\n", out, "\n")
        return out


# --- create once (module-level) ---
ADVISOR = OpenAIAdvisor(model="gpt-5.2")

# ----------------------------
# Session state
# ----------------------------
@dataclass
class SessionState:
    base_cfg_yaml: str = ""
    base_cfg_obj: Optional[Dict[str, Any]] = None

    base_run_dir: Optional[str] = None
    base_metrics: Optional[Dict[str, Any]] = None

    proposed: Optional[Dict[str, Any]] = None
    proposed_cfgs: List[Dict[str, Any]] = field(default_factory=list)

    batch_run_dirs: List[str] = field(default_factory=list)
    batch_metrics: List[Dict[str, Any]] = field(default_factory=list)

# ----------------------------
# Callbacks
# ----------------------------
def on_run_base(base_yaml: str, st: SessionState):
    chat: List[Tuple[str, str]] = []
    try:
        base_cfg = safe_yaml_load(base_yaml)
    except Exception as e:
        chat.append(("system", f"Config YAML parse error: {e}"))
        return chat, st, "", "", ""

    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("run", {})
    cfg["run"]["run_name"] = cfg["run"].get("run_name", "base") + f"_base_{now_id()}"

    run_dir = run_mvp(cfg)
    metrics = read_json(Path(run_dir) / "metrics.json")

    st.base_cfg_yaml = base_yaml
    st.base_cfg_obj = base_cfg
    st.base_run_dir = run_dir
    st.base_metrics = metrics

    summary = summarize_metrics(metrics)
    chat.append(("system", f"✅ Base run finished.\n\n{summary}\n\nRun dir: {run_dir}"))

    # LLM proposes patches
    proposal = ADVISOR.analyze_one_run(base_cfg=base_cfg, run_metrics=metrics, n_patches=6)
    st.proposed = proposal

    proposed_cfgs = []
    for i, p in enumerate(proposal["patches"], start=1):
        c = apply_patch(base_cfg, p["ops"])
        c.setdefault("run", {})
        c["run"]["run_name"] = f"batch_cfg{i}_{now_id()}"
        proposed_cfgs.append(c)
    st.proposed_cfgs = proposed_cfgs

    proposed_yaml = yaml_dump({
        "analysis": proposal["analysis"],
        "patches": proposal["patches"],
        "configs": proposed_cfgs
    })

    chat.append(("assistant", proposal["analysis"]))
    chat.append(("assistant", "I proposed 6 patch configs. Review/edit if needed, then click **Run next batch**."))

    return chat, st, summary, proposed_yaml, run_dir


def on_chat(user_msg: str, chat_history: List[Tuple[str, str]], st: SessionState):
    chat_history = chat_history or []
    chat_history.append(("user", user_msg))

    if st.base_metrics is None:
        chat_history.append(("assistant", "Run the base config first so I can analyze metrics and propose patches."))
        return chat_history, st

    # Lightweight: you can extend this to call OpenAI for free-form Q&A too.
    msg = user_msg.strip().lower()
    if msg in {"run next batch", "next batch"}:
        chat_history.append(("assistant", "Click **Run next batch** to execute the proposed configs."))
    else:
        chat_history.append(("assistant", "You can: (1) adjust base YAML and re-run base, or (2) run next batch, or ask about specific metrics (ITS, occupancy, sparsity)."))
    return chat_history, st


def on_run_next_batch(st: SessionState):
    chat: List[Tuple[str, str]] = []
    if not st.proposed_cfgs:
        chat.append(("system", "No proposed configs found. Run base first."))
        return chat, st, "", ""

    st.batch_run_dirs = []
    st.batch_metrics = []

    for i, cfg in enumerate(st.proposed_cfgs, start=1):
        cfg2 = copy.deepcopy(cfg)
        cfg2.setdefault("run", {})
        cfg2["run"]["run_name"] = cfg2["run"].get("run_name", f"batch{i}") + f"_{now_id()}"

        run_dir = run_mvp(cfg2)
        m = read_json(Path(run_dir) / "metrics.json")

        st.batch_run_dirs.append(run_dir)
        st.batch_metrics.append(m)

    # quick leaderboard
    rows = []
    for i, m in enumerate(st.batch_metrics, start=1):
        g = m.get("grade", {})
        plat = m.get("plateau", {})
        rows.append({
            "idx": i,
            "label": g.get("label"),
            "rel_std_max": plat.get("rel_std_max"),
            "run_dir": st.batch_run_dirs[i-1],
        })

    def sort_key(r):
        rank = {"PASS": 0, "WARN": 1, "FAIL": 2}.get(r["label"], 9)
        rel = r["rel_std_max"] if r["rel_std_max"] is not None else 1e9
        return (rank, rel)

    rows_sorted = sorted(rows, key=sort_key)
    leaderboard = "Batch results (sorted):\n" + "\n".join(
        [f"- cfg{r['idx']}: {r['label']} | rel_std_max={r['rel_std_max']} | {r['run_dir']}" for r in rows_sorted]
    )

    # next round: analyze best run and propose next patches
    best = rows_sorted[0]
    best_metrics = st.batch_metrics[best["idx"] - 1]
    proposal = ADVISOR.analyze_one_run(base_cfg=st.base_cfg_obj, run_metrics=best_metrics, n_patches=6)
    st.proposed = proposal

    st.proposed_cfgs = []
    for i, p in enumerate(proposal["patches"], start=1):
        c = apply_patch(st.base_cfg_obj, p["ops"])
        c.setdefault("run", {})
        c["run"]["run_name"] = f"round2_cfg{i}_{now_id()}"
        st.proposed_cfgs.append(c)

    proposed_yaml = yaml_dump({
        "analysis": proposal["analysis"],
        "patches": proposal["patches"],
        "configs": st.proposed_cfgs
    })

    chat.append(("system", "✅ Batch finished. I analyzed the best run and proposed the next round patches."))
    chat.append(("assistant", proposal["analysis"]))

    last_dir = st.batch_run_dirs[-1]
    return chat, st, leaderboard, last_dir, proposed_yaml


# ----------------------------
# UI
# ----------------------------
def build_app():
    with gr.Blocks(title="MSM Agent Chat (MVP)") as demo:
        st = gr.State(SessionState())

        gr.Markdown("## MSM Agent Chat (MVP)\nRun base → LLM proposes patches → run batch → repeat.")

        with gr.Row():
            with gr.Column(scale=1):
                chat = gr.Chatbot(label="Chat", height=520)
                user_in = gr.Textbox(label="Message", placeholder='Ask or type "run next batch"...')
                with gr.Row():
                    btn_send = gr.Button("Send")
                    btn_run_batch = gr.Button("Run next batch")

            with gr.Column(scale=1):
                base_yaml = gr.Code(label="Base config (YAML)", language="yaml", value=init_default_config())
                btn_run_base = gr.Button("Run base config")
                summary = gr.Textbox(label="Latest summary", lines=8)
                proposed = gr.Code(label="Proposed patches + configs (YAML)", language="yaml")
                leaderboard = gr.Textbox(label="Batch leaderboard", lines=10)
                latest_run_dir = gr.Textbox(label="Latest run dir")

        btn_run_base.click(
            fn=on_run_base,
            inputs=[base_yaml, st],
            outputs=[chat, st, summary, proposed, latest_run_dir],
            api_visibility="private",
        )

        btn_send.click(fn=on_chat, inputs=[user_in, chat, st], outputs=[chat, st],
                       api_visibility="private",)
        user_in.submit(fn=on_chat, inputs=[user_in, chat, st], outputs=[chat, st],
                       api_visibility="private",)

        btn_run_batch.click(
            fn=on_run_next_batch,
            inputs=[st],
            outputs=[chat, st, leaderboard, latest_run_dir, proposed],
            api_visibility="private",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue()  # allow long-running jobs
    demo.launch()