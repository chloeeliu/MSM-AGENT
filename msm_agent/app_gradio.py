# app_gradio.py
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml
from openai import OpenAI

from msm_agent.pipeline import run_mvp


# ----------------------------
# Helpers
# ----------------------------
def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def safe_yaml_load(s: str) -> Dict[str, Any]:
    obj = yaml.safe_load(s)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return obj


def yaml_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False)


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
# OpenAI LLM Advisor (text-only)
# ----------------------------
class OpenAIAdvisor:
    def __init__(self, model: str = "gpt-5.2"):
        self.client = OpenAI()
        self.model = model

    def analyze_one_run_text(self, base_cfg: Dict[str, Any], run_metrics: Dict[str, Any]) -> str:
        grade = run_metrics.get("grade", {})
        plateau = run_metrics.get("plateau", {})
        occupancy = run_metrics.get("occupancy", {})
        sparsity = run_metrics.get("sparsity", {})

        cur = {
            "tica": base_cfg.get("tica", {}),
            "clustering": base_cfg.get("clustering", {}),
            "msm": base_cfg.get("msm", {}),
            "gates": base_cfg.get("gates", {}),
        }

        prompt = {
            "role": "user",
            "content": (
                "You are an MSM validation assistant.\n\n"
                "Task: Analyze ONE MSM run.\n"
                "Output format:\n"
                "A) Diagnosis (2-5 bullets)\n"
                "B) Recommended next changes (5-8 bullets). Each bullet MUST include:\n"
                "   - exact config key path (e.g., tica.lag_time_frames)\n"
                "   - exact proposed value(s)\n"
                "   - what hypothesis it tests (Markovianity vs discretization vs sampling)\n"
                "C) A short prioritized plan (3 steps)\n\n"
                "Constraints:\n"
                "- Only suggest edits within: tica.lag_time_frames, tica.n_components, clustering.n_clusters, "
                "msm.lag_time_frames_list, msm.n_timescales.\n"
                "- Keep changes minimal; prefer one-factor-at-a-time.\n"
                "- Avoid long explanations.\n\n"
                "Current hyperparams:\n"
                f"{json.dumps(cur, indent=2)}\n\n"
                "Run signals:\n"
                f"{json.dumps({'grade': grade, 'plateau': plateau, 'occupancy': occupancy, 'sparsity': sparsity}, indent=2)}\n"
            ),
        }

        resp = self.client.responses.create(
            model=self.model,
            input=[prompt],
        )
        return resp.output_text
    

    def chat_followup(self, chat_history: List[Dict[str, str]], st: SessionState) -> str:
        # Provide minimal context: last run summary + last advice + last few messages
        context = []
        if st.base_metrics and st.base_cfg_obj:
            context.append({
                "role": "system",
                "content": "You are an MSM validation assistant. Continue the conversation about the user's latest run."
            })
            context.append({
                "role": "system",
                "content": "Latest run summary:\n" + summarize_metrics(st.base_metrics)
            })
            if st.last_advice_text:
                context.append({"role": "assistant", "content": st.last_advice_text})

        # Append last ~8 turns for brevity
        tail = chat_history[-8:] if chat_history else []

        def to_openai_msg(m: Dict[str, Any]) -> Dict[str, str]:
            # Strip any extra keys like metadata, id, etc.
            return {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}

        input_msgs = context + [to_openai_msg(m) for m in tail]

        resp = self.client.responses.create(model=self.model, input=input_msgs)

        return resp.output_text


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
    last_advice_text: str = ""


# ----------------------------
# Callbacks
# ----------------------------
def on_run_base(base_yaml: str, st: SessionState):
    chat: List[Tuple[str, str]] = []
    try:
        base_cfg = safe_yaml_load(base_yaml)
    except Exception as e:
        chat.append(msg("system", f"Config YAML parse error: {e}"))
        return chat, st, "", ""

    # run base
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("run", {})
    cfg["run"]["run_name"] = cfg["run"].get("run_name", "base") + f"_base_{now_id()}"

    run_dir = run_mvp(cfg)
    metrics = read_json(Path(run_dir) / "metrics.json")

    # store state
    st.base_cfg_yaml = base_yaml
    st.base_cfg_obj = base_cfg
    st.base_run_dir = run_dir
    st.base_metrics = metrics

    summary = summarize_metrics(metrics)
    chat.append(msg("system", f"✅ Base run finished.\n\n{summary}\n\nRun dir: {run_dir}"))

    # advisor text-only
    try:
        advice = ADVISOR.analyze_one_run_text(base_cfg=base_cfg, run_metrics=metrics)
    except Exception as e:
        # don't crash UI; show error in chat
        chat.append(msg("system", f"OpenAI advisor error: {e}"))
        return chat, st, summary, run_dir

    st.last_advice_text = advice
    chat.append(msg("assistant", advice))

    return chat, st, summary, run_dir




def on_chat(user_msg: str, chat_history: List[Dict[str, str]], st: SessionState):
    chat_history = chat_history or []
    chat_history.append(msg("user", user_msg))

    cmd = (user_msg or "").strip().lower()

    if st.base_metrics is None or st.base_cfg_obj is None:
        chat_history.append(msg("assistant", "Run the base config first so I can analyze the MSM metrics."))
        return chat_history, st

    # # optional command: re-run advisor on last base metrics
    # if cmd in {"suggest next", "advise", "analyze", "analyze base"}:
    #     try:
    #         advice = ADVISOR.analyze_one_run_text(base_cfg=st.base_cfg_obj, run_metrics=st.base_metrics)
    #         st.last_advice_text = advice
    #         chat_history.append(msg("assistant", advice))
    #     except Exception as e:
    #         chat_history.append(msg("assistant", f"OpenAI advisor error: {e}"))
    #     return chat_history, st

    # # otherwise: lightweight local response (you can expand later to call OpenAI chat)
    # chat_history.append(msg
    #     ("assistant",
    #      "I can help you interpret the latest run. Try:\n"
    #      "- Ask about specific metrics (e.g., 'what does rel_std_max mean?')\n"
    #      "- Type 'suggest next' to get another set of concrete parameter edits\n"
    #      "- Edit the YAML on the right and click **Run base config** again")
    # )

    reply = ADVISOR.chat_followup(chat_history, st)
    chat_history.append(msg("assistant", reply))
    return chat_history, st



# ----------------------------
# UI 
# ----------------------------
def build_app():
    with gr.Blocks(title="MSM Agent Chat ") as demo:
        st = gr.State(SessionState())

        gr.Markdown("## MSM Agent Chat \nRun base → get summary → get advisor suggestions → iterate.")

        with gr.Row():
            with gr.Column(scale=1):
                chat = gr.Chatbot(label="Chat", height=560)
                user_in = gr.Textbox(label="Message", placeholder='Ask or type "suggest next"...')
                with gr.Row():
                    btn_send = gr.Button("Send")

            with gr.Column(scale=1):
                base_yaml = gr.Code(label="Base config (YAML)", language="yaml", value=init_default_config())
                btn_run_base = gr.Button("Run base config")
                summary = gr.Textbox(label="Latest summary", lines=8)
                latest_run_dir = gr.Textbox(label="Latest run dir")

        btn_run_base.click(
            fn=on_run_base,
            inputs=[base_yaml, st],
            outputs=[chat, st, summary, latest_run_dir],
            api_visibility="private",
        )

        btn_send.click(
            fn=on_chat,
            inputs=[user_in, chat, st],
            outputs=[chat, st],
            api_visibility="private",
        )
        user_in.submit(
            fn=on_chat,
            inputs=[user_in, chat, st],
            outputs=[chat, st],
            api_visibility="private",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue()  # allow long-running jobs
    demo.launch()