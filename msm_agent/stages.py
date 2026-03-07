from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import yaml
from openai import OpenAI
from google import genai

from msm_agent.pipeline_w_mdtraj import run_mvp


# ============================================================
# Small helpers
# ============================================================

def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def msg(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}


def safe_yaml_load(s: str) -> Dict[str, Any]:
    obj = yaml.safe_load(s)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return obj


def yaml_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    grade = metrics.get("grade", {})
    data = metrics.get("data", {})
    occupancy = metrics.get("occupancy", {})
    plateau = metrics.get("plateau", {})

    lines = [
        f"Grade: {grade.get('label', '?')}",
        f"Total time (ns): {data.get('total_time_ns', 'NA')}",
        f"dt_ns_effective: {data.get('dt_ns_effective', 'NA')}",
        f"n_trajs: {data.get('n_trajs', 'NA')}",
        f"n_used_states: {occupancy.get('n_used', 'NA')} / {occupancy.get('n_clusters', 'NA')}",
        f"ITS rel_std_max(top-k): {plateau.get('rel_std_max', 'NA')}",
    ]

    fail_reasons = grade.get("fail_reasons") or []
    warn_reasons = grade.get("warn_reasons") or []

    if fail_reasons:
        lines.append("Fail reasons: " + "; ".join(fail_reasons))
    if warn_reasons:
        lines.append("Warnings: " + "; ".join(warn_reasons))

    return "\n".join(lines)


def init_default_config() -> str:
    example_path = Path("examples/fs_peptide_mvp.yaml")
    if example_path.exists():
        return example_path.read_text()

    return yaml_dump(
        {
            "run": {"output_dir": "runs", "run_name": "test_ala2", "seed": 42},
            "data": {
                "kind": "xtc",
                "dir": "./ala2_test_data",
                "topology": "./ala2_test_data/ala2.pdb",
                "stride": 10,
                "saving_interval": 50,
                "load_preprocessed_dir": None,
            },
            "features": {"type": "distance", "selection": "distances", "atom_selection": "heavy"},
            "tica": {"lag_time_frames_range": [2, 100], "n_components": 4},
            "clustering": {"method": "kmeans", "n_clusters": 100},
            "msm": {
                "lag_time_frames_range": [2, 100],
                "n_timescales": 10,
                "reversible_type": "mle",
                "ergodic_cutoff": False,
            },
            "plots": {"bins": 90, "gridsize": 120},
            "gates": {
                "min_occupancy": 10,
                "max_tiny_state_frac": 0.30,
                "min_avg_out_degree": 3,
                "plateau_k": 3,
                "plateau_rel_var": 0.30,
            },
        }
    )


# ============================================================
# Session state for UI / agent orchestration
# ============================================================

@dataclass
class SessionState:
    base_cfg_yaml: str = ""
    base_cfg_obj: Optional[Dict[str, Any]] = None
    base_run_dir: Optional[str] = None
    base_metrics: Optional[Dict[str, Any]] = None
    last_advice_text: str = ""
    latest_summary: str = ""
    chat_history: List[Dict[str, str]] = field(default_factory=list)


# ============================================================
# Advisor abstraction
# ============================================================

class Advisor(Protocol):
    def analyze_one_run_text(self, base_cfg: Dict[str, Any], run_metrics: Dict[str, Any]) -> str:
        ...

    def chat_followup(self, chat_history: List[Dict[str, str]], st: SessionState) -> str:
        ...


class OpenAIAdvisor:
    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def analyze_one_run_text(self, base_cfg: Dict[str, Any], run_metrics: Dict[str, Any]) -> str:
        prompt = build_run_analysis_prompt(base_cfg, run_metrics)
        resp = self.client.responses.create(model=self.model, input=[prompt])
        return resp.output_text

    def chat_followup(self, chat_history: List[Dict[str, str]], st: SessionState) -> str:
        input_msgs = build_followup_context(chat_history, st)
        resp = self.client.responses.create(model=self.model, input=input_msgs)
        return resp.output_text


class GoogleAdvisor:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model = model
        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))

    def analyze_one_run_text(self, base_cfg: Dict[str, Any], run_metrics: Dict[str, Any]) -> str:
        prompt = build_run_analysis_prompt(base_cfg, run_metrics)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=json.dumps(prompt, indent=2),
        )
        return resp.text

    def chat_followup(self, chat_history: List[Dict[str, str]], st: SessionState) -> str:
        input_msgs = build_followup_context(chat_history, st)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=json.dumps(input_msgs, indent=2),
        )
        return resp.text


# ============================================================
# Prompt/context builders
# ============================================================

def build_run_analysis_prompt(base_cfg: Dict[str, Any], run_metrics: Dict[str, Any]) -> Dict[str, str]:
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

    return {
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
            "- Only suggest edits within: tica.lag_time_frames, tica.n_components, "
            "clustering.n_clusters, msm.lag_time_frames_list, msm.n_timescales.\n"
            "- Keep changes minimal; prefer one-factor-at-a-time.\n"
            "- Avoid long explanations.\n\n"
            "Current hyperparams:\n"
            f"{json.dumps(cur, indent=2)}\n\n"
            "Run signals:\n"
            f"{json.dumps({'grade': grade, 'plateau': plateau, 'occupancy': occupancy, 'sparsity': sparsity}, indent=2)}\n"
        ),
    }


def build_followup_context(chat_history: List[Dict[str, str]], st: SessionState, tail_turns: int = 8) -> List[Dict[str, str]]:
    context: List[Dict[str, str]] = []

    if st.base_metrics and st.base_cfg_obj:
        context.append(
            {
                "role": "system",
                "content": "You are an MSM validation assistant. Continue the conversation about the user's latest run.",
            }
        )
        context.append(
            {
                "role": "system",
                "content": "Latest run summary:\n" + summarize_metrics(st.base_metrics),
            }
        )
        if st.last_advice_text:
            context.append({"role": "assistant", "content": st.last_advice_text})

    def normalize_message(m: Dict[str, Any]) -> Dict[str, str]:
        return {
            "role": str(m.get("role", "user")),
            "content": str(m.get("content", "")),
        }

    tail = chat_history[-tail_turns:] if chat_history else []
    return context + [normalize_message(m) for m in tail]


# ============================================================
# Stage functions
# These are the main reusable units for later tool calling.
# Keep them small, deterministic, and easy to chain.
# ============================================================

def stage_parse_base_config(base_yaml: str) -> Dict[str, Any]:
    """Parse YAML text into a config dict."""
    return safe_yaml_load(base_yaml)



def stage_prepare_base_run_cfg(base_cfg: Dict[str, Any], run_suffix: str = "base") -> Dict[str, Any]:
    """Clone and decorate the base config for an execution run."""
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("run", {})

    base_name = cfg["run"].get("run_name", "base")
    cfg["run"]["run_name"] = f"{base_name}_{run_suffix}_{now_id()}"
    return cfg



def stage_execute_msm_run(cfg: Dict[str, Any]) -> str:
    """Execute the MVP pipeline and return the run directory."""
    run_dir = run_mvp(cfg)
    if not run_dir:
        raise RuntimeError("run_mvp returned an empty run_dir.")
    return run_dir



def stage_load_run_metrics(run_dir: str) -> Dict[str, Any]:
    """Load metrics.json from a finished run directory."""
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    return read_json(metrics_path)



def stage_build_run_summary(metrics: Dict[str, Any]) -> str:
    """Turn raw run metrics into a compact human-readable summary."""
    return summarize_metrics(metrics)



def stage_analyze_base_run(
    advisor: Advisor,
    base_cfg: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    """Ask an advisor model to analyze one completed run."""
    return advisor.analyze_one_run_text(base_cfg=base_cfg, run_metrics=metrics)



def stage_update_state_after_base_run(
    st: SessionState,
    *,
    base_yaml: str,
    base_cfg: Dict[str, Any],
    run_dir: str,
    metrics: Dict[str, Any],
    summary: str,
    advice: str,
) -> SessionState:
    """Mutate and return session state after a successful base run."""
    st.base_cfg_yaml = base_yaml
    st.base_cfg_obj = base_cfg
    st.base_run_dir = run_dir
    st.base_metrics = metrics
    st.latest_summary = summary
    st.last_advice_text = advice
    return st



def stage_run_base_pipeline(base_yaml: str, advisor: Advisor, st: Optional[SessionState] = None) -> Dict[str, Any]:
    """
    End-to-end wrapper for the first major run.

    Returns a structured dict so later Gradio tools / agent tools can consume
    the same payload without depending on UI formatting.
    """
    st = st or SessionState()

    base_cfg = stage_parse_base_config(base_yaml)
    run_cfg = stage_prepare_base_run_cfg(base_cfg, run_suffix="base")
    run_dir = stage_execute_msm_run(run_cfg)
    metrics = stage_load_run_metrics(run_dir)
    summary = stage_build_run_summary(metrics)
    advice = stage_analyze_base_run(advisor, base_cfg, metrics)
    st = stage_update_state_after_base_run(
        st,
        base_yaml=base_yaml,
        base_cfg=base_cfg,
        run_dir=run_dir,
        metrics=metrics,
        summary=summary,
        advice=advice,
    )

    system_text = f"✅ Base run finished.\n\n{summary}\n\nRun dir: {run_dir}"
    st.chat_history.append(msg("system", system_text))
    st.chat_history.append(msg("assistant", advice))

    return {
        "state": st,
        "run_dir": run_dir,
        "summary": summary,
        "advice": advice,
        "chat_messages": st.chat_history,
    }



def stage_chat_turn(
    user_msg: str,
    advisor: Advisor,
    st: SessionState,
    *,
    require_base_run: bool = True,
) -> Dict[str, Any]:
    """
    Handle a follow-up chat turn using current state.

    This is intentionally lightweight so later you can replace this with a
    planner/router/tool-calling stage without touching UI code.
    """
    st.chat_history.append(msg("user", user_msg))

    if require_base_run and (st.base_metrics is None or st.base_cfg_obj is None):
        reply = "Run the base config first so I can analyze the MSM metrics."
        st.chat_history.append(msg("assistant", reply))
        return {"state": st, "reply": reply, "chat_messages": st.chat_history}

    reply = advisor.chat_followup(st.chat_history, st)
    st.chat_history.append(msg("assistant", reply))
    return {"state": st, "reply": reply, "chat_messages": st.chat_history}


# ============================================================
# Optional factory for runtime config
# ============================================================

def make_advisor(provider: str = "google", model: Optional[str] = None) -> Advisor:
    provider_normalized = provider.strip().lower()

    if provider_normalized in {"google", "gemini"}:
        return GoogleAdvisor(model=model or "gemini-2.5-flash")
    if provider_normalized in {"openai", "gpt"}:
        return OpenAIAdvisor(model=model or "gpt-5.2")

    raise ValueError(f"Unsupported advisor provider: {provider}")
