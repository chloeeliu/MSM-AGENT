from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml
from openai import OpenAI

# ===== Adjust imports to your real project layout =====
from msm_agent.stage import (
    run_stage1_featurization,
    run_stage2_tica_scan,
    run_stage3_tica_fit,
)


# ----------------------------
# YAML / config helpers
# ----------------------------
def safe_yaml_load(s: str) -> Dict[str, Any]:
    obj = yaml.safe_load(s)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return obj


def yaml_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def set_nested_key(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def get_nested_key(cfg: Dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    cur = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(path)
        cur = cur[p]
    return cur


def init_default_config() -> str:
    example_path = Path("examples/ala2_mvp.yaml")
    if example_path.exists():
        return example_path.read_text()

    fallback = {
        "run": {
            "output_dir": "runs",
            "run_name": "ala2_test",
            "seed": 42,
        },
        "data": {
            "kind": "xtc",
            "dir": "/path/to/xtc_dir",
            "topology": "/path/to/topology.pdb",
            "stride": 1,
            "saving_interval": 1.0,
            "load_preprocessed_dir": None,
        },
        "features": {
            "type": "angle",
            "selection": ["phi", "psi"],
            "atom_selection": "BACKBONE",
        },
        "tica": {
            "lag_time_frames_range": [1, 10],
            "lag_time_frames_grid_size": 20,
            "n_components": 2,
            "selected_lag_time": None,
            "selected_n_components": None,
        },
        "plots": {
            "gridsize": 40,
            "bins": 50,
        },
        "gates": {
            "plateau_k": 3,
            "plateau_last_step": 3,
            "min_occupancy": 5,
        },
        "clustering": {
            "method": "KMeans",
            "n_clusters": 50,
        },
        "msm": {
            "lag_time_frames_range": [1, 10],
            "n_timescales": 5,
            "reversible_type": "mle",
            "ergodic_cutoff": 1.0,
        },
    }
    return yaml_dump(fallback)


# ----------------------------
# Session state
# ----------------------------
@dataclass
class SessionState:
    current_cfg_yaml: str = ""
    current_cfg_obj: Optional[Dict[str, Any]] = None

    current_run_dir: Optional[str] = None
    current_stage: str = "init"

    latest_summary: str = ""
    latest_plot_path: Optional[str] = None

    # optional: keep tool events for debugging
    tool_log: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------
# LLM agent
# ----------------------------
CLIENT = OpenAI()
MODEL = "gpt-5.2"

SYSTEM_PROMPT = """You are an MSM pipeline agent for a multi-stage MD workflow.

Your role:
- Help the user iteratively run Stage 1 -> Stage 2 -> Stage 3.
- Default behavior is sequential:
  1) Stage 1: featurization
  2) Stage 2: tICA scan
  3) Stage 3: final tICA fit
- If the user asks to modify config, use update_config_value first, then rerun the relevant stage.
- Do not rewrite the whole YAML unless necessary. Prefer update_config_value.
- After each tool result, summarize clearly and ask the user what they want to do next.

Important rules:
- Stage 2 requires Stage 1 output already exists in the current run_dir.
- Stage 3 requires Stage 1 output already exists, and tica.selected_lag_time must be set.
- If the user says 'ok', 'continue', or 'next', usually move to the next stage without editing config.
- If the user asks to change feature settings, update config and rerun Stage 1.
- If the user asks to change tICA scan settings, update config and rerun Stage 2.
- If the user asks to change final tICA parameters, update config and rerun Stage 3.
- Keep responses concise, practical, and stage-aware.
"""

TOOLS = [
    {
        "type": "function",
        "name": "get_current_status",
        "description": "Get current workflow status, current stage, current run_dir, latest summary, and latest plot path.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_current_config",
        "description": "Get the current config object.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_config_value",
        "description": "Update one config field by dotted path. value_yaml can be a scalar, list, dict, string, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "value_yaml": {"type": "string"},
            },
            "required": ["path", "value_yaml"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage1",
        "description": "Run Stage 1: load data and featurize.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage2",
        "description": "Run Stage 2: tICA lag scan using the latest Stage 1 result in current_run_dir.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage3",
        "description": "Run Stage 3: final tICA fit using current_run_dir and current config.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]


# ----------------------------
# Tool executor
# ----------------------------
def tool_get_current_status(st: SessionState) -> Dict[str, Any]:
    return {
        "current_stage": st.current_stage,
        "current_run_dir": st.current_run_dir,
        "latest_summary": st.latest_summary,
        "latest_plot_path": st.latest_plot_path,
    }


def tool_get_current_config(st: SessionState) -> Dict[str, Any]:
    return copy.deepcopy(st.current_cfg_obj or {})


def tool_update_config_value(st: SessionState, path: str, value_yaml: str) -> Dict[str, Any]:
    if st.current_cfg_obj is None:
        raise ValueError("No current config loaded.")
    value = yaml.safe_load(value_yaml)
    set_nested_key(st.current_cfg_obj, path, value)
    st.current_cfg_yaml = yaml_dump(st.current_cfg_obj)
    return {
        "ok": True,
        "updated_path": path,
        "new_value": value,
    }


def tool_run_stage1(st: SessionState) -> Dict[str, Any]:
    if st.current_cfg_obj is None:
        raise ValueError("No current config loaded.")
    result = run_stage1_featurization(st.current_cfg_obj)
    st.current_run_dir = result["run_dir"]
    st.current_stage = "stage1_done"
    st.latest_summary = result["summary"]
    st.latest_plot_path = result.get("plot_path")
    return result


def tool_run_stage2(st: SessionState) -> Dict[str, Any]:
    if st.current_cfg_obj is None:
        raise ValueError("No current config loaded.")
    if not st.current_run_dir:
        raise ValueError("No current_run_dir found. Please run Stage 1 first.")
    result = run_stage2_tica_scan(st.current_cfg_obj, st.current_run_dir)
    st.current_stage = "stage2_done"
    st.latest_summary = result["summary"]
    st.latest_plot_path = result.get("plot_path")
    return result


def tool_run_stage3(st: SessionState) -> Dict[str, Any]:
    if st.current_cfg_obj is None:
        raise ValueError("No current config loaded.")
    if not st.current_run_dir:
        raise ValueError("No current_run_dir found. Please run Stage 1 first.")
    result = run_stage3_tica_fit(st.current_cfg_obj, st.current_run_dir)
    st.current_stage = "stage3_done"
    st.latest_summary = result["summary"]
    st.latest_plot_path = result.get("plot_path")
    return result


def execute_tool(st: SessionState, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_current_status":
        result = tool_get_current_status(st)
    elif name == "get_current_config":
        result = tool_get_current_config(st)
    elif name == "update_config_value":
        result = tool_update_config_value(
            st,
            path=args["path"],
            value_yaml=args["value_yaml"],
        )
    elif name == "run_stage1":
        result = tool_run_stage1(st)
    elif name == "run_stage2":
        result = tool_run_stage2(st)
    elif name == "run_stage3":
        result = tool_run_stage3(st)
    else:
        raise ValueError(f"Unknown tool: {name}")

    st.tool_log.append(
        {
            "tool": name,
            "args": args,
            "result": result,
        }
    )
    return result


# ----------------------------
# Agent loop
# ----------------------------
def normalize_chat_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "value" in content and isinstance(content["value"], str):
            return content["value"]
        return json.dumps(content, ensure_ascii=False)

    if isinstance(content, list):
        parts = [normalize_chat_content(x) for x in content]
        return "\n".join([p for p in parts if p])

    return str(content)

def to_llm_messages(chat_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    msgs = []
    for m in chat_history or []:
        role = str(m.get("role", "user"))
        if role not in {"system", "user", "assistant", "developer"}:
            continue

        text = normalize_chat_content(m.get("content", ""))

        msgs.append(
            {
                "role": role,
                "content": text,
            }
        )
    return msgs


def run_agent_once(
    user_message: str,
    chat_history: List[Dict[str, str]],
    yaml_text: str,
    st: SessionState,
) -> Tuple[List[Dict[str, str]], SessionState, str, str, str, Optional[str]]:
    # 1) Sync YAML editor -> session config
    try:
        cfg_obj = safe_yaml_load(yaml_text)
    except Exception as e:
        chat_history = chat_history or []
        chat_history.append({"role": "assistant", "content": f"Config YAML parse error: {e}"})
        return (
            chat_history,
            st,
            yaml_text,
            st.latest_summary,
            st.current_run_dir or "",
            st.latest_plot_path,
        )

    st.current_cfg_obj = cfg_obj
    st.current_cfg_yaml = yaml_text

    # 2) Append user message to UI chat history
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})

    # 3) Build LLM inputs
    context_text = (
    f"Current stage: {st.current_stage}\n"
    f"Current run_dir: {st.current_run_dir}\n"
    f"Latest summary:\n{st.latest_summary or 'None'}\n"
    f"Latest plot path: {st.latest_plot_path or 'None'}\n"
)

    input_msgs = [
    {
        "role": "developer",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "system",
        "content": context_text,
    },
    ] + to_llm_messages(chat_history)


    # 4) Initial model call
    response = CLIENT.responses.create(
        model=MODEL,
        input=input_msgs,
        tools=TOOLS,
    )

    # 5) Tool-calling loop
    max_loops = 8
    loops = 0
    while loops < max_loops:
        loops += 1
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            break

        tool_outputs = []
        for fc in function_calls:
            try:
                args = json.loads(fc.arguments or "{}")
            except Exception:
                args = {}

            try:
                result = execute_tool(st, fc.name, args)
                output = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as e:
                output = json.dumps(
                    {"ok": False, "error": str(e)},
                    ensure_ascii=False,
                )

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": output,
                }
            )

        response = CLIENT.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOLS,
        )

    assistant_text = response.output_text or "Done."

    chat_history.append({"role": "assistant", "content": assistant_text})

    # 6) Sync config back to YAML editor in case tool updated it
    yaml_text = st.current_cfg_yaml or yaml_text

    return (
        chat_history,
        st,
        yaml_text,
        st.latest_summary,
        st.current_run_dir or "",
        st.latest_plot_path,
    )


# ----------------------------
# UI
# ----------------------------
def build_app():
    with gr.Blocks(title="MSM Agent MVP") as demo:
        st = gr.State(SessionState())

        gr.Markdown("## MSM Agent MVP\nLLM-driven sequential workflow: Stage 1 → Stage 2 → Stage 3")

        with gr.Row():
            with gr.Column(scale=1):
                chat = gr.Chatbot(label="Chat", height=560)
                user_in = gr.Textbox(
                    label="Message",
                    placeholder='Examples: "run stage 1", "change feature to distance", "ok continue", "set selected lag to 3 and run stage 3"',
                )
                btn_send = gr.Button("Send")

            with gr.Column(scale=1):
                cfg_editor = gr.Code(
                    label="Current config (YAML)",
                    language="yaml",
                    value=init_default_config(),
                )
                latest_summary = gr.Textbox(label="Latest summary", lines=12)
                current_run_dir = gr.Textbox(label="Current run dir")
                latest_image = gr.Image(label="Latest plot", type="filepath", height=320)

        btn_send.click(
            fn=run_agent_once,
            inputs=[user_in, chat, cfg_editor, st],
            outputs=[chat, st, cfg_editor, latest_summary, current_run_dir, latest_image],
            api_visibility="private",
        )

        user_in.submit(
            fn=run_agent_once,
            inputs=[user_in, chat, cfg_editor, st],
            outputs=[chat, st, cfg_editor, latest_summary, current_run_dir, latest_image],
            api_visibility="private",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue()
    demo.launch()