<img width="2736" height="1706" alt="image" src="https://github.com/user-attachments/assets/02441092-9f14-4fca-8a10-8426fa24b2c3" />

# MSM Agent Pipeline
An interactive human-in-the-loop pipeline for iterative chemistry experiment exploration and optimization.
The system combines automated parameter sweeps, structured result summarization, and LLM-assisted planning to guide experiment refinement over multiple cycles.

# Overview
```
User → Base Config
        ↓
Config Generator / Runner
        ↓
Raw Experiment Results
        ↓
Summarizer → runs_summary.json
        ↓
LLM Planner → plan_json + explanation
        ↓
User Feedback / Edits
        ↓
Next Batch Execution
        ↺ (loop)
```

# Setup

```
python app_gradio.py
```
    
