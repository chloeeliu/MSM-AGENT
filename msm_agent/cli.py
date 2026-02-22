import argparse
from pathlib import Path
from msm_agent.config import load_config
from msm_agent.pipeline import run_mvp

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--config", required=True, help="Path to YAML config")

    args = p.parse_args()
    if args.cmd == "run":
        cfg = load_config(Path(args.config))
        out = run_mvp(cfg)
        print(f"[msm-agent] done: {out}")

if __name__ == "__main__":
    main()