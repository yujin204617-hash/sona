"""热点态势命令入口：执行独立的热点抓取与分析流程。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

_LOG_PATH = "/Users/biaowenhuang/Documents/sona-master/.cursor/debug.log"


def _hot_cli_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Optional[dict] = None,
) -> None:
    """Append one NDJSON line for /hot debugging (no secrets)."""
    run_id = os.environ.get("HOT_DEBUG_RUN_ID", f"hot_cli_{int(time.time() * 1000)}")
    payload = {
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Debug logs must never break the main flow.
        pass


def run_hot_command(config_path: Optional[str] = None) -> None:
    """执行热点抓取与态势感知流程，并输出结果路径。"""
    try:
        from utils.hot_topics_env import ensure_hot_topics_cwd, prepare_hot_topics_environment

        prepare_hot_topics_environment()
        ensure_hot_topics_cwd()

        # 惰性导入，避免 CLI 启动时引入重依赖（在 .env 与 cwd 就绪之后）
        from tools.hottopics import run as run_hot_topics

        # Share run id with tools/hottopics.py
        run_id = f"hot_{int(time.time() * 1000)}_{os.getpid()}"
        os.environ["HOT_DEBUG_RUN_ID"] = run_id

        # H1_ENV_MISSING: check env status without printing secrets
        # #region hot_debug_H1_CLI_ENV
        _hot_cli_debug_log(
            hypothesis_id="H1_ENV_MISSING",
            location="cli/hot_ui.py:run_hot_command",
            message="hot cli env check",
            data={
                "has_qwen_apikey": bool(os.environ.get("QWEN_APIKEY") or os.environ.get("QWEN_API_KEY") or os.environ.get("APIKEY")),
                "has_insight_engine_apikey": bool(os.environ.get("INSIGHT_ENGINE_API_KEY")),
                "insight_engine_base_url": os.environ.get("INSIGHT_ENGINE_BASE_URL", ""),
                "has_config_config_yaml": str(Path(os.environ.get("CONFIG_PATH", "config/config.yaml"))).endswith("config.yaml")
                or bool(Path("config/config.yaml").exists()),
            },
        )
        # #endregion

        normalized_config_path: Optional[str] = None
        if config_path:
            normalized_config_path = str(Path(config_path).expanduser().resolve())

        console.print()
        console.print("[bold cyan]启动热点抓取与态势感知流程...[/bold cyan]")
        if normalized_config_path:
            console.print(f"[dim]配置文件: {normalized_config_path}[/dim]")

        if not os.environ.get("INSIGHT_ENGINE_API_KEY"):
            console.print(
                "[yellow]提示: 未检测到 INSIGHT_ENGINE_API_KEY。[/yellow] "
                "请在 .env 中至少配置其一：QWEN_APIKEY、OPENAI_APIKEY、DEEPSEEK_APIKEY、KIMI_APIKEY，"
                "或显式设置 INSIGHT_ENGINE_API_KEY / QUERY_ENGINE_API_KEY。"
            )
            console.print()

        # 展示当前热点流程实际使用的 base_url（便于确认走 coding plan）
        base_url = os.environ.get("INSIGHT_ENGINE_BASE_URL") or ""
        if base_url:
            console.print(f"[dim]hot LLM base_url: {base_url}[/dim]")

        report_path = run_hot_topics(config_path=normalized_config_path)
        if report_path:
            console.print(f"[green]✅ 热点报告已生成: {report_path}[/green]")
        else:
            console.print("[yellow]流程已执行，但未返回报告路径。[/yellow]")
        console.print()
    except Exception as exc:
        # #region hot_debug_H5_EXCEPTION
        _hot_cli_debug_log(
            hypothesis_id="H5_EXCEPTION",
            location="cli/hot_ui.py:run_hot_command",
            message="hot cli exception",
            data={"error_type": type(exc).__name__, "error_message": str(exc)[:500]},
        )
        # #endregion
        console.print(f"[red]❌ 热点流程执行失败: {exc}[/red]")
