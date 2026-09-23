"""ロギング設定。24時間稼働でログを肥大化させないよう RotatingFileHandler を使う。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s"


def setup_logging(
    log_dir: str = "logs", level: str = "INFO", filename: str = "llm_radio_daemon.log"
) -> None:
    """ロガーを構成する。

    ``filename`` は放送プロセスと執筆バッチ（``generated_drama_writer``）でファイルを分けるための逃げ道。
    RotatingFileHandler は2プロセスから同じファイルを開くとローテーションで衝突するので、
    放送本体が別プロセスとして起動する執筆バッチには別名を渡す。
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        Path(log_dir) / filename,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
