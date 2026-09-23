"""翻訳朗読コーナー（type = "translated_reading"）。docs/idea-translated-reading.md 案1。

Project Gutenberg（英語）の原文を、literary_reading のように直接TTSへ渡すのではなく、
チャンクごとにLLMで日本語へ訳しながらナレーションする。biography_reading と同じ
「全チャンクがLLM入力」の構造に、literary_reading の Beat（感想パート）を組み合わせた形。
"""
