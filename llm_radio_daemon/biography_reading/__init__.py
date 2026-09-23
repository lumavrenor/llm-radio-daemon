"""偉人伝トーク（type = "biography_reading"）。

Wikipedia記事を figures_file の明示指定で取り上げ、ローリング要約を
持ち越しながら先頭から順にMCが解説していく。literary_reading（読書コーナー）と
同じセッション管理の考え方だが、朗読ではなく「全チャンクがLLM入力」になる点が違う。
"""
