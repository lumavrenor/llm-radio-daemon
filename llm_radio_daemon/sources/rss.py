"""RSS/Atom フィードソース（v3）。

SPEC.md 4.1(c): config で指定した任意のフィードURLを巡回する。
単独ではネタが枯れるため、他ソースと併用する前提（呼び出し側の責任）。
ポーリング間隔は最低60秒。フィード単位で失敗しても他のフィードは続行する。

本文の取得は3段構え（下へ行くほど重い / 精度は上がるが壊れやすい）:
  1. フィードの content:encoded / description  … _entry_body_text
  2. 記事ページの og:description 系 meta        … _page_description_from_html
  3. 記事ページの本文コンテナからの抽出         … extract_article_body
配信元によって RSS が全文を載せるところ（1で十分）と、1文しか載せないところ
（animeanime / natalie 等。3が要る）がある。3はドメインごとに DOM が違うので
_BODY_HINTS のルール表で当てる。未知ドメインは「<p> が最も多い塊」を拾う。
"""

from __future__ import annotations

import html
import logging
import re
import threading
from typing import Iterator
from urllib.parse import urlparse

import feedparser
import requests

from ..source_status import SourceStatus
from . import USER_AGENT, Topic

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
# タグを空白へ潰した名残で句読点の前に空白が残る（"…bold</b>." → "…bold ."）。
# 読み上げで不自然なので、閉じ括弧・句読点の直前の空白だけ寄せる。
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([。、．，.!?！？」』）)\]])")
# 1フィードあたり何件まで見るか。少なすぎるとフィードの更新が遅い時間帯に
# 新着ゼロが続いてフィラーだけになるので、config の max_entries_per_feed で上書きできる。
_DEFAULT_MAX_ENTRIES_PER_FEED = 10
# LLM へ渡す本文の上限。content:encoded に全文を載せるフィードだと数千字になるため、
# ここで頭打ちにしてコンテキストを守る（要約フィードでは元から短いので影響しない）。
# config の max_body_chars で上書きできる。
_DEFAULT_MAX_BODY_CHARS = 2000
# これ未満なら「フィードが本文を持っていない」とみなし、記事ページ側を取りに行く
# （enrich_thin_bodies=True のとき）。animeanime.jp や natalie.mu は description が
# 空〜1文しか無く、卓が見出しだけで話す羽目になっていた。閾値は「見出しの言い換え
# 1文」より少し長いくらい。
_THIN_BODY_CHARS = 80
# 記事ページ側から拾う meta。長いものを採る（og:description が name=description より
# 詳しいサイトがある）。属性の並び順はサイト差があるので tag 全体を見てから content= を抜く。
_DESC_META_RE = re.compile(
    r'(?:property|name)\s*=\s*["\'](?:og:description|twitter:description|description)["\']',
    re.I,
)
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.I)
_CONTENT_ATTR_RE = re.compile(r'content\s*=\s*["\']([^"\']*)["\']', re.I)

# --- 本文抽出（3段目） -----------------------------------------------------
# 抽出前に丸ごと消す要素（本文コンテナの中に居ても要らない）。
_DROP_BLOCK_RE = re.compile(
    r"(?is)<(script|style|noscript|svg|template|iframe|form|figure|figcaption|aside|nav|header|footer)\b[^>]*>.*?</\1>"
)
_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")
# 本文として拾う block 要素。<p> を主役に、見出し・箇条書きも拾う。
_BLOCK_TEXT_RE = re.compile(r"(?is)<(p|h2|h3|h4|li|blockquote)\b[^>]*>(.*?)</\1>")
# 汎用フォールバック（コンテナ不明のドメイン）は <p> だけ。<li>/<h*> はナビ率が高い。
_P_TEXT_RE = re.compile(r"(?is)<p\b[^>]*>(.*?)</p>")
# 抽出テキストがこの語以降になったら打ち切る（アフィリエイト枠・関連リンク・
# クレジット・SNS ボタン・関連記事群の島）。行頭一致ではなく最初の出現で切る。
_STOP_MARKERS = (
    "関連リンク",
    "関連記事",
    "あわせて読みたい",
    "この記事の画像",
    "この記事にあるおすすめのポイント",
    "編集部おすすめのニュース",
    "編集部おすすめ",
    "アンケート募集記事",
    "注目記事",
    "特集・インタビュー",
    "このマンガ、もう読んだ？",
    "アクセスランキング",
    "深掘りコンテンツ",
    "最新・注目の動画",
    "ご利用にあたって",
    "(C)",
    "（C）",
    "(c)",
    "Copyright",
    "価格・在庫状況は記事公開時点",
    "Amazon 楽天市場",
    "AmazonでこのDVD",
    "編集部にメッセージを送る",
    "外部サイト",
    "《",  # animeanime の "《animeanime》" 署名
    "©",
    "Source:",
    "参考記事",
    "この記事にはアフィリエイトリンク",
    "みんなの ポスト コピー",
)
# 関連記事リストの1行（末尾が "N月N日" や " HH:MM" の見出し）。
_HEADLINE_LIST_RE = re.compile(r"(?:\d{1,2}月\d{1,2}日|\d{1,2}:\d{2})\s*$")
# 本文でない定型行（ナビ・SNS ボタン・埋め込みカードの見出し）。完全一致・部分一致。
_JUNK_LINE_EXACT = frozenset(
    {
        "Global Sites",
        "コメント を書く コメント を見る",
        "みんなの ポスト コピー",
        "しまむらの投稿を見る",
        "ねとらぼの人気記事",
        "読まれている記事",
        "この記事をシェア",
    }
)
_JUNK_LINE_SUBSTR = ("｜ ライフハッカー", "| ライフハッカー", "｜ ギズモード", "| ギズモード", "［高画質で見る］")


def _is_junk_line(line: str) -> bool:
    if line in _JUNK_LINE_EXACT or any(s in line for s in _JUNK_LINE_SUBSTR):
        return True
    if _HEADLINE_LIST_RE.search(line):
        return True
    # 句点が無く短い行 = 見出し・ラベル・キャプションの崩れ。
    return len(line) < 40 and "。" not in line
# ドメイン別の本文コンテナ候補。(tag, class トークン または id) を優先度順に並べ、
# 本文中で最初に見つかった開始タグから、対応する閉じタグまでを厳密に切り出す。
# ここに無いドメインは汎用フォールバック（<body> 内の <p> をすべて拾う）へ。
# host が "<key>" と一致 or ".<key>" で終わればそのルール。長い key を優先。
_BODY_HINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "animeanime.jp": (("article", "arti-body"),),
    "natalie.mu": (("div", "NA_article_body"),),
    "gigazine.net": (("div", "article"),),
    "nlab.itmedia.co.jp": (("div", "post-contents"),),
    "itmedia.co.jp": (("div", "cmsBody"), ("div", "l-block__main")),
    "lifehacker.jp": (("div", "article_pArticle_Body"),),
    "gizmodo.jp": (("div", "article_pArticle_Body"),),
    "karapaia.com": (("div", "entry-content"),),
    "watch.impress.co.jp": (("div", "text"),),
    "4gamer.net": (("div", "entry-content"), ("div", "maintxt")),
    "automaton-media.com": (("div", "entry-content"),),
    # nhk.or.jp は本文が JS 描画で HTML に無いため入れない（meta / RSS 本文で足りる）。
}
_MIN_LINE_CHARS = 8  # これ未満の行は見出し崩れ・キャプション扱いで捨てる
_MAX_CONTAINER_BYTES = 200_000  # コンテナの閉じタグ探索の上限

# タイトルにこの語を含む記事はネタにしない（config の title_exclude で上書き）。
# 「◯話 反応まとめ」「【ネタバレ】感想まとめ」系は、それ自体が引用の寄せ集めで
# トーク素材として弱く、小型モデルだと引用をそのまま台詞にしてしまう（qwen3.5:4b で確認）。
_DEFAULT_TITLE_EXCLUDE = ("ネタバレ", "反応まとめ", "感想まとめ", "実況まとめ")


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def _html_to_text(raw: str) -> str:
    """フィードの本文（多くは HTML 片）をプレーンテキストへ均す。

    タグは空白へ置換してから消す（<br>・</p> をまたいで単語がくっつかないように）。
    &amp; などの実体参照も戻す。最後に空白を1個へ圧縮。
    """
    if not raw:
        return ""
    text = _clean(html.unescape(_TAG_RE.sub(" ", raw)))
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def _entry_body_text(entry) -> str:
    """フィードエントリの本文候補を選ぶ。content:encoded を最優先。

    RSS の <content:encoded> / Atom の <content> は feedparser では entry.content
    （dict のリスト）。無いフィードは多く、その場合は <description> / <summary>
    （feedparser はどちらも entry.summary に入れる）へフォールバックする。
    どの経路でも HTML を含みうるので _html_to_text で均す。
    """
    for block in entry.get("content") or ():
        text = _html_to_text(block.get("value", ""))
        if text:
            return text
    return _html_to_text(entry.get("summary", entry.get("description", "")))


def _decoded_html(resp: requests.Response) -> str:
    """記事ページのバイト列を正しい文字コードで復号する。

    requests は Content-Type に charset が無いと text/* を ISO-8859-1 とみなすため、
    Shift_JIS / EUC-JP の日本語サイト（ITmedia・4Gamer など）が文字化けする。
    ヘッダに charset が無い or フォールバック値なら、中身から推定した encoding を使う。
    """
    enc = (resp.encoding or "").lower()
    if not enc or enc == "iso-8859-1":
        enc = (resp.apparent_encoding or "utf-8").lower()
    try:
        return resp.content.decode(enc, errors="replace")
    except (LookupError, TypeError):
        return resp.text


def _hint_rules(url: str) -> tuple[tuple[str, str], ...]:
    """URL のホストに合う _BODY_HINTS のルールを返す（最長一致のドメインを優先）。"""
    host = urlparse(url).netloc.lower().split(":")[0]
    best_key = ""
    for key in _BODY_HINTS:
        if (host == key or host.endswith("." + key)) and len(key) > len(best_key):
            best_key = key
    return _BODY_HINTS.get(best_key, ())


# 本文コンテナ内に埋め込まれる「関連記事・レコメンド」ブロックの class トークン。
# 対応する閉じ <div> まで balanced に消す（4Gamer の CARDVIEW など）。
_JUNK_CONTAINER_TOKENS = (
    "CARDVIEW",
    "related",
    "related-articles",
    "p-related",
    "recommend",
    "recommendation",
    "outbrain",
    "taboola",
    "article-related",
)


def _strip_balanced_div(markup: str, token: str) -> str:
    pat = re.compile(
        rf'(?is)<div\b[^>]*?class\s*=\s*["\'][^"\']*(?<![\w-]){re.escape(token)}(?![\w-])[^"\']*["\'][^>]*>'
    )
    while True:
        m = pat.search(markup)
        if not m:
            return markup
        start, depth = m.end(), 1
        cut = len(markup)
        for t in re.finditer(r"(?is)<(/?)div\b[^>]*?(/?)>", markup[start:]):
            if t.group(1):
                depth -= 1
                if depth == 0:
                    cut = start + t.end()
                    break
            elif not t.group(2):
                depth += 1
        markup = markup[: m.start()] + " " + markup[cut:]


def _strip_noise(markup: str) -> str:
    markup = _COMMENT_RE.sub(" ", markup)
    markup = _DROP_BLOCK_RE.sub(" ", markup)
    for token in _JUNK_CONTAINER_TOKENS:
        if token.lower() in markup.lower():
            markup = _strip_balanced_div(markup, token)
    return markup


def _container_html(markup: str, tag: str, token: str) -> str:
    """<tag> で class トークン == token か id == token の要素を、対応する閉じ
    タグまで厳密に切り出す（同名タグのネストを数える）。見つからなければ ""。"""
    open_re = re.compile(
        rf'(?is)<{tag}\b[^>]*?'
        rf'(?:class\s*=\s*["\'][^"\']*(?<![\w-]){re.escape(token)}(?![\w-])[^"\']*["\']'
        rf'|id\s*=\s*["\']{re.escape(token)}["\'])[^>]*>'
    )
    m = open_re.search(markup)
    if not m:
        return ""
    start = m.end()
    depth = 1
    scan_end = min(len(markup), start + _MAX_CONTAINER_BYTES)
    for t in re.finditer(rf"(?is)<(/?){tag}\b[^>]*?(/?)>", markup[start:scan_end]):
        if t.group(1):  # </tag>
            depth -= 1
            if depth == 0:
                return markup[start : start + t.start()]
        elif not t.group(2):  # <tag ...> 非自己完結
            depth += 1
    return markup[start:scan_end]


def _lines_from_blocks(markup: str, block_re: re.Pattern = _BLOCK_TEXT_RE) -> list[str]:
    lines: list[str] = []
    for m in block_re.finditer(markup):
        inner = m.group(m.lastindex)
        text = _html_to_text(inner)
        if len(text) < _MIN_LINE_CHARS:
            continue
        if lines and lines[-1] == text:  # 画像キャプションの繰り返しを潰す
            continue
        lines.append(text)
    return lines


_LINE_BREAK_RE = re.compile(r"(?is)<(?:br|/p|/div|/h[1-6]|/li|/tr|/blockquote)\s*/?>")


def _flatten_lines(markup: str) -> list[str]:
    """<p> を使わず <br> 区切りで本文を書くサイト（4Gamer 等）向け。

    ブロックの終わりと <br> を改行に変えてからタグを落とし、行に割る。
    """
    text = _LINE_BREAK_RE.sub("\n", markup)
    text = html.unescape(_TAG_RE.sub(" ", text))
    out: list[str] = []
    for raw in text.split("\n"):
        line = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", _clean(raw))
        if len(line) >= _MIN_LINE_CHARS and (not out or out[-1] != line):
            out.append(line)
    return out


_RELATED_STAMP_RE = re.compile(r"^［.*\d{4}/\d{1,2}/\d{1,2}.*］$")


def _drop_related_headlines(lines: list[str]) -> list[str]:
    """関連記事カード（見出しの直後に「［2025/09/27 21:56］」の行が続く）を落とす。

    4Gamer の本文コンテナには関連記事の埋め込みが混ざる。タイムスタンプ行と、
    その直前の見出し行（1〜2行）をまとめて捨てる。
    """
    drop: set[int] = set()
    for i, line in enumerate(lines):
        if _RELATED_STAMP_RE.match(line):
            drop.add(i)
            for j in (i - 1, i - 2):
                if j >= 0 and j not in drop and not _is_junk_line(lines[j]):
                    drop.add(j)
    return [ln for k, ln in enumerate(lines) if k not in drop]


def _skip_leading_junk(lines: list[str]) -> list[str]:
    """先頭のナビ・ラベル行（ページ内リンク・日付・バイライン）を落とす。

    本文っぽい行（40字以上 or 句点入り）が来た時点で残りをそのまま返す
    （本文の途中に出る短い小見出しは消さない）。
    """
    for i, line in enumerate(lines):
        if not _is_junk_line(line):
            return lines[i:]
    return []


def _trim_trailing_junk(lines: list[str]) -> list[str]:
    """末尾の関連記事リスト・クレジット行の連続（3行以上）を落とす。"""
    end = len(lines)
    while end > 0 and _is_junk_line(lines[end - 1]):
        end -= 1
    return lines[:end] if len(lines) - end >= 3 else lines


def _leading_prose_run(lines: list[str]) -> list[str]:
    """先頭のナビを飛ばし、本文の連続した段落だけを取る（最初のナビ行で打ち切り）。

    コンテナ不明のドメイン向け。<body> 直下の <p> を全部拾うとナビ・フッタも
    混ざるので、「本文が始まったら、次にナビ行が出るまで」を本文とみなす。
    """
    out: list[str] = []
    for line in lines:
        if _is_junk_line(line):
            if out:
                break
            continue
        out.append(line)
    return out


def _cut_at_stop_marker(text: str) -> str:
    cut = len(text)
    for marker in _STOP_MARKERS:
        i = text.find(marker)
        if 0 <= i < cut:
            cut = i
    return text[:cut].strip()


def _prose_len(lines: list[str]) -> int:
    """散文らしい行（句点入り or 長め）の文字数合計。見出し・ラベルだけの塊を弾く。"""
    return sum(len(ln) for ln in lines if "。" in ln or len(ln) >= 50)


def _looks_like_article(text: str) -> bool:
    """汎用フォールバックの結果が本文っぽいか（ナビの羅列を弾く）。

    句点「。」は日本語の目印。hackernews.py が英語サイトにも同じ関数を使うため、
    ピリオド終わりの行も文として認める（ナビの羅列はどちらも持たない）。
    """
    if text.count("\n") < 1:
        return False
    return any(
        len(ln) >= 60 and ("。" in ln or ". " in ln or ln.rstrip().endswith("."))
        for ln in text.split("\n")
    )


def extract_article_body(page_html: str, url: str, max_chars: int = _DEFAULT_MAX_BODY_CHARS) -> str:
    """記事ページ HTML から本文だけを抜く。失敗時は ""。

    1. ドメイン別ヒント（_BODY_HINTS）で本文コンテナを対応閉じタグまで厳密に切り、
       <p>/<h2-4>/<li> のテキストを改行でつなぐ。
    2. ヒントが無い/外れたら汎用フォールバック（<body> 内の <p> を全部）。ただし
       結果が本文っぽくなければ（ナビの羅列など）捨てて "" を返す。
    どちらも末尾の関連記事リストを落とし、_STOP_MARKERS で打ち切る。
    """
    if not page_html or "<" not in page_html:
        return ""
    markup = _strip_noise(page_html)

    lines: list[str] = []
    for tag, token in _hint_rules(url):
        container = _container_html(markup, tag, token)
        if not container:
            continue
        # まず <p> ベースで拾う。それがほぼ空なら（本文を <p> で囲まない 4Gamer
        # のようなサイト）<br> 平坦化にフォールバックする。<p> が取れているのに
        # 平坦化に乗り換えると関連カードまで巻き込むので、閾値を超えたら採用。
        def _tidy(raw: list[str]) -> list[str]:
            return _trim_trailing_junk(_skip_leading_junk(_drop_related_headlines(raw)))

        block = _tidy(_lines_from_blocks(container))
        if _prose_len(block) >= 200:
            lines = block
        else:
            flat = _tidy(_flatten_lines(container))
            lines = flat if _prose_len(flat) > _prose_len(block) else block
        if len("".join(lines)) >= 100:
            break
        lines = []

    if not lines:
        body_m = re.search(r"(?is)<body\b[^>]*>", markup)
        tail = markup[body_m.end() :] if body_m else markup
        generic = _leading_prose_run(_lines_from_blocks(tail, _P_TEXT_RE))
        text = _cut_at_stop_marker("\n".join(generic))[:max_chars].strip()
        return text if _looks_like_article(text) else ""

    text = _cut_at_stop_marker("\n".join(lines))
    return text[:max_chars].strip()


class RssSource:
    name = "rss"

    def __init__(
        self,
        urls: list[str],
        poll_interval_sec: float = 300.0,
        stop_event: threading.Event | None = None,
        timeout_sec: int = 10,
        max_entries_per_feed: int = _DEFAULT_MAX_ENTRIES_PER_FEED,
        max_body_chars: int = _DEFAULT_MAX_BODY_CHARS,
        enrich_thin_bodies: bool = True,
        fetch_article_body: bool = False,
        title_exclude: list[str] | None = None,
        status: SourceStatus | None = None,
    ):
        self._urls = urls
        self._base_poll_interval_sec = max(poll_interval_sec, 60.0)
        self._stop_event = stop_event or threading.Event()
        self._timeout_sec = timeout_sec
        self._max_entries_per_feed = max(1, max_entries_per_feed)
        self._max_body_chars = max(200, max_body_chars)
        # タイトルにこれらの語を含む記事はスキップ（None で既定リスト、[] で無効）。
        self._title_exclude = tuple(
            _DEFAULT_TITLE_EXCLUDE if title_exclude is None else title_exclude
        )
        # 本文の無いフィード（animeanime / natalie 等）で、記事ページの meta description を
        # 1 回だけ取りに行って本文を補う。フィードだけで完結させたいなら config で false。
        self._enrich_thin_bodies = enrich_thin_bodies
        # meta description の1文でも足りないとき、記事ページの本文コンテナから全文を抜く
        # （_BODY_HINTS）。既定 off（重い・壊れやすい）。
        self._fetch_article_body = fetch_article_body
        self._status = status

    def fetch(self) -> Iterator[Topic]:
        headers = {"User-Agent": USER_AGENT}
        while not self._stop_event.is_set():
            entries = 0
            for url in self._urls:
                if self._stop_event.is_set():
                    return
                try:
                    resp = requests.get(url, headers=headers, timeout=self._timeout_sec)
                    resp.raise_for_status()
                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries[: self._max_entries_per_feed]:
                        if self._stop_event.is_set():
                            return
                        topic = self._to_topic(entry)
                        if topic is not None:
                            entries += 1
                            yield topic
                except requests.RequestException as e:
                    if self._status is not None:
                        self._status.fetch_failed(url, e)
                    else:
                        logger.warning("rss fetch failed for %s: %s", url, e)

            # 全フィードを1周した。ここで必ず1行残す（新着ゼロでも WARNING が出る）ので、
            # ログが途切れていたらスレッドが死んでいる、と切り分けられる。
            if self._status is not None:
                self._status.cycle_end(entries, self._base_poll_interval_sec)
            if self._stop_event.wait(self._base_poll_interval_sec):
                return

    def _to_topic(self, entry) -> Topic | None:
        link = entry.get("link") or entry.get("id")
        title = _clean(entry.get("title", ""))
        if not link or not title:
            return None
        if any(word in title for word in self._title_exclude):
            logger.debug("rss: skipping due to title exclude word %r", title)
            return None
        body = _entry_body_text(entry)
        if len(body) < _THIN_BODY_CHARS and (self._enrich_thin_bodies or self._fetch_article_body):
            page_html = self._get_page(link)
            if self._fetch_article_body:
                extracted = extract_article_body(page_html, link, self._max_body_chars)
                if len(extracted) > len(body):
                    body = extracted
            if self._enrich_thin_bodies and len(body) < _THIN_BODY_CHARS:
                enriched = self._page_description_from_html(page_html)
                if len(enriched) > len(body):
                    body = enriched

        return Topic(
            source=self.name,
            external_id=link,
            title=title,
            body=(body or title)[: self._max_body_chars],
            url=link,
            hint="最近のニュース",
        )

    def _get_page(self, url: str) -> str:
        """記事ページを1回だけ GET して HTML 文字列を返す（失敗は ""）。"""
        if not url.lower().startswith(("http://", "https://")):
            return ""
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=min(self._timeout_sec, 8),
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug("rss: failed to fetch article page %s: %s", url, e)
            return ""
        return _decoded_html(resp)

    @staticmethod
    def _page_description_from_html(page_html: str) -> str:
        """og:description 系 meta の中で最長のものを返す（本文抽出はしない）。"""
        best = ""
        for tag in _META_TAG_RE.findall(page_html or ""):
            if not _DESC_META_RE.search(tag):
                continue
            m = _CONTENT_ATTR_RE.search(tag)
            if m:
                cand = _html_to_text(m.group(1))
                if len(cand) > len(best):
                    best = cand
        return best
