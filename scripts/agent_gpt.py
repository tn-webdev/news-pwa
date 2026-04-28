import os
import re
import requests
import feedparser
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from translation_parser import parse_translation_response

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "docs" / "data" / "summary_v2.json"

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# .env 読み込み
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

RSS_SOURCES = {
    "ai": {
        "source": "VentureBeat",
        "url": "https://venturebeat.com/feed/"
    },
    "economy": {
        "source": "NHK",
        "url": "https://www3.nhk.or.jp/rss/news/cat5.xml"
    },
    "world": {
        "source": "BBC",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml"
    },
    "japan_politics": {
        "source": "NHK",
        "url": "https://www3.nhk.or.jp/rss/news/cat3.xml"
    }
}

# ********** RSS取得 **********
def fetch_rss(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    feed = feedparser.parse(resp.content)
    entries = feed.entries
        
    # 記事が空でない時だけ1件
    return feed.entries[0] if feed.entries else None

def fetch_rss_ai_multiple(url, max_items=2):
    """
    AIカテゴリ専用：OpenAI/ChatGPT関連を優先しつつ、
    最大 max_items 件のニュースを返す関数。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    feed = feedparser.parse(resp.content)
    entries = feed.entries

    if not entries:
        return []

    keywords = ["openai", "chatgpt", "sam altman", "gpt", "large language model"]

    # 優先記事（OpenAI/ChatGPT関連）を先に取得
    priority_items = []
    normal_items = []

    for e in entries:
        text = (e.title + " " + e.get("summary", "")).lower()
        if any(k in text for k in keywords):
            priority_items.append(e)
        else:
            normal_items.append(e)

    # 優先 → 通常 の順で max_items 件取り出す
    combined = priority_items + normal_items
    return combined[:max_items]

def deduplicate_articles(entries):
    seen_links = set()
    seen_titles = set()
    unique_entries = []
    duplicate_count = 0

    for entry in entries:
        link = (getattr(entry, "link", "") or "").strip()
        title = (getattr(entry, "title", "") or "").strip()

        if link in seen_links or title in seen_titles:
            duplicate_count += 1
            continue

        seen_links.add(link)
        seen_titles.add(title)
        unique_entries.append(entry)

    print(f"⚠️ 重複記事を{duplicate_count}件除外しました")
    return unique_entries

# ********** title・summary抽出 / 要約（約150文字、3文程度） **********
# 日本語要約
def summarize(text, title=""):
    date_rule = (
            "要約では年号（例：2023年、2025年など）を使用しないでください。"
            "日付が必要な場合は「今年」「最近」「9月末時点」などの相対表現のみを使ってください。"
    )
 
    if not text or text.strip() == "":
        prompt = (
            "次のニュースタイトルから、記事の内容を推測して3行の要約文を生成してください。\n"
            f"{date_rule}\n"
            f"タイトル: {title}\n"
        )
    else:
            prompt = (
                f"{date_rule}\n"
                f"次の記事を3行で要約してください：\n{text}"
            )
    
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = res.choices[0].message.content

    # content が文字列の場合
    if isinstance(content, str):
        return content

    # content が配列（MessageContent）で返る場合（将来の仕様変更対策）
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        # text属性を持つタイプ
        if hasattr(first, "text"):
            return first.text
        # 万が一 text がなくても string_value がある
        if hasattr(first, "string_value"):
            return first.string_value

    # それでもダメなら、とりあえず文字列化して返す
    return str(content)

# 英語要約
def summarize_en(text, title=""):
    if not text or text.strip() == "":
        prompt = (
            "Summarize the following news title in about 3 sentences.\n"
            f"Title: {title}\n"
        )
    else:
        prompt = (
            "Summarize the following article in about 3 sentences:\n"
            f"{text}"
        )

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

# ********** 英語タイトル＆要約 → 日本語翻訳 **********
def translate_to_japanese(title_en: str, summary_en: str):
    """
    VentureBeat / BBC など英語記事専用。
    タイトルと要約をまとめて日本語ニュース文体に翻訳する。
    うまくパースできなければ、元の英語をそのまま返す。
    """
    system_prompt = (
        "You are a professional Japanese news editor.\n"
        "Translate the provided English title and summary into clear, natural Japanese "
        "suitable for news readers. Preserve meaning strictly, avoid embellishment, "
        "and maintain factual accuracy.\n"
        "Return the result as JSON with keys: title_ja, summary_ja."
    )

    user_prompt = f"""
Translate the following text into Japanese.

Title:
{title_en}

Summary:
{summary_en}
""".strip()

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = res.choices[0].message.content
        return parse_translation_response(content)
    except Exception as e:
        return {
            "translated_title_ja": "",
            "translated_summary_ja": "",
            "error": str(e)
        }

# ********** カテゴリ判定（AI / 経済 / その他） **********
# 3分類: AI / 経済 / その他
def classify_category(title: str, summary: str, feed_key: str, description: str = "", debug: bool = False) -> str:
    key = (feed_key or "").strip().lower()
    text = f"{title} {description} {summary}"
    t = text.lower()

    # AIマーカー（said等の誤検出抑制）
    has_ai_marker = bool(re.search(r"(?i)(?<![a-z])ai(?![a-z])", text)) or ("人工知能" in text)

    # 経済は、金融に特化しているワードだけにする（汎用語は入れない）
    finance_strong = [
        # JP
        "株", "株価", "銘柄", "決算", "業績", "投資", "相場", "バブル",
        "急騰", "暴落", "反発", "調整", "時価総額",
        "金利", "利上げ", "利下げ", "国債", "債券", "利回り",
        "為替", "円高", "円安",
        "日経", "日経平均", "topix", "ダウ", "ナスダック", "s&p",
        "ipo", "etf",

        # EN
        "stock", "stocks", "share", "shares", "equity", "earnings", "revenue",
        "invest", "investor", "fund", "ipo", "etf", "valuation",
        "interest rate", "bond", "treasury", "yield",
        "nasdaq", "dow", "s&p", "nikkei", "topix", "forex", "fx",
    ]
    finance_hits = [k for k in finance_strong if k in t]

    # AI（A方針：生成AI/LLM中心 + AIインフラ）
    ai_core = [
        "chatgpt", "openai", "gpt", "llm", "large language model",
        "anthropic", "claude", "gemini", "deepseek",
        "生成ai", "大規模言語モデル", "基盤モデル",
    ]
    ai_infra = [
        "hbm", "dram", "memory", "メモリ", "gpu", "nvidia", "cuda",
        "datacenter", "data center", "データセンター",
        "training", "inference", "推論", "学習",
        "ai向け", "ai用", "ai対応"
    ]
    is_ai = any(k in t for k in ai_core) or (has_ai_marker and any(k in t for k in ai_infra))

    # --- デバッグ（誤爆ワードを見る）---
    if debug:
        print("DEBUG classify:", {"feed_key": key, "finance_hits": finance_hits, "is_ai": is_ai, "title": title})

    # economyフィードは経済固定
    if key == "economy":
        return "経済"

    # aiフィード（VentureBeatなど）は基本AI固定
    # ただし「AI関連銘柄/AIバブル」のような金融語が出た場合は、経済へ
    if key == "ai":
        if finance_hits:
            return "経済"
        return "AI"

    # 他フィードは「金融が強い→経済」「AI主題→AI」「それ以外→その他」
    if len(finance_hits) >= 2 or (finance_hits and has_ai_marker):
        return "経済"
    if is_ai:
        return "AI"
    return "その他"

# ********** timestamp生成 **********
def format_timestamp(entry):
    """
    published を優先し、なければ updated を使用する。
    いずれも取得不可/パース失敗時は現在時刻（UTC）を用い、
    JST（+09:00）へ変換して 'YYYY-MM-DD HH:MM' 形式で出力する。
    """
    try:
        if hasattr(entry, "published"):
            dt = feedparser._parse_date(entry.published)
        elif hasattr(entry, "updated"):
            dt = feedparser._parse_date(entry.updated)
        else:
            dt = None
    except:
        dt = None

    # pubDate取得失敗 → 今の日時を使う
    if dt is None:
        dt_obj = datetime.now(timezone.utc)
    else:
        dt_obj = datetime(*dt[:6], tzinfo=timezone.utc)

    # JSTへ変換
    jst = dt_obj.astimezone(timezone(timedelta(hours=9)))

    # フォーマット
    return jst.strftime("%Y-%m-%d %H:%M")

def main():
    output_items = []

    for category, info in RSS_SOURCES.items():
        print(f"\n🔁 [{info['source']}] RSS取得中...")

        # --- AIカテゴリは 2件ロジック ---
        if category == "ai":
            entries = fetch_rss_ai_multiple(info["url"], max_items=2)
        else:
            # --- それ以外のカテゴリは通常1件 ---
            entry = fetch_rss(info["url"])
            if not entry:
                print(f"⚠️ {info['source']} のRSSが取得できませんでした。")
                continue
            entries = [entry]  # ← 1件をリスト化して統一処理にする

        entries = deduplicate_articles(entries)

        # --- entriesの共通処理 ---
        for entry in entries:
            title = entry.title
            link = entry.link
            description = entry.summary if hasattr(entry, "summary") else ""

            print(f"🧠 [{info['source']}] 要約中...")
            summary = summarize(description, title)

            title_ja = title
            summary_ja = ""
            title_en = ""
            summary_en = ""

            if info["source"] in ["VentureBeat", "BBC"]:
                title_en = title
                summary_en = summary
            
            # 英語記事（VentureBeat / BBC）のみ日本語翻訳をかける
            if info["source"] in ["VentureBeat", "BBC"]:
                print(f"🧠 [{info['source']}] 英語要約中...")
                summary_en = summarize_en(description, title)
                title_en = title

                print(f"🌐 [{info['source']}] 日本語翻訳中...")
                translation = translate_to_japanese(title_en, summary_en)

                if translation["error"] == "":
                    title_ja = translation["translated_title_ja"] or title
                    summary_ja = translation["translated_summary_ja"] or summary_en
                else:
                    title_ja = "翻訳に失敗しました"
                    summary_ja = "翻訳に失敗しました。原文は英語フィールドを参照してください。"

            # 日本語記事
            else:
                print(f"🧠 [{info['source']}] 日本語要約中...")
                summary_ja = summarize(description, title)
           
            # カテゴリ判定（title + description + summary で判定）
            category_final = classify_category(title, summary, category, description)
            
            # VentureBeatはすべて「AI」に固定
            if info.get("source") == "VentureBeat":
                category_final = "AI"

            timestamp = format_timestamp(entry)

            output_items.append({
                "source": info['source'],
                "title": title_ja,        # 日本語タイトル
                "title_en": title_en,     # 英語タイトル（英語記事のみ、それ以外は空文字）
                "summary": summary_ja,    # 日本語要約
                "summary_en": summary_en, # 英語要約（英語記事のみ、それ以外は空文字）
                "link": link,
                "category": category_final,
                "publishedAt": timestamp
            })

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)
 
    print(f"\n✅ 複数ニュースまとめて {os.path.basename(DATA_PATH)} を生成しました！")

if __name__ == "__main__":
    main()
