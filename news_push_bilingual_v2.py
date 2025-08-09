#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 每日推送：全球热点(自动翻译) + 国内热点 到微信（Server酱）
------------------------------------------------------------
改动要点：
- 使用 feedparser 优先解析 RSS（兼容性更强），失败再退回到 BeautifulSoup。
- 更新更稳定的新闻源（BBC/CNN/Reuters；新华社/央视/澎湃）。
- 加入请求重试、超时、UA 标头与简单限流，减少偶发失败。
- 翻译链：OpenAI -> DeepL -> MyMemory（均可选；无密钥也能跑）。
- 输出严格 Markdown，避免被微信折叠；链接显示域名。
运行：
  pip install requests beautifulsoup4 feedparser
  export SERVERCHAN_SENDKEY=你的Key
  python news_push_bilingual_v2.py
"""

import os
import time
import json
import html
import requests
import feedparser
from urllib.parse import urlparse, quote
from datetime import datetime, timezone
from bs4 import BeautifulSoup

TITLE = "今日热点简报｜全球 + 国内"
TOP_K_PER_SOURCE = int(os.getenv("TOP_K_PER_SOURCE", "6"))

# ==== 新闻源（可自行增删）====
GLOBAL_RSS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",       # BBC World
    "http://rss.cnn.com/rss/edition_world.rss",         # CNN World
    "https://feeds.reuters.com/reuters/worldNews",      # Reuters World
]
CHINA_RSS = [
    "http://www.news.cn/rss/politics.xml",              # 新华社 时政
    "https://news.cctv.com/data/rss/newsChina.xml",     # 央视 国内
    "https://www.thepaper.cn/rss.jsp?nodeid=25434",     # 澎湃 国内要闻
]

UA = os.getenv("HTTP_UA", "Mozilla/5.0 (NewsPushBot/2.0; +https://github.com/)")
TIMEOUT = 20
RETRIES = 2
SLEEP_BETWEEN = 0.6  # 每个请求之间的间隔，避免触发限流

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
session.mount("http://", adapter)
session.mount("https://", adapter)
session.headers.update({"User-Agent": UA})

def host_of(link: str) -> str:
    try:
        return urlparse(link).netloc or "source"
    except Exception:
        return "source"

def get_text(obj, default=""):
    try:
        return (obj or "").strip()
    except Exception:
        return default

def fetch_via_feedparser(url, topk):
    try:
        d = feedparser.parse(url)
        items = []
        for e in d.entries[:topk]:
            title = get_text(e.get("title", ""))
            link = get_text(e.get("link", ""))
            pub = get_text(e.get("published", "")) or get_text(e.get("updated", ""))
            items.append((title, link, pub, host_of(link)))
        return items
    except Exception:
        return []

def fetch_via_bs4(url, topk):
    try:
        r = session.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        items = []
        for item in soup.find_all("item")[:topk]:
            title = get_text(item.title.text if item.title else "")
            link = get_text(item.link.text if item.link else "")
            pub = get_text(item.pubDate.text if item.pubDate else "")
            items.append((title, link, pub, host_of(link)))
        if not items:
            for entry in soup.find_all("entry")[:topk]:
                title = get_text(entry.title.text if entry.title else "")
                link_tag = entry.find("link")
                link = get_text(link_tag.get("href") if link_tag else "")
                pub = get_text(entry.updated.text if entry.find("updated") else "")
                items.append((title, link, pub, host_of(link)))
        return items
    except Exception:
        return []

def fetch_rss_items(url, topk):
    for i in range(RETRIES + 1):
        items = fetch_via_feedparser(url, topk)
        if not items:
            items = fetch_via_bs4(url, topk)
        if items:
            return items
        time.sleep(1.2 + 0.3 * i)
    return [("【抓取失败】%s" % url, "", "", "error")]

def dedup(items):
    seen = set()
    out = []
    for t, l, p, h in items:
        key = (t.strip(), h)
        if key in seen or not t.strip():
            continue
        seen.add(key)
        out.append((t, l, p, h))
    return out

# ===== 翻译模块 =====
def translate_openai(texts):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = "将以下英文新闻标题逐条翻译成简洁的中文（保留专有名词），只返回JSON数组，不要其他文字：\n" + json.dumps(texts, ensure_ascii=False)
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    try:
        resp = session.post(url, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        arr = json.loads(content)
        if isinstance(arr, list) and len(arr) == len(texts):
            return [str(x).strip() for x in arr]
        return None
    except Exception:
        return None

def translate_deepl(texts):
    key = os.getenv("DEEPL_API_KEY")
    if not key:
        return None
    url = "https://api-free.deepl.com/v2/translate"
    outs = []
    try:
        for t in texts:
            data = {"auth_key": key, "text": t, "target_lang": "ZH"}
            r = session.post(url, data=data, timeout=20)
            r.raise_for_status()
            outs.append(r.json()["translations"][0]["text"])
            time.sleep(0.35)
        return outs if len(outs) == len(texts) else None
    except Exception:
        return None

def translate_mymemory(texts):
    outs = []
    for t in texts:
        try:
            url = f"https://api.mymemory.translated.net/get?q={quote(t)}&langpair=en|zh-CN"
            r = session.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            outs.append(data.get("responseData", {}).get("translatedText", t))
            time.sleep(0.5)
        except Exception:
            outs.append(t)
    return outs

def auto_translate(texts):
    if not texts:
        return []
    def looks_chinese(s):
        for ch in s[:8]:
            if "\u4e00" <= ch <= "\u9fff":
                return True
        return False
    if all(looks_chinese(t) for t in texts):
        return texts
    outs = translate_openai(texts)
    if outs:
        return outs
    outs = translate_deepl(texts)
    if outs:
        return outs
    return translate_mymemory(texts)

# ===== 发送 =====
def send_serverchan(title, markdown):
    send_key = os.getenv("SERVERCHAN_SENDKEY")
    if not send_key:
        raise RuntimeError("缺少环境变量 SERVERCHAN_SENDKEY")
    url = f"https://sctapi.ftqq.com/{send_key}.send"
    data = {"title": title, "desp": markdown}
    r = session.post(url, data=data, timeout=25)
    r.raise_for_status()
    return r.text

# ===== 渲染 =====
def build_markdown(globals_items, china_items, translated):
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"**{TITLE}**  \n更新：{now}\n")

    lines.append("### 🌍 全球热点（已译）")
    for i, (item, zh) in enumerate(zip(globals_items, translated), 1):
        t, link, _, h = item
        t = html.escape(t)
        zh = html.escape(zh)
        if link:
            lines.append(f"{i}. {zh}  \n    *{t}*  \n    [{h}]({link})")
        else:
            lines.append(f"{i}. {zh}  \n    *{t}*")
    lines.append("")

    lines.append("### 🇨🇳 国内热点")
    for i, (t, link, _, h) in enumerate(china_items, 1):
        t = html.escape(t)
        if link:
            lines.append(f"{i}. {t}  \n    [{h}]({link})")
        else:
            lines.append(f"{i}. {t}")
    return "\n".join(lines)

def main():
    g_items = []
    for src in GLOBAL_RSS:
        g_items += fetch_rss_items(src, TOP_K_PER_SOURCE)
        time.sleep(SLEEP_BETWEEN)
    g_items = dedup(g_items)

    c_items = []
    for src in CHINA_RSS:
        c_items += fetch_rss_items(src, TOP_K_PER_SOURCE)
        time.sleep(SLEEP_BETWEEN)
    c_items = dedup(c_items)

    g_titles = [x[0] for x in g_items]
    translated = auto_translate(g_titles)
    if len(translated) != len(g_items):
        translated = g_titles

    md = build_markdown(g_items, c_items, translated)
    resp = send_serverchan(TITLE, md)
    print(resp)

if __name__ == "__main__":
    main()
