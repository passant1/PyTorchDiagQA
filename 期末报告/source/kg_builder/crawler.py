"""
网页抓取模块
负责从 PyTorch 官方文档抓取内容
"""
import json
import os
import time

from utils.path_utils import get_cache_path, get_data_path
from kg_builder.sources import OFFLINE_DOCS


def crawl_documents(sources_path: str = None, cache_dir: str = None, timeout: int = 15) -> dict:
    """
    抓取文档内容

    Args:
        sources_path: pytorch_sources.json 路径
        cache_dir: 缓存目录
        timeout: 请求超时时间

    Returns:
        dict: {source_id: text_content, ...}
    """
    if sources_path is None:
        sources_path = get_data_path("pytorch_sources.json")
    if cache_dir is None:
        cache_dir = get_cache_path()

    with open(sources_path, "r", encoding="utf-8") as f:
        sources = json.load(f)

    results = {}
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    }

    try:
        import requests
        from bs4 import BeautifulSoup
        network_available = True
    except ImportError:
        network_available = False
        print("[Crawler] requests 或 BeautifulSoup 未安装，使用离线模式")

    for source in sources:
        source_id = source["id"]
        url = source.get("url", "")
        title = source.get("title", source_id)

        print(f"[Crawler] 处理: {title} ({source_id})")

        # 1. 尝试从缓存加载
        cache_file = os.path.join(cache_dir, f"{source_id}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_text = f.read()
            if cached_text.strip():
                print(f"[Crawler] 从缓存加载: {source_id}")
                results[source_id] = cached_text
                continue

        # 2. 尝试从网络抓取
        text = ""
        if network_available:
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # 移除 script 和 style
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                # 提取正文
                body = soup.find("body") or soup
                paragraphs = []
                for tag in body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "code"]):
                    paragraphs.append(tag.get_text(strip=True))

                text = "\n".join(paragraphs)
                print(f"[Crawler] 网络抓取成功: {source_id} ({len(text)} 字符)")

            except Exception as e:
                print(f"[Crawler] 网络抓取失败: {source_id} - {e}")

        # 3. 如果网络失败，使用离线文本
        if not text.strip():
            if source_id in OFFLINE_DOCS:
                text = OFFLINE_DOCS[source_id]
                print(f"[Crawler] 使用内置离线文本: {source_id}")
            else:
                text = f"# {title}\n\n暂无离线文本。"
                print(f"[Crawler] 无离线文本: {source_id}")

        # 4. 写入缓存
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[Crawler] 缓存写入失败: {e}")

        results[source_id] = text
        time.sleep(0.5)  # 避免请求过快

    return results
