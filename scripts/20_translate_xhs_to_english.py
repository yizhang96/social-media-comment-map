from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]


LOCATION_MAP = {
    "中国香港": "Hong Kong",
    "湖北": "Hubei",
    "四川": "Sichuan",
    "美国": "United States",
    "澳大利亚": "Australia",
    "加拿大": "Canada",
    "英国": "United Kingdom",
    "法国": "France",
    "德国": "Germany",
    "日本": "Japan",
    "韩国": "South Korea",
    "新加坡": "Singapore",
    "格鲁吉亚": "Georgia",
    "西班牙": "Spain",
    "卡塔尔": "Qatar",
    "北京": "Beijing",
    "上海": "Shanghai",
    "广东": "Guangdong",
    "浙江": "Zhejiang",
    "江苏": "Jiangsu",
    "山东": "Shandong",
    "河南": "Henan",
    "河北": "Hebei",
    "湖南": "Hunan",
    "福建": "Fujian",
    "重庆": "Chongqing",
    "天津": "Tianjin",
    "陕西": "Shaanxi",
    "安徽": "Anhui",
    "江西": "Jiangxi",
    "辽宁": "Liaoning",
    "广西": "Guangxi",
    "云南": "Yunnan",
    "贵州": "Guizhou",
    "山西": "Shanxi",
    "黑龙江": "Heilongjiang",
    "吉林": "Jilin",
    "内蒙古": "Inner Mongolia",
    "新疆": "Xinjiang",
    "甘肃": "Gansu",
    "海南": "Hainan",
    "宁夏": "Ningxia",
    "青海": "Qinghai",
    "西藏": "Tibet",
    "中国台湾": "Taiwan",
    "中国澳门": "Macau",
}


SYSTEM_PROMPT = """You translate Xiaohongshu social media comments from Chinese to natural public-facing English.
Preserve the speaker's meaning, tone, emphasis, slang, and emoji.
Do not add explanations. Do not censor. Do not summarize.
If a comment is already English, return it unchanged except for obvious punctuation cleanup.
Return strict JSON only."""


def translate_batch(client: OpenAI, model: str, items: list[dict]) -> list[dict]:
    payload = json.dumps(items, ensure_ascii=False)
    prompt = (
        "Translate each item's text field to English. Return a JSON array with the same "
        "ids and one field named translated_text.\n\n"
        f"Input JSON:\n{payload}"
    )

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Return exactly this object shape: "
                            '{"translations":[{"id":1,"translated_text":"..."}]}.\n'
                            + prompt
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content or ""
            data = json.loads(content)
            translations = data.get("translations")
            if not isinstance(translations, list):
                raise ValueError("Missing translations array")
            return translations
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("unreachable")


def normalize_time(value):
    if pd.isna(value):
        return value
    s = str(value).strip()
    s = s.replace("昨天", "Yesterday ")
    s = re.sub(r"^(\d+)分钟前$", r"\1 minutes ago", s)
    s = re.sub(r"^(\d+)小时前$", r"\1 hours ago", s)
    s = re.sub(r"^1天前$", "1 day ago", s)
    s = re.sub(r"^(\d+)天前$", r"\1 days ago", s)
    if s == "刚刚":
        return "Just now"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xhs_2026-02-07_studyabroad")
    parser.add_argument("--model", default=os.environ.get("TRANSLATION_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    path = ROOT / "data" / "datasets" / args.dataset / "processed" / "comments_cleaned.xlsx"
    archive_dir = ROOT / "archive_data" / f"{args.dataset}_chinese" / "translation_work"
    archive_dir.mkdir(parents=True, exist_ok=True)
    backup_path = archive_dir / "comments_cleaned_chinese_backup.xlsx"
    translations_path = archive_dir / "comments_translation_en.json"

    df = pd.read_excel(path).copy()
    if "comment_text" not in df.columns:
        raise KeyError("comments_cleaned.xlsx must contain comment_text")

    if not backup_path.exists():
        df.to_excel(backup_path, index=False)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    translated_by_id = {}
    if translations_path.exists():
        existing = json.loads(translations_path.read_text(encoding="utf-8"))
        translated_by_id.update({int(k): v for k, v in existing.items()})

    rows = []
    for i, text in enumerate(df["comment_text"].fillna("").astype(str).tolist(), start=1):
        if i not in translated_by_id:
            rows.append({"id": i, "text": text})

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        if not batch:
            continue
        translations = translate_batch(client, args.model, batch)
        expected = {item["id"] for item in batch}
        received = set()
        for item in translations:
            item_id = int(item["id"])
            if item_id in expected:
                translated_by_id[item_id] = str(item["translated_text"]).strip()
                received.add(item_id)
        missing = expected - received
        if missing:
            raise RuntimeError(f"Missing translations for ids: {sorted(missing)}")
        translations_path.write_text(
            json.dumps(translated_by_id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Translated {min(start + len(batch), len(rows))}/{len(rows)} remaining rows")

    df["comment_text"] = [
        translated_by_id[i] for i in range(1, len(df) + 1)
    ]

    if "location_raw" in df.columns:
        df["location_raw"] = df["location_raw"].map(
            lambda x: LOCATION_MAP.get(str(x), str(x)) if pd.notna(x) else x
        )

    if "time_raw" in df.columns:
        df["time_raw"] = df["time_raw"].map(normalize_time)

    if "raw" in df.columns:
        df = df.drop(columns=["raw"])

    df.to_excel(path, index=False)
    print(f"Wrote translated workbook: {path}")
    print(f"Saved translation cache: {translations_path}")


if __name__ == "__main__":
    main()
