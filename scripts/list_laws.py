#!/usr/bin/env python3
"""
列出法規資料庫中的所有法規名稱
方便查詢正確的法規名稱用於 --filter
"""
import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="查詢法規名稱")
    parser.add_argument("keyword", nargs="?", help="搜尋關鍵字（如：刑法、勞動）")
    parser.add_argument("--all", action="store_true", help="列出所有法規")
    args = parser.parse_args()

    laws_file = Path(__file__).parent.parent / "data" / "raw" / "laws" / "laws.json"

    with open(laws_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    laws = data.get("Laws", [])

    if args.keyword:
        # 搜尋包含關鍵字的法規
        matches = [law["LawName"] for law in laws if args.keyword in law.get("LawName", "")]
        print(f"包含「{args.keyword}」的法規 ({len(matches)} 筆):\n")
        for name in matches:
            print(f"  {name}")
    elif args.all:
        # 列出所有法規
        print(f"共 {len(laws)} 部法規:\n")
        for law in laws:
            print(f"  {law.get('LawName')} ({law.get('LawCategory')})")
    else:
        # 顯示統計
        categories = {}
        for law in laws:
            cat = law.get("LawCategory", "未分類")
            categories[cat] = categories.get(cat, 0) + 1

        print(f"法規資料庫統計：共 {len(laws)} 部法規\n")
        print("依類別分類：")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:15]:
            print(f"  {cat}: {count} 部")

        print("\n使用方式：")
        print("  python scripts/list_laws.py 刑法      # 搜尋包含「刑法」的法規")
        print("  python scripts/list_laws.py 勞動      # 搜尋包含「勞動」的法規")
        print("  python scripts/list_laws.py --all    # 列出所有法規")


if __name__ == "__main__":
    main()
