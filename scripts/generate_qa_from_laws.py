#!/usr/bin/env python3
"""
從法律資料生成問答對
Step 1: 讀取法律 JSON → LLM 生成問答對

使用範例:
    # 測試模式（少量資料）
    python scripts/generate_qa_from_laws.py --test

    # 指定法規（只處理民法、刑法）
    python scripts/generate_qa_from_laws.py --filter 民法 刑法

    # 完整處理（所有法規）
    python scripts/generate_qa_from_laws.py --full
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.qa_generator import QAGenerator


def main():
    parser = argparse.ArgumentParser(description="從法律資料生成問答對")
    parser.add_argument("--test", action="store_true", help="測試模式：只處理 5 部法規，每部 3 條")
    parser.add_argument("--full", action="store_true", help="完整模式：處理所有法規")
    parser.add_argument("--filter", nargs="+", help="只處理這些法規（精確匹配，如：民法 刑法）")
    parser.add_argument("--contains", nargs="+", help="處理名稱包含關鍵字的法規（如：勞動 會包含勞動基準法等）")
    parser.add_argument("--max-laws", type=int, default=50, help="最多處理幾部法規 (預設 50)")
    parser.add_argument("--max-articles", type=int, default=10, help="每部法規最多幾條 (預設 10)")
    parser.add_argument("--qa-per-article", type=int, default=2, help="每條生成幾組問答 (預設 2)")
    parser.add_argument("--ollama-host", type=str, help="遠端 Ollama 主機 (如: http://192.168.1.100:11434)")
    args = parser.parse_args()

    # 載入設定
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 法律資料目錄
    laws_dir = Path(__file__).parent.parent / "data" / "raw" / "laws"

    # 初始化生成器
    generator = QAGenerator(
        model_name=config["qa_generation"]["ollama_model"],
        ollama_host=args.ollama_host
    )

    # 設定載入參數
    if args.test:
        max_laws = 5
        max_articles = 3
        print("=== 測試模式 ===")
    elif args.full:
        max_laws = None
        max_articles = None
        print("=== 完整模式（處理所有法規）===")
    else:
        max_laws = args.max_laws
        max_articles = args.max_articles

    # 處理過濾參數
    law_filter = args.filter or args.contains
    exact_match = args.filter is not None  # --filter 用精確匹配，--contains 用包含匹配

    if args.filter:
        print(f"精確匹配法規: {args.filter}")
    elif args.contains:
        print(f"包含匹配法規: {args.contains}")

    # 載入法律文件
    print(f"\n從 {laws_dir} 載入法律資料...")
    docs = generator.load_documents(
        str(laws_dir),
        max_laws=max_laws,
        max_articles_per_law=max_articles,
        law_filter=law_filter,
        exact_match=exact_match
    )

    if not docs:
        print("錯誤: 沒有找到任何法律條文")
        sys.exit(1)

    # 預估時間
    estimated_time = len(docs) * 6  # 約 6 秒/條
    print(f"\n預估生成時間: {estimated_time // 60} 分 {estimated_time % 60} 秒")

    # 使用 LLM 生成問答對
    print(f"\n使用 {config['qa_generation']['ollama_model']} 生成問答對...")
    qa_pairs = generator.generate_all_qa(num_qa_per_doc=args.qa_per_article)

    # 儲存問答對
    output_path = Path(__file__).parent.parent / "data" / "raw" / "generated_qa.json"
    generator.save_qa_pairs(str(output_path))

    # 顯示統計
    print(f"\n=== 生成完成 ===")
    print(f"法律條文數: {len(docs)}")
    print(f"問答對數量: {len(qa_pairs)}")
    print(f"輸出檔案: {output_path}")

    # 顯示範例
    print("\n=== 問答範例 ===")
    for qa in qa_pairs[:3]:
        print(f"\n[{qa.category}] Q: {qa.question}")
        print(f"A: {qa.answer[:100]}...")


if __name__ == "__main__":
    main()
