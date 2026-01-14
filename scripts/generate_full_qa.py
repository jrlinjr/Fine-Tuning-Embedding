#!/usr/bin/env python3
"""
完整法規問答生成腳本
處理指定的 29 部重要法規
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.qa_generator import QAGenerator

# 目標法規清單
TARGET_LAWS = [
    "中華民國憲法",
    "中華民國憲法增修條文",
    "憲法訴訟法",
    "民法",
    "中華民國刑法",
    "民事訴訟法",
    "刑事訴訟法",
    "行政程序法",
    "公司法",
    "證券交易法",
    "票據法",
    "保險法",
    "海商法",
    "勞動基準法",
    "性別平等工作法",
    "勞工退休金條例",
    "全民健康保險法",
    "著作權法",
    "專利法",
    "商標法",
    "營業秘密法",
    "個人資料保護法",
    "國家賠償法",
    "社會秩序維護法",
    "道路交通管理處罰條例",
    "政府採購法",
    "警察職權行使法",
    "警察法",
    "道路交通安全基本法",
]


def main():
    parser = argparse.ArgumentParser(description="完整法規問答生成")
    parser.add_argument("--ollama-host", type=str, help="遠端 Ollama 主機")
    parser.add_argument("--qa-per-article", type=int, default=2, help="每條生成幾組問答")
    parser.add_argument("--resume-from", type=str, help="從指定法規繼續（跳過之前的）")
    args = parser.parse_args()

    # 載入設定
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化生成器
    generator = QAGenerator(
        model_name=config["qa_generation"]["ollama_model"],
        ollama_host=args.ollama_host
    )

    # 法律資料目錄
    laws_dir = Path(__file__).parent.parent / "data" / "raw" / "laws"

    # 處理的法規清單
    laws_to_process = TARGET_LAWS.copy()
    if args.resume_from:
        try:
            idx = laws_to_process.index(args.resume_from)
            laws_to_process = laws_to_process[idx:]
            print(f"從 {args.resume_from} 繼續，跳過前 {idx} 部法規")
        except ValueError:
            print(f"警告: 找不到 {args.resume_from}，從頭開始")

    print(f"=== 完整法規問答生成 ===")
    print(f"目標法規: {len(laws_to_process)} 部")
    print(f"每條問答數: {args.qa_per_article}")
    print()

    # 載入所有條文
    print("載入法律資料...")
    docs = generator.load_documents(
        str(laws_dir),
        law_filter=laws_to_process,
        exact_match=True,
        max_articles_per_law=None  # 不限制
    )

    if not docs:
        print("錯誤: 沒有找到任何法律條文")
        sys.exit(1)

    # 預估時間
    estimated_time = len(docs) * 5
    print(f"\n預估生成時間: {estimated_time // 3600} 小時 {(estimated_time % 3600) // 60} 分鐘")
    print()

    # 生成問答對
    qa_pairs = generator.generate_all_qa(num_qa_per_doc=args.qa_per_article)

    # 儲存
    output_path = Path(__file__).parent.parent / "data" / "raw" / "generated_qa.json"
    generator.save_qa_pairs(str(output_path))

    print(f"\n=== 完成 ===")
    print(f"條文數: {len(docs)}")
    print(f"問答對: {len(qa_pairs)}")
    print(f"輸出: {output_path}")


if __name__ == "__main__":
    main()
