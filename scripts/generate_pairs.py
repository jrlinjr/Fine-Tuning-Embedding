#!/usr/bin/env python3
"""
配對生成腳本
Step 2: 從問答對生成對比學習訓練配對（正負樣本）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data_loader import LegalQADataLoader
from src.pair_generator import PairGenerator


def main():
    # 載入設定
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 檢查是否有生成的問答對
    data_dir = Path(__file__).parent.parent / config["paths"]["data_dir"]
    generated_qa = data_dir / "generated_qa.json"

    loader = LegalQADataLoader(str(data_dir))

    if generated_qa.exists():
        print(f"載入生成的問答對: {generated_qa}")
        qa_pairs = loader.load_json("generated_qa.json")
    else:
        # 回退到範例資料
        print("未找到生成的問答對，請先執行: python scripts/generate_qa_from_laws.py")
        sample_file = data_dir / "sample_qa.json"
        if not sample_file.exists():
            loader.save_sample_template("sample_qa.json")
        qa_pairs = loader.load_json("sample_qa.json")

    # 初始化配對生成器
    pair_config = config["pair_generation"]
    generator = PairGenerator(model_name=pair_config["ollama_model"])

    # 生成訓練配對
    print("\n開始生成訓練配對...")
    training_pairs = generator.generate_training_pairs(
        qa_pairs,
        num_positive_variants=pair_config["num_positive_variants"],
        num_hard_negatives=pair_config["num_hard_negatives"]
    )

    # 儲存配對
    pairs_dir = Path(__file__).parent.parent / config["paths"]["pairs_dir"]
    output_path = pairs_dir / "training_pairs.json"
    generator.save_pairs(str(output_path))

    # 顯示範例
    print("\n=== 生成的配對範例 ===")
    for i, pair in enumerate(training_pairs[:3], 1):
        print(f"\n配對 {i}:")
        print(f"  錨點: {pair.anchor[:60]}...")
        print(f"  正樣本: {pair.positive[:60]}...")
        print(f"  負樣本: {pair.negative[:60]}...")

    print(f"\n總共生成 {len(training_pairs)} 組訓練配對")
    print(f"已儲存至: {output_path}")


if __name__ == "__main__":
    main()
