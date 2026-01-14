#!/usr/bin/env python3
"""
嵌入模型訓練腳本
執行對比學習微調
"""
import sys
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.pair_generator import PairGenerator
from src.contrastive_trainer import ContrastiveTrainer, TrainerConfig


def main():
    # 載入設定
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 載入訓練配對
    pairs_dir = Path(__file__).parent.parent / config["paths"]["pairs_dir"]
    pairs_file = pairs_dir / "training_pairs.json"

    if not pairs_file.exists():
        print("錯誤: 找不到訓練配對檔案")
        print("請先執行: python scripts/generate_pairs.py")
        sys.exit(1)

    generator = PairGenerator()
    training_pairs = generator.load_pairs(str(pairs_file))

    if len(training_pairs) == 0:
        print("錯誤: 沒有訓練配對")
        sys.exit(1)

    # 分割訓練/評估資料 (90/10)
    split_idx = int(len(training_pairs) * 0.9)
    train_pairs = training_pairs[:split_idx]
    eval_pairs = training_pairs[split_idx:] if split_idx < len(training_pairs) else None

    print(f"訓練樣本: {len(train_pairs)}")
    if eval_pairs:
        print(f"評估樣本: {len(eval_pairs)}")

    # 設定訓練器
    output_dir = Path(__file__).parent.parent / config["paths"]["output_dir"]
    trainer_config = TrainerConfig(
        model_name=config["model"]["name"],
        output_dir=str(output_dir),
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        num_epochs=config["training"]["num_epochs"],
        warmup_ratio=config["training"]["warmup_ratio"],
        use_mps=config["hardware"].get("use_mps", False),
        use_cuda=config["hardware"].get("use_cuda", True)
    )

    trainer = ContrastiveTrainer(trainer_config)

    # 載入基底模型並顯示微調前的相似度
    trainer.load_model()
    print("\n=== 微調前相似度 ===")
    trainer.compare_before_after(training_pairs, num_samples=2)

    # 執行訓練
    trainer.train(train_pairs, eval_pairs)

    # 顯示微調後的相似度
    print("\n=== 微調後相似度 ===")
    trainer.compare_before_after(training_pairs, num_samples=2)

    print("\n訓練完成！")
    print(f"模型已儲存至: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
