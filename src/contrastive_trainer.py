"""
對比學習訓練器
使用 Sentence Transformers 進行嵌入模型微調
"""
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset

from .pair_generator import TrainingPair


@dataclass
class TrainerConfig:
    """訓練設定"""
    model_name: str = "BAAI/bge-base-zh-v1.5"
    output_dir: str = "models/finetuned"
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    use_mps: bool = False   # Apple Silicon MPS
    use_cuda: bool = True   # NVIDIA GPU


class ContrastiveTrainer:
    """對比學習嵌入模型訓練器"""

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.model: Optional[SentenceTransformer] = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """取得運算裝置"""
        if self.config.use_cuda and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"使用 CUDA 加速: {gpu_name}")
            return "cuda"
        elif self.config.use_mps and torch.backends.mps.is_available():
            print("使用 MPS (Metal Performance Shaders) 加速")
            return "mps"
        else:
            print("使用 CPU")
            return "cpu"

    def load_model(self):
        """載入基底嵌入模型"""
        print(f"載入模型: {self.config.model_name}")
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.device
        )
        print(f"模型維度: {self.model.get_sentence_embedding_dimension()}")
        return self.model

    def prepare_dataset(self, pairs: list[TrainingPair]) -> Dataset:
        """準備訓練資料集 (Triplet 格式)"""
        data = {
            "anchor": [p.anchor for p in pairs],
            "positive": [p.positive for p in pairs],
            "negative": [p.negative for p in pairs]
        }
        dataset = Dataset.from_dict(data)
        print(f"訓練樣本數: {len(dataset)}")
        return dataset

    def train(self, pairs: list[TrainingPair], eval_pairs: Optional[list[TrainingPair]] = None):
        """執行對比學習訓練"""
        if self.model is None:
            self.load_model()

        # 準備資料集
        train_dataset = self.prepare_dataset(pairs)

        # 設定 Loss: TripletLoss
        # 讓 anchor 接近 positive，遠離 negative
        train_loss = losses.TripletLoss(model=self.model)

        # 訓練參數
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            fp16=False,  # MPS 不支援 fp16
            bf16=False,
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=False,
        )

        # 設定 Evaluator（如果有評估資料）
        evaluator = None
        if eval_pairs:
            evaluator = TripletEvaluator(
                anchors=[p.anchor for p in eval_pairs],
                positives=[p.positive for p in eval_pairs],
                negatives=[p.negative for p in eval_pairs],
                name="legal-qa-eval"
            )

        # 建立 Trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )

        # 開始訓練
        print("\n開始訓練...")
        print(f"  - 裝置: {self.device}")
        print(f"  - Batch Size: {self.config.batch_size}")
        print(f"  - Learning Rate: {self.config.learning_rate}")
        print(f"  - Epochs: {self.config.num_epochs}")
        print()

        trainer.train()

        # 儲存模型
        final_path = output_dir / "final"
        self.model.save(str(final_path))
        print(f"\n模型已儲存至: {final_path}")

        return self.model

    def evaluate_similarity(self, text1: str, text2: str) -> float:
        """計算兩個文本的相似度"""
        if self.model is None:
            raise ValueError("請先載入或訓練模型")

        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        )
        return similarity.item()

    def compare_before_after(self, pairs: list[TrainingPair], num_samples: int = 3):
        """比較微調前後的相似度變化"""
        print("\n相似度比較 (微調後):")
        print("-" * 60)

        for pair in pairs[:num_samples]:
            pos_sim = self.evaluate_similarity(pair.anchor, pair.positive)
            neg_sim = self.evaluate_similarity(pair.anchor, pair.negative)

            print(f"問題: {pair.anchor[:40]}...")
            print(f"  正樣本相似度: {pos_sim:.4f}")
            print(f"  負樣本相似度: {neg_sim:.4f}")
            print(f"  差距: {pos_sim - neg_sim:.4f}")
            print()


if __name__ == "__main__":
    # 測試訓練器
    config = TrainerConfig(
        model_name="BAAI/bge-base-zh-v1.5",
        batch_size=8,
        num_epochs=1
    )

    trainer = ContrastiveTrainer(config)
    trainer.load_model()

    # 測試相似度計算
    sim = trainer.evaluate_similarity(
        "租屋押金可以收幾個月？",
        "房東最多可以收取多少押金？"
    )
    print(f"相似度: {sim:.4f}")
