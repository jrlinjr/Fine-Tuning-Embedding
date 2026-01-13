"""
LLM 配對生成器
使用本地 Ollama 模型自動生成對比學習所需的正負樣本配對
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import ollama
from tqdm import tqdm

from .data_loader import QAPair


@dataclass
class TrainingPair:
    """訓練配對資料結構"""
    anchor: str       # 原始問題/錨點
    positive: str     # 正樣本（相似問題或對應答案）
    negative: str     # 負樣本（困難負樣本）
    qa_id: str        # 原始問答 ID


class PairGenerator:
    """使用 LLM 生成對比學習配對"""

    def __init__(self, model_name: str = "llama3:8b"):
        self.model_name = model_name
        self.pairs: list[TrainingPair] = []

    def _call_ollama(self, prompt: str) -> str:
        """呼叫 Ollama API"""
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def generate_similar_questions(self, qa: QAPair, num_variants: int = 3) -> list[str]:
        """生成相似問題變體（正樣本）"""
        prompt = f"""你是一個法律問答資料增強專家。請根據以下問題生成 {num_variants} 個意思相同但措辭不同的問題變體。

原始問題：{qa.question}
對應答案：{qa.answer}

要求：
1. 保持問題的核心意圖相同
2. 使用不同的措辭和句式
3. 可以使用口語化或正式的表達方式
4. 每個變體獨立一行，不要編號

請直接輸出問題變體，每行一個："""

        response = self._call_ollama(prompt)
        variants = [line.strip() for line in response.strip().split("\n") if line.strip()]
        # 過濾掉太短或包含編號的行
        variants = [v for v in variants if len(v) > 5 and not re.match(r"^\d+[\.\)、]", v)]
        return variants[:num_variants]

    def generate_hard_negatives(self, qa: QAPair, all_qas: list[QAPair], num_negatives: int = 2) -> list[str]:
        """生成困難負樣本（看似相關但答案不同的問題）"""
        # 取得其他類別的問答作為參考
        other_qas = [q for q in all_qas if q.id != qa.id][:5]
        other_answers_text = "\n".join([f"- {q.answer[:100]}..." for q in other_qas])

        prompt = f"""你是一個法律問答資料增強專家。請生成 {num_negatives} 個「困難負樣本」問題。

原始問題：{qa.question}
原始答案：{qa.answer}

其他參考答案：
{other_answers_text}

困難負樣本的要求：
1. 問題看起來與原始問題相關（例如相同法律領域）
2. 但實際上是在問不同的事情，對應不同的答案
3. 這種問題容易讓模型混淆

請直接輸出困難負樣本問題，每行一個："""

        response = self._call_ollama(prompt)
        negatives = [line.strip() for line in response.strip().split("\n") if line.strip()]
        negatives = [n for n in negatives if len(n) > 5 and not re.match(r"^\d+[\.\)、]", n)]
        return negatives[:num_negatives]

    def generate_training_pairs(
        self,
        qa_pairs: list[QAPair],
        num_positive_variants: int = 2,
        num_hard_negatives: int = 1
    ) -> list[TrainingPair]:
        """為所有問答生成訓練配對"""
        self.pairs = []

        for qa in tqdm(qa_pairs, desc="生成訓練配對"):
            # 生成正樣本：問題變體
            positive_questions = self.generate_similar_questions(qa, num_positive_variants)

            # 生成困難負樣本
            hard_negatives = self.generate_hard_negatives(qa, qa_pairs, num_hard_negatives)

            # 建立配對
            # 策略 1: anchor=問題, positive=答案
            if hard_negatives:
                self.pairs.append(TrainingPair(
                    anchor=qa.question,
                    positive=qa.answer,
                    negative=hard_negatives[0] if hard_negatives else "",
                    qa_id=qa.id
                ))

            # 策略 2: anchor=問題, positive=問題變體
            for pos_q in positive_questions:
                for neg in hard_negatives:
                    self.pairs.append(TrainingPair(
                        anchor=qa.question,
                        positive=pos_q,
                        negative=neg,
                        qa_id=qa.id
                    ))

        print(f"已生成 {len(self.pairs)} 組訓練配對")
        return self.pairs

    def save_pairs(self, output_path: str = "data/pairs/training_pairs.json"):
        """儲存訓練配對"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "pairs": [
                {
                    "anchor": p.anchor,
                    "positive": p.positive,
                    "negative": p.negative,
                    "qa_id": p.qa_id
                }
                for p in self.pairs
            ],
            "total_count": len(self.pairs)
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已儲存配對至: {output_file}")

    def load_pairs(self, input_path: str = "data/pairs/training_pairs.json") -> list[TrainingPair]:
        """載入訓練配對"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.pairs = [
            TrainingPair(
                anchor=p["anchor"],
                positive=p["positive"],
                negative=p["negative"],
                qa_id=p["qa_id"]
            )
            for p in data["pairs"]
        ]

        print(f"已載入 {len(self.pairs)} 組訓練配對")
        return self.pairs


if __name__ == "__main__":
    from .data_loader import LegalQADataLoader

    # 載入資料
    loader = LegalQADataLoader()
    loader.save_sample_template()
    qa_pairs = loader.load_json("sample_qa.json")

    # 生成配對
    generator = PairGenerator(model_name="llama3:8b")
    training_pairs = generator.generate_training_pairs(
        qa_pairs,
        num_positive_variants=2,
        num_hard_negatives=1
    )

    # 儲存配對
    generator.save_pairs()

    # 顯示範例
    print("\n範例配對:")
    for pair in training_pairs[:2]:
        print(f"  Anchor: {pair.anchor[:50]}...")
        print(f"  Positive: {pair.positive[:50]}...")
        print(f"  Negative: {pair.negative[:50]}...")
        print()
