"""
資料載入與處理模組
處理法律問答資料的載入、驗證和格式化
"""
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class QAPair:
    """法律問答對資料結構"""
    id: str
    question: str
    answer: str
    category: Optional[str] = None
    source: Optional[str] = None


class LegalQADataLoader:
    """法律問答資料載入器"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.qa_pairs: list[QAPair] = []

    def load_json(self, filename: str) -> list[QAPair]:
        """從 JSON 檔案載入問答資料"""
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"找不到資料檔案: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_list = data.get("qa_pairs", [])
        self.qa_pairs = [
            QAPair(
                id=item.get("id", str(i)),
                question=item["question"],
                answer=item["answer"],
                category=item.get("category"),
                source=item.get("source")
            )
            for i, item in enumerate(qa_list)
        ]

        print(f"已載入 {len(self.qa_pairs)} 筆問答資料")
        return self.qa_pairs

    def get_questions(self) -> list[str]:
        """取得所有問題"""
        return [qa.question for qa in self.qa_pairs]

    def get_answers(self) -> list[str]:
        """取得所有答案"""
        return [qa.answer for qa in self.qa_pairs]

    def get_qa_tuples(self) -> list[tuple[str, str]]:
        """取得問答配對 (用於對比學習)"""
        return [(qa.question, qa.answer) for qa in self.qa_pairs]

    def save_sample_template(self, filename: str = "sample_qa.json"):
        """儲存範例資料模板"""
        sample_data = {
            "qa_pairs": [
                {
                    "id": "001",
                    "question": "租屋押金最多可以收幾個月？",
                    "answer": "依據民法第422條及租賃住宅市場發展及管理條例第7條規定，住宅租賃的押金最高不得超過兩個月租金。若房東收取超過兩個月的押金，超過部分得抵付租金。",
                    "category": "租賃",
                    "source": "民法"
                },
                {
                    "id": "002",
                    "question": "房東可以隨時進入出租房屋嗎？",
                    "answer": "房東不可以隨意進入已出租的房屋。依據刑法第306條，無故侵入他人住宅者可處一年以下有期徒刑。房東若需進入房屋檢查或維修，應事先與房客約定時間。",
                    "category": "租賃",
                    "source": "刑法"
                },
                {
                    "id": "003",
                    "question": "車禍肇事者逃逸會有什麼法律責任？",
                    "answer": "依據刑法第185-4條，駕駛動力交通工具肇事，致人死傷而逃逸者，處一年以上七年以下有期徒刑。此外還需負擔民事損害賠償責任。",
                    "category": "交通",
                    "source": "刑法"
                }
            ]
        }

        filepath = self.data_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        print(f"已儲存範例模板至: {filepath}")
        return filepath


if __name__ == "__main__":
    # 測試資料載入器
    loader = LegalQADataLoader()
    loader.save_sample_template()

    # 載入範例資料
    qa_pairs = loader.load_json("sample_qa.json")

    print("\n範例問題:")
    for qa in qa_pairs[:3]:
        print(f"  - {qa.question}")
