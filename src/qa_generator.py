"""
問答生成器
使用 LLM 從原始法律文本自動生成問答對
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import ollama
from tqdm import tqdm


@dataclass
class LegalDocument:
    """法律文件資料結構"""
    id: str
    title: str
    content: str
    category: Optional[str] = None
    source: Optional[str] = None


@dataclass
class GeneratedQA:
    """生成的問答對"""
    id: str
    question: str
    answer: str
    source_doc_id: str
    category: Optional[str] = None


class QAGenerator:
    """從法律文本自動生成問答對"""

    def __init__(self, model_name: str = "", ollama_host: Optional[str] = None):
        """
        Args:
            model_name: Ollama 模型名稱
            ollama_host: 遠端 Ollama 主機 (如: "http://192.168.1.100:11434")
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.documents: list[LegalDocument] = []
        self.qa_pairs: list[GeneratedQA] = []

        # 設定 Ollama client
        if ollama_host:
            self.client = ollama.Client(host=ollama_host)
            print(f"連接遠端 Ollama: {ollama_host}")
        else:
            self.client = ollama.Client()

    def _call_ollama(self, prompt: str) -> str:
        """呼叫 Ollama API"""
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def load_documents(
        self,
        data_dir: str,
        pattern: str = "*.json",
        max_laws: Optional[int] = None,
        max_articles_per_law: Optional[int] = None,
        law_filter: Optional[list[str]] = None,
        exact_match: bool = True
    ) -> list[LegalDocument]:
        """
        載入所有法律 JSON 檔案

        Args:
            data_dir: 資料目錄
            pattern: 檔案匹配模式
            max_laws: 最多載入幾部法規 (None = 全部)
            max_articles_per_law: 每部法規最多載入幾條 (None = 全部)
            law_filter: 只載入這些法規名稱
            exact_match: True=精確匹配法規名稱, False=包含匹配
        """
        data_path = Path(data_dir)
        self.documents = []

        for json_file in data_path.glob(pattern):
            # 支援 BOM 編碼
            try:
                with open(json_file, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # 支援法務部法規資料庫格式 (Laws -> LawArticles)
            if isinstance(data, dict) and "Laws" in data:
                self._load_moj_format(data, max_laws, max_articles_per_law, law_filter, exact_match)
            # 支援多種其他 JSON 格式
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    doc = self._parse_document(item, f"{json_file.stem}_{i}")
                    if doc:
                        self.documents.append(doc)
            elif isinstance(data, dict):
                items = (
                    data.get("articles") or
                    data.get("laws") or
                    data.get("items") or
                    data.get("data") or
                    [data]
                )
                for i, item in enumerate(items):
                    doc = self._parse_document(item, f"{json_file.stem}_{i}")
                    if doc:
                        self.documents.append(doc)

        print(f"已載入 {len(self.documents)} 筆法律條文")
        return self.documents

    def _load_moj_format(
        self,
        data: dict,
        max_laws: Optional[int],
        max_articles_per_law: Optional[int],
        law_filter: Optional[list[str]],
        exact_match: bool = True
    ):
        """載入法務部法規資料庫格式"""
        laws = data.get("Laws", [])

        law_count = 0
        for law in laws:
            law_name = law.get("LawName", "")
            law_category = law.get("LawCategory", "")

            # 過濾法規
            if law_filter:
                if exact_match:
                    # 精確匹配：法規名稱必須完全等於 filter 中的某一項
                    if law_name not in law_filter:
                        continue
                else:
                    # 包含匹配：法規名稱包含 filter 中的某個關鍵字
                    if not any(kw in law_name for kw in law_filter):
                        continue

            # 限制法規數量
            if max_laws and law_count >= max_laws:
                break

            articles = law.get("LawArticles", [])
            article_count = 0

            for article in articles:
                # 只處理實際條文 (ArticleType = 'A')
                if article.get("ArticleType") != "A":
                    continue

                # 限制每部法規的條文數量
                if max_articles_per_law and article_count >= max_articles_per_law:
                    break

                article_no = article.get("ArticleNo", "").strip()
                article_content = article.get("ArticleContent", "").strip()

                if article_content:
                    doc_id = f"{law_name}_{article_no}".replace(" ", "")
                    self.documents.append(LegalDocument(
                        id=doc_id,
                        title=f"{law_name} {article_no}",
                        content=article_content,
                        category=law_category,
                        source=law_name
                    ))
                    article_count += 1

            if article_count > 0:
                law_count += 1

    def _parse_document(self, item: dict, default_id: str) -> Optional[LegalDocument]:
        """解析單一文件"""
        # 嘗試找出標題和內容
        title = (
            item.get("title") or
            item.get("name") or
            item.get("法條名稱") or
            item.get("條號") or
            ""
        )
        content = (
            item.get("content") or
            item.get("text") or
            item.get("內容") or
            item.get("條文") or
            item.get("description") or
            ""
        )

        if not content:
            return None

        return LegalDocument(
            id=item.get("id", default_id),
            title=str(title),
            content=str(content),
            category=item.get("category") or item.get("類別"),
            source=item.get("source") or item.get("來源") or item.get("法規名稱")
        )

    def generate_qa_from_document(self, doc: LegalDocument, num_qa: int = 2) -> list[GeneratedQA]:
        """從單一文件生成問答對"""
        prompt = f"""你是一個法律問答專家。請根據以下法律條文內容，生成 {num_qa} 組自然的問答對。

法規來源：{doc.source or '未知'}
條文標題：{doc.title}
條文內容：{doc.content}

要求：
1. 問題要像一般民眾會問的問題，使用口語化的方式
2. 答案要準確引用條文內容，但用易懂的方式解釋
3. 問題要涵蓋條文的重要概念
4. 使用繁體中文

請用以下 JSON 格式輸出，只輸出 JSON，不要其他文字：
[
  {{"question": "問題1", "answer": "答案1"}},
  {{"question": "問題2", "answer": "答案2"}}
]
"""
        response = self._call_ollama(prompt)

        # 解析 JSON 回應
        qa_list = []
        try:
            # 嘗試找出 JSON 部分
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                parsed = json.loads(json_match.group())
                for i, item in enumerate(parsed):
                    if "question" in item and "answer" in item:
                        qa_list.append(GeneratedQA(
                            id=f"{doc.id}_qa_{i}",
                            question=item["question"],
                            answer=item["answer"],
                            source_doc_id=doc.id,
                            category=doc.category
                        ))
        except json.JSONDecodeError:
            print(f"  警告: 無法解析文件 {doc.id} 的回應")

        return qa_list

    def generate_all_qa(self, num_qa_per_doc: int = 2) -> list[GeneratedQA]:
        """為所有文件生成問答對"""
        self.qa_pairs = []

        for doc in tqdm(self.documents, desc="生成問答對"):
            qa_list = self.generate_qa_from_document(doc, num_qa_per_doc)
            self.qa_pairs.extend(qa_list)

        print(f"共生成 {len(self.qa_pairs)} 組問答對")
        return self.qa_pairs

    def save_qa_pairs(self, output_path: str = "data/raw/generated_qa.json"):
        """儲存生成的問答對"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "qa_pairs": [
                {
                    "id": qa.id,
                    "question": qa.question,
                    "answer": qa.answer,
                    "category": qa.category,
                    "source": qa.source_doc_id
                }
                for qa in self.qa_pairs
            ],
            "total_count": len(self.qa_pairs)
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已儲存至: {output_file}")

    def save_sample_law_template(self, output_path: str = "data/raw/laws"):
        """建立範例法律資料模板"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 範例：民法租賃
        rental_law = {
            "law_name": "民法",
            "category": "租賃",
            "articles": [
                {
                    "id": "civil_422",
                    "title": "第422條",
                    "content": "不動產之租賃契約，其期限逾一年者，應以字據訂立之，未以字據訂立者，視為不定期限之租賃。",
                    "source": "民法"
                },
                {
                    "id": "civil_440",
                    "title": "第440條",
                    "content": "承租人租金支付有遲延者，出租人得定相當期限，催告承租人支付租金，如承租人於其期限內不為支付，出租人得終止契約。",
                    "source": "民法"
                }
            ]
        }

        # 範例：刑法
        criminal_law = {
            "law_name": "刑法",
            "category": "交通",
            "articles": [
                {
                    "id": "criminal_185_4",
                    "title": "第185-4條",
                    "content": "駕駛動力交通工具肇事，致人死傷而逃逸者，處一年以上七年以下有期徒刑。",
                    "source": "刑法"
                },
                {
                    "id": "criminal_306",
                    "title": "第306條",
                    "content": "無故侵入他人住宅、建築物或附連圍繞之土地或船艦者，處一年以下有期徒刑、拘役或九千元以下罰金。",
                    "source": "刑法"
                }
            ]
        }

        # 儲存範例
        with open(output_dir / "civil_law.json", "w", encoding="utf-8") as f:
            json.dump(rental_law, f, ensure_ascii=False, indent=2)

        with open(output_dir / "criminal_law.json", "w", encoding="utf-8") as f:
            json.dump(criminal_law, f, ensure_ascii=False, indent=2)

        print(f"已建立範例法律資料模板於: {output_dir}")
        print("  - civil_law.json")
        print("  - criminal_law.json")


if __name__ == "__main__":
    generator = QAGenerator(model_name="llama3:8b")

    # 建立範例模板
    generator.save_sample_law_template()

    # 載入文件
    docs = generator.load_documents("data/raw/laws")

    # 生成問答
    qa_pairs = generator.generate_all_qa(num_qa_per_doc=2)

    # 儲存
    generator.save_qa_pairs()

    # 顯示範例
    print("\n=== 生成的問答範例 ===")
    for qa in qa_pairs[:3]:
        print(f"\nQ: {qa.question}")
        print(f"A: {qa.answer[:80]}...")
