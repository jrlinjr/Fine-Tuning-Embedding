#!/bin/bash
# 遠端 Mac M4 Pro 快速設置腳本

set -e

echo "=== 法律問答 RAG 嵌入模型微調 - 環境設置 ==="

# 1. 檢查 Python
echo ""
echo "[1/4] 檢查 Python..."
if command -v python3.11 &> /dev/null; then
    PYTHON=python3.11
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "錯誤: 找不到 Python 3，請先安裝"
    exit 1
fi
echo "使用 Python: $($PYTHON --version)"

# 2. 建立虛擬環境
echo ""
echo "[2/4] 建立虛擬環境..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "已建立 venv/"
else
    echo "venv/ 已存在，跳過"
fi

# 3. 安裝依賴
echo ""
echo "[3/4] 安裝依賴套件..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "依賴安裝完成"

# 4. 檢查 Ollama
echo ""
echo "[4/4] 檢查 Ollama..."
if command -v ollama &> /dev/null; then
    echo "Ollama 已安裝"
    if ollama list 2>/dev/null | grep -q "llama3"; then
        echo "llama3 模型已存在"
    else
        echo "正在下載 llama3:8b 模型..."
        ollama pull llama3:8b
    fi
else
    echo "警告: Ollama 未安裝"
    echo "請從 https://ollama.ai/ 下載安裝"
    echo "安裝後執行: ollama pull llama3:8b"
fi

# 5. 檢查 MPS
echo ""
echo "=== 環境檢查 ==="
source venv/bin/activate
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS 可用: {torch.backends.mps.is_available()}')
"

echo ""
echo "=== 設置完成 ==="
echo ""
echo "使用方式:"
echo "  source venv/bin/activate"
echo ""
echo "  # 1. 放入法律資料到 data/raw/laws/"
echo "  # 2. 查詢法規名稱"
echo "  python scripts/list_laws.py 民法"
echo ""
echo "  # 3. 生成問答對"
echo "  python scripts/generate_qa_from_laws.py --filter 民法 中華民國刑法"
echo ""
echo "  # 4. 生成配對"
echo "  python scripts/generate_pairs.py"
echo ""
echo "  # 5. 訓練模型"
echo "  python scripts/train_embedding.py"
