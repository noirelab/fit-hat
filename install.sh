#!/usr/bin/env bash
# install.sh — Instala todas as dependências do Gemma-4-Flex no Linux Mint
set -euo pipefail

echo "================================================================="
echo " Gemma-4-Flex — Instalação de dependências (Linux Mint)"
echo "================================================================="

# ---------------------------------------------------------------------------
# 1. Pacotes do sistema
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Atualizando repositórios e instalando pacotes do sistema..."
sudo apt-get update -y
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget

# ---------------------------------------------------------------------------
# 2. Ambiente virtual Python
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Criando ambiente virtual Python (.venv)..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. Dependências principais (PyPI)
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Instalando dependências Python principais..."
pip install \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "psutil>=5.9.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "aiofiles>=23.0.0"

# ---------------------------------------------------------------------------
# 4. PyTorch (com suporte a CUDA 12.1; troque cu121 por cpu se não tiver GPU)
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Instalando PyTorch (CUDA 12.1)..."
echo "      Se não tiver GPU NVIDIA, substitua 'cu121' por 'cpu' no install.sh"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 5. Pacotes de otimização (flextensor / turboquant)
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Instalando pacotes de otimização..."

if pip install flextensor 2>/dev/null; then
    echo "  ✔  flextensor instalado via PyPI"
else
    echo "  ⚠  flextensor não encontrado no PyPI."
    echo "     Instale manualmente conforme a documentação do pacote."
fi

if pip install turboquant 2>/dev/null; then
    echo "  ✔  turboquant instalado via PyPI"
else
    echo "  ⚠  turboquant não encontrado no PyPI."
    echo "     Instale manualmente conforme a documentação do pacote."
fi

# ---------------------------------------------------------------------------
# Concluído
# ---------------------------------------------------------------------------
echo ""
echo "================================================================="
echo " Instalação concluída!"
echo " Para iniciar a aplicação execute:  bash run.sh"
echo "================================================================="
