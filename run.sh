#!/usr/bin/env bash
# run.sh — Inicia o servidor Gemma-4-Flex e exibe o link de acesso
set -euo pipefail

HOST="0.0.0.0"
PORT="8000"

# ---------------------------------------------------------------------------
# Ativa o ambiente virtual criado pelo install.sh (se existir)
# ---------------------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ---------------------------------------------------------------------------
# Verifica se o uvicorn está disponível
# ---------------------------------------------------------------------------
if ! command -v uvicorn &>/dev/null; then
    echo "Erro: uvicorn não encontrado. Execute primeiro: bash install.sh"
    exit 1
fi

echo "================================================================="
echo " Gemma-4-Flex — Servidor de Inferência"
echo "================================================================="
echo ""
echo "  Acesse a interface em:"
echo ""
echo "    http://localhost:${PORT}"
echo ""
echo "  Endpoints da API:"
echo "    POST http://localhost:${PORT}/gerar"
echo "    GET  http://localhost:${PORT}/metricas"
echo ""
echo "  Pressione Ctrl+C para encerrar."
echo "================================================================="
echo ""

# ---------------------------------------------------------------------------
# Inicia o servidor
# ---------------------------------------------------------------------------
uvicorn servidor:app --host "${HOST}" --port "${PORT}" --reload false
