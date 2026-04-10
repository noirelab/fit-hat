"""
servidor.py — Backend de Inferência Gemma-4-Flex
Requisitos: FastAPI, Uvicorn, PyTorch, psutil, transformers, flextensor, turboquant
"""

import time
import asyncio

import psutil
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Importações dos otimizadores
import flextensor
from flextensor import OffloadConfig
import turboquant

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-31b"
# Peso total estimado do modelo quantizado (GB) — usado para calcular offloading
MODEL_TOTAL_GB = 18.5
MAX_NEW_TOKENS = 512

# Referências globais ao modelo e tokenizer (populadas no startup)
modelo = None
tokenizer = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Gemma-4-Flex Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup — carrega o modelo com otimizações de memória
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def carregar_modelo():
    """
    1. Carrega google/gemma-4-31b em bfloat16 na CPU.
    2. Aplica FlexTensor para offloading dinâmico GPU ↔ RAM.
    3. Aplica TurboQuant para compressão do cache KV (key=3 bits, value=2 bits).
    """
    global modelo, tokenizer

    print("[startup] Carregando tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("[startup] Carregando modelo em bfloat16 na CPU…")
    modelo = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # --- FlexTensor: offloading dinâmico de tensores GPU ↔ RAM ---
    print("[startup] Configurando FlexTensor (offloading GPU↔RAM)…")
    offload_cfg = OffloadConfig(
        gpu_device=0,
        warmup_iters=2,
        module_patterns=["model.layers.*"],
    )
    modelo = flextensor.offload(modelo, offload_cfg)

    # --- TurboQuant: compressão do cache KV ---
    # key_bits=3 → chaves quantizadas em 3 bits (método Lloyd-Max)
    # value_bits=2 → valores quantizados em 2 bits
    print("[startup] Aplicando TurboQuant (KV cache compression: key=3b, value=2b)…")
    modelo = turboquant.compress_kv_cache(
        modelo,
        method="lloyd-max",
        key_bits=3,
        value_bits=2,
    )

    print("[startup] Modelo pronto para inferência.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PromptPayload(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Rota de Inferência — POST /gerar
# ---------------------------------------------------------------------------
@app.post("/gerar")
async def gerar(payload: PromptPayload):
    """
    Executa a inferência e retorna a resposta junto com a métrica TPS.

    TPS (Tokens Per Second) = tokens_gerados / tempo_decorrido
    Mede a taxa de geração, útil para monitorar degradação por offloading.
    """
    inputs = tokenizer(payload.prompt, return_tensors="pt")
    # Move os inputs para o mesmo dispositivo do modelo (pode ser GPU ou CPU via offload)
    inputs = {k: v.to(next(modelo.parameters()).device) for k, v in inputs.items()}

    inicio = time.time()  # Marca o início da geração

    with torch.no_grad():
        output_ids = modelo.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    tempo_decorrido = time.time() - inicio  # Tempo total de geração em segundos

    # Conta apenas os tokens gerados (exclui o prompt)
    tokens_gerados = output_ids.shape[-1] - inputs["input_ids"].shape[-1]

    # TPS: taxa de geração de tokens
    tps = tokens_gerados / tempo_decorrido if tempo_decorrido > 0 else 0.0

    resposta = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove o prompt da resposta para retornar apenas o texto gerado
    if resposta.startswith(payload.prompt):
        resposta = resposta[len(payload.prompt):].strip()

    return {
        "resposta": resposta,
        "tps": round(tps, 2),
        "tokens_gerados": int(tokens_gerados),
        "tempo_s": round(tempo_decorrido, 3),
    }


# ---------------------------------------------------------------------------
# Rota de Telemetria — GET /metricas
# ---------------------------------------------------------------------------
@app.get("/metricas")
async def metricas():
    """
    Retorna métricas de hardware em tempo real (leve e assíncrona).

    - vram_active  : memória GPU efetivamente alocada pelos tensores (GB).
    - vram_reserved: memória GPU reservada pelo PyTorch (inclui fragmentação) (GB).
    - system_ram   : RAM do sistema em uso (GB).
    - offloading_gb: estimativa de quanto do modelo está na RAM em vez da GPU.
                     Fórmula: max(0, MODEL_TOTAL_GB - vram_active)
                     Quando a VRAM não comporta o modelo inteiro, o excesso
                     fica na RAM via FlexTensor.
    """
    gb = 1024 ** 3  # Fator de conversão bytes → GB

    # VRAM efetivamente usada pelos tensores PyTorch
    vram_active = torch.cuda.memory_allocated(0) / gb

    # VRAM reservada (pode incluir cache não liberado pelo alocador)
    vram_reserved = torch.cuda.memory_reserved(0) / gb

    # RAM do sistema em uso
    system_ram = psutil.virtual_memory().used / gb

    # Offloading estimado: parte do modelo que não cabe na VRAM e vai para RAM
    offloading_gb = max(0.0, MODEL_TOTAL_GB - vram_active)

    return {
        "vram_active": round(vram_active, 2),
        "vram_reserved": round(vram_reserved, 2),
        "system_ram": round(system_ram, 2),
        "offloading_gb": round(offloading_gb, 2),
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("servidor:app", host="0.0.0.0", port=8000, reload=False)
