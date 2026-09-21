"""
Bot de suporte de TI com RAG (Retrieval-Augmented Generation)
=============================================================

Protótipo de linha de comando. Responde dúvidas de usuários com base
NA SUA documentação interna, e não no conhecimento genérico do modelo.

O fluxo segue exatamente o diagrama:
    pergunta -> embedding -> busca na base -> trechos -> modelo -> resposta

Como rodar:
    python bot.py

Autor: Gustavo Paiva
"""

import os
import sys
import glob
import logging

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# ETAPA 0 — Configuração (nada de segredo "chumbado" no código)
# ---------------------------------------------------------------------------
# As configurações vêm de um arquivo .env, que NÃO vai para o Git.
# Assim o mesmo código roda em qualquer máquina só trocando o .env.

load_dotenv()

PASTA_BASE       = os.getenv("PASTA_BASE", "base_conhecimento")
TAMANHO_CHUNK    = int(os.getenv("TAMANHO_CHUNK", "500"))   # caracteres por trecho
TRECHOS_BUSCA    = int(os.getenv("TRECHOS_BUSCA", "3"))     # quantos trechos recuperar
MODELO_LLM       = os.getenv("MODELO_LLM", "claude-sonnet-4-5")
API_KEY          = os.getenv("ANTHROPIC_API_KEY")           # opcional (modo demo se ausente)

# Logging: registra o que o bot faz. Em produção iria para um arquivo.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")


# ---------------------------------------------------------------------------
# ETAPA 1 (indexação) — Carregar documentos e quebrar em trechos (chunks)
# ---------------------------------------------------------------------------
# Modelos trabalham melhor com pedaços pequenos e focados do que com um
# documento inteiro de uma vez. Quebramos cada arquivo em trechos.

def carregar_documentos(pasta: str) -> list[dict]:
    caminhos = glob.glob(os.path.join(pasta, "*.md")) + \
               glob.glob(os.path.join(pasta, "*.txt"))
    if not caminhos:
        raise FileNotFoundError(
            f"Nenhum documento (.md/.txt) encontrado em '{pasta}'."
        )

    trechos = []
    for caminho in sorted(caminhos):
        nome = os.path.basename(caminho)
        with open(caminho, encoding="utf-8") as f:
            texto = f.read().strip()

        # Quebra simples por tamanho, tentando cortar em quebras de parágrafo.
        inicio = 0
        while inicio < len(texto):
            fim = inicio + TAMANHO_CHUNK
            pedaco = texto[inicio:fim].strip()
            if pedaco:
                trechos.append({"fonte": nome, "conteudo": pedaco})
            inicio = fim

    log.info("Carregados %d trechos de %d documentos.", len(trechos), len(caminhos))
    return trechos


# ---------------------------------------------------------------------------
# ETAPA 2 e 3 — Indexar (virar vetores) e buscar por similaridade
# ---------------------------------------------------------------------------
# Aqui é o "embedding" do diagrama. Nesta versão enxuta usamos TF-IDF, que
# transforma texto em vetores por frequência de palavras — leve e roda em
# qualquer máquina, sem baixar modelos. (Mais abaixo, no README, mostro como
# trocar por embeddings semânticos de verdade com uma linha.)

class IndiceBusca:
    def __init__(self, trechos: list[dict]):
        self.trechos = trechos
        self.vetorizador = TfidfVectorizer()
        corpus = [t["conteudo"] for t in trechos]
        self.matriz = self.vetorizador.fit_transform(corpus)
        log.info("Índice construído (%d dimensões).", self.matriz.shape[1])

    def buscar(self, pergunta: str, k: int) -> list[dict]:
        vetor_pergunta = self.vetorizador.transform([pergunta])
        similaridades = cosine_similarity(vetor_pergunta, self.matriz)[0]
        # pega os k trechos com maior similaridade
        melhores = similaridades.argsort()[::-1][:k]
        resultado = []
        for i in melhores:
            if similaridades[i] > 0:  # ignora trechos sem nenhuma relação
                item = dict(self.trechos[i])
                item["score"] = float(similaridades[i])
                resultado.append(item)
        return resultado


# ---------------------------------------------------------------------------
# ETAPA 4 e 5 — Montar o prompt com os trechos e enviar ao modelo
# ---------------------------------------------------------------------------
# O truque do RAG está aqui: entregamos ao modelo a pergunta JUNTO com os
# trechos recuperados, e instruímos que responda apenas com base neles.

def montar_prompt(pergunta: str, trechos: list[dict]) -> str:
    contexto = "\n\n".join(
        f"[Fonte: {t['fonte']}]\n{t['conteudo']}" for t in trechos
    )
    return (
        "Você é um assistente de suporte de TI. Responda à pergunta do usuário "
        "usando APENAS as informações dos trechos abaixo. Se a resposta não "
        "estiver nos trechos, diga que não encontrou a informação na base e "
        "sugira abrir um chamado. Cite a fonte usada.\n\n"
        f"=== TRECHOS DA BASE DE CONHECIMENTO ===\n{contexto}\n\n"
        f"=== PERGUNTA DO USUÁRIO ===\n{pergunta}"
    )


def responder(pergunta: str, indice: IndiceBusca) -> str:
    trechos = indice.buscar(pergunta, TRECHOS_BUSCA)

    if not trechos:
        return ("Não encontrei nada relacionado na base de conhecimento. "
                "Sugiro abrir um chamado para o suporte.")

    prompt = montar_prompt(pergunta, trechos)

    # Sem chave de API: MODO DEMONSTRAÇÃO — mostra o que seria enviado ao modelo.
    # Isso deixa ver a "inteligência" do RAG (a recuperação) funcionando.
    if not API_KEY:
        fontes = ", ".join(sorted({t["fonte"] for t in trechos}))
        return (
            "  [MODO DEMONSTRAÇÃO — sem chave de API]\n"
            f"  Trechos recuperados de: {fontes}\n"
            f"  (Melhor similaridade: {trechos[0]['score']:.2f})\n\n"
            "  Com uma chave configurada, o modelo receberia esses trechos e\n"
            "  redigiria a resposta final. Prompt montado:\n"
            "  " + "-" * 60 + "\n"
            + "\n".join("  " + linha for linha in prompt.splitlines())
        )

    # Com chave: chama o modelo de verdade.
    try:
        import anthropic
        cliente = anthropic.Anthropic(api_key=API_KEY)
        resposta = cliente.messages.create(
            model=MODELO_LLM,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resposta.content[0].text
    except Exception as e:
        log.error("Falha ao chamar o modelo: %s", e)
        return "Ocorreu um erro ao consultar o modelo. Tente novamente."


# ---------------------------------------------------------------------------
# ETAPA 6 — Loop de conversa (a interface, por enquanto no terminal)
# ---------------------------------------------------------------------------

def main():
    print("\n=== Bot de suporte de TI (protótipo RAG) ===")
    print("Digite sua dúvida, ou 'sair' para encerrar.\n")

    try:
        trechos = carregar_documentos(PASTA_BASE)
        indice = IndiceBusca(trechos)
    except Exception as e:
        log.error("Não foi possível iniciar: %s", e)
        sys.exit(1)

    if not API_KEY:
        print(">> Rodando em MODO DEMONSTRAÇÃO (sem ANTHROPIC_API_KEY).\n")

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo!")
            break

        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Até logo!")
            break
        if not pergunta:
            continue

        print("\nBot:", responder(pergunta, indice), "\n")


if __name__ == "__main__":
    main()
