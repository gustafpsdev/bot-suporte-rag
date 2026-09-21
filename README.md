# Bot de suporte de TI (RAG)

Assistente de suporte que responde dúvidas de usuários com base na
uma **base de conhecimento de exemplo** — não apenas no conhecimento genérico do modelo.
Isso é feito com a técnica **RAG** (Retrieval-Augmented Generation): antes de
perguntar ao modelo, o sistema busca os trechos relevantes da base e os entrega
junto com a pergunta. Resultado: respostas ancoradas nos seus procedimentos, sem
"alucinação".

## Como funciona

```
pergunta -> vira vetor -> busca na base -> trechos -> modelo de IA -> resposta
```

## Como rodar

1. Crie e ative um ambiente virtual:
   ```
   python -m venv .venv
   .venv\Scripts\activate        (Windows)
   source .venv/bin/activate      (Linux/Mac)
   ```
2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
3. Copie `.env.example` para `.env`. O bot já roda sem chave, em
   **modo demonstração** (mostra os trechos recuperados). Para respostas
   redigidas de verdade, coloque sua `ANTHROPIC_API_KEY` no `.env`.
4. Rode:
   ```
   python bot.py
   ```

## A base de conhecimento

É só a pasta `base_conhecimento/` com arquivos `.md` ou `.txt`. Para ensinar
coisas novas ao bot, basta adicionar arquivos ali — nenhum código muda.

## Upgrade: busca semântica de verdade

A versão atual usa TF-IDF (busca por palavras). Para busca **semântica**
(entende sinônimos e intenção, não só palavras iguais), troque o índice por
embeddings com `sentence-transformers`. É o passo que conecta este projeto a
Machine Learning de verdade. Próximo da lista de melhorias abaixo.

## Próximos passos

- [ ] Trocar TF-IDF por embeddings semânticos (sentence-transformers)
- [ ] Ler a base direto de uma lista/biblioteca do SharePoint
- [ ] Trocar a linha de comando por uma interface web
- [ ] Registrar as perguntas sem resposta para melhorar a base
