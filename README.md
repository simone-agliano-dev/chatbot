# chatbot

Projects developed for the Udacity **Generative AI Nanodegree**.

The main project is a **custom Q&A chatbot about 2023 fashion trends**, built with
Retrieval-Augmented Generation (RAG) on top of OpenAI models.

| Notebook | What it is |
|---|---|
| [`custom_chatbot.ipynb`](custom_chatbot.ipynb) | Fashion-trends chatbot (RAG with OpenAI embeddings + completions) |
| [`week1_setup_and_first_tool.ipynb`](week1_setup_and_first_tool.ipynb) | Agentic customer data assistant, week 1: CRM data in DuckDB and a first read-only SQL tool |

---

## The fashion-trends chatbot

### Why

A general-purpose language model doesn't know much about recent, niche topics. Asked
*"What were the major fashion trends in 2023?"*, the plain model answers that it can't
predict the future. The custom chatbot answers with what the 2023 reports actually say.

### Dataset

`data/2023_fashion_trends.csv` holds curated reports and expert quotes about fashion trends
observed in 2023. Each row has three columns:

- `URL`: the source article
- `Trends`: the text snippet describing a trend or designer commentary
- `Source`: the article title

The two text columns are merged into a single `text` column, which is what the chatbot
searches over.

### How it works

```
question ──► embedding ──► cosine distance vs. every row ──► most relevant rows
                                                                    │
                    answer ◄── gpt-3.5-turbo-instruct ◄── prompt = context + question
```

1. **Embed the dataset.** Every row of `text` is turned into a vector with
   `text-embedding-ada-002`, sent in batches of 200.
2. **Retrieve.** `get_rows_sorted_by_relevance()` embeds the user question and sorts
   the rows by cosine distance from it.
3. **Build the prompt.** `build_prompt()` adds the most relevant rows as context until a
   token budget is reached (counted with `tiktoken`, `cl100k_base`). The model is told to
   answer from that context only, and to reply *"I don't know"* if the context isn't enough.
4. **Answer.** The prompt is sent to `gpt-3.5-turbo-instruct`.

### Results

The notebook compares the plain model against the custom chatbot on two questions:

| Question | Plain model | Custom chatbot |
|---|---|---|
| What were the major fashion trends in 2023? | "It is impossible to accurately predict the fashion trends for a specific year…" | Sheer clothing, red hues, denim, cobalt blue, elevated basics, daytime shine, maxi skirts, cargo pants, indie sleaze, minimalism, tailored looks, mesh details |
| What did fashion experts say about fashion in 2023? | "There is no way to accurately predict what fashion experts will say…" | Continued maximalism, surrealism and 3D designs, a return to simple everyday dressing, green and cobalt blue, a revival of nostalgic and grunge looks |

---

## How to run it

### Requirements

- Python 3.9+
- An OpenAI API key. The notebook points to the Udacity/Vocareum proxy
  (`https://openai.vocareum.com/v1`); to use OpenAI directly, remove the `api_base` line.
- The **legacy OpenAI SDK (0.28)**. The notebook uses `openai.Embedding`,
  `openai.Completion` and `openai.embeddings_utils`, which were removed in `openai>=1.0`.

```bash
pip install "openai==0.28" pandas tiktoken python-dateutil scipy scikit-learn plotly matplotlib
```

(`scipy`, `scikit-learn`, `plotly` and `matplotlib` are needed by `openai.embeddings_utils`.)

### Steps

1. Put the dataset at `data/2023_fashion_trends.csv`, next to the notebook.
   The CSV is not included in this repository.
2. Open `custom_chatbot.ipynb` and set your key:
   ```python
   openai.api_key = "YOUR-API-KEY"
   ```
   Don't commit a real key. Reading it from an environment variable is safer:
   ```python
   import os
   openai.api_key = os.environ["OPENAI_API_KEY"]
   ```
3. Run all cells. To ask your own question:
   ```python
   question = "Which colors were trending in 2023?"
   prompt = build_prompt(question, df, tokenizer, token_limit=2000)
   answer = openai.Completion.create(
       model="gpt-3.5-turbo-instruct",
       prompt=prompt,
       max_tokens=150,
   )["choices"][0]["text"].strip()
   print(answer)
   ```

### Limitations

- The chatbot only knows what's in the dataset. Questions outside 2023 fashion trends
  should get *"I don't know"*.
- Embeddings are recomputed on every run and are not cached to disk.
- It uses the legacy OpenAI SDK and completion models. Porting it to `openai>=1.0` and
  chat models would require small code changes.
