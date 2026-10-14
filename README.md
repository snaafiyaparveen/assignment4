# AI customer support assistant

A Streamlit chat application that uses LangChain to answer e-commerce support
questions through a local Ollama model, OpenRouter, or Google Gemini. Conversation
history is kept in Streamlit's session state so the assistant can answer follow-up
questions. The assistant does not have access to orders or store policies and
will not claim to have processed a return, refund, or replacement.

## Requirements

- Python 3.10 or newer
- For Ollama: [Ollama](https://ollama.com/) installed and running, with a model
  downloaded locally
- For OpenRouter: an API key from [OpenRouter](https://openrouter.ai/)
- For Gemini: an API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Setup on Windows

Run these commands in the project folder using PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` to choose your provider:

- **Ollama:** Keep `LLM_PROVIDER=ollama`. Set `OLLAMA_MODEL` to an installed
  model. For example, run `ollama pull qwen2.5:3b` in a separate terminal.
- **OpenRouter:** Set `LLM_PROVIDER=openrouter`, add your key to
  `OPENROUTER_API_KEY`, and set `OPENROUTER_MODEL` to a model available to your
  account.
- **Gemini:** Set `LLM_PROVIDER=gemini`, add your Google AI Studio key to
  `GOOGLE_API_KEY`, and set `GEMINI_MODEL` to a model enabled for your key.
  The default is `gemini-3.5-flash`.

Do not share or commit `.env`; it is excluded by `.gitignore`.

## Run

```powershell
streamlit run app.py
```

## Try these questions

- “My order is late. What should I do?”
- “The item arrived damaged. How do I request a replacement?”
- “Can I return an item I bought last week?”
- “When will my refund arrive?”
- Follow up with “What information should I include?”

The assistant provides general guidance, not verified account-specific answers.
Evaluate its responses for relevance, accuracy, and consistency against your
store's actual policies before using it with customers.

## Gemini quota limits

If the app reports that Gemini's request quota is exhausted, the application is
running but Google is temporarily refusing requests for that API key. Check
[Google AI Studio usage and rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
wait for the quota to reset, or enable an appropriate billing plan. Alternatively,
configure `LLM_PROVIDER` and the matching credentials for OpenRouter or a running
local Ollama server in `.env`, then restart Streamlit.
