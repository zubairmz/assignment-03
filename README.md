# Assignment 3 — Notebook to PDF Agent

An AI-powered agent that converts any Jupyter Notebook (`.ipynb`) file into a professionally formatted PDF report. The agent uses LangChain tools orchestrated by a local LLM to parse the notebook structure and generate a polished, ready-to-submit PDF.

---

## What It Does

- **Parses** the `.ipynb` JSON to extract markdown cells, code cells, and cell outputs
- **Converts** markdown to formatted text — headings (`H1/H2/H3`), bold, italic, lists, blockquotes, code fences
- **Styles** code cells in monospace gray boxes with `In [N]:` labels
- **Includes** cell outputs — printed text, images, and error tracebacks in styled blocks
- **Generates** a single professional PDF with:
  - Cover page (title, date, kernel, cell counts)
  - Auto-generated Table of Contents from headings
  - Header (notebook title) and footer (page number) on every page

---

## Project Structure

```
pdf_agent.py          # Main agent script
demo_pdf_output.pdf   # Sample output PDF (generated from the included notebook)
README.md   # This file
pdf_agent.db          # SQLite memory store (auto-created on first run)
```

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally with the `minimax-m2.5:cloud` model

### Install dependencies

```bash
uv sync
```

This installs all required packages including `reportlab`, `langchain`, `langchain-ollama`, and `langgraph`.

### Verify Ollama is running

```bash
ollama list        # confirm minimax-m2.5:cloud is available
ollama serve       # start Ollama if not already running
```

---

## Usage

### CLI mode — pass the notebook path directly

```bash
uv run pdf_agent.py notebooks/01_Your_First_AI_Agent.ipynb
```

The output PDF is saved as `demo_pdf_output.pdf` in the current directory.

### Interactive mode

```bash
uv run pdf_agent.py
```

Then type at the prompt:

```
You: Convert the notebook at notebooks/exercises_solved.ipynb to exercises_solved.pdf
```

Type `quit` or `exit` to stop.

---

## How the Agent Works

The agent follows a two-step tool workflow:

1. **`parse_notebook`** — Reads the `.ipynb` file and returns a summary: cell counts, kernel info, and a preview of headings.
2. **`generate_pdf`** — Parses the full notebook and builds the PDF with cover page, TOC, styled cells, and outputs.

Before each tool call, the agent pauses and asks for your approval (human-in-the-loop):

```
  [Tool Call]  parse_notebook  |  args: {'notebook_path': 'notebooks/...'}
  Allow? (yes/no): yes

  [Tool Call]  generate_pdf  |  args: {'notebook_path': '...', 'output_path': 'pdf_agent.pdf'}
  Allow? (yes/no): yes

Agent: PDF generated successfully: pdf_agent.pdf
```

---

## Dependencies

| Package            | Purpose                                 |
| ------------------ | --------------------------------------- |
| `langchain`        | Agent framework and tool decorators     |
| `langchain-ollama` | Local LLM via Ollama                    |
| `langgraph`        | Agent graph execution and SQLite memory |
| `reportlab`        | PDF generation                          |

---

## Sample Output

`demo_pdf_output.pdf` — generated from `notebooks/01_Your_First_AI_Agent.ipynb` (17 pages).

`exercies_solved.pdf` — generated from `notebooks/exercies_solved.ipynb`
