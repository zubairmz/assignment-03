"""
Notebook to PDF Agent  (pdf_agent.py)

An AI-powered agent that converts Jupyter Notebook (.ipynb) files into
professionally formatted PDF reports.

Features:
- Parses .ipynb JSON structure (markdown, code, and output cells)
- Converts markdown to formatted text with proper heading hierarchy
- Applies code-block styling to Python code cells
- Includes cell outputs (text, images, errors) in styled blocks
- Generates a professional PDF with header, page numbers, and table of contents

Usage:
    uv run pdf_agent.py notebooks/my_notebook.ipynb
    uv run pdf_agent.py                 # interactive mode

Type 'quit', 'exit', or 'q' to exit.
"""

import base64
import json
import re
import sqlite3
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

# LangChain Agent & Middleware
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware, wrap_tool_call
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus.tableofcontents import TableOfContents

# ─── PAGE CONSTANTS ───────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 20 * mm
T_MARGIN = B_MARGIN = 25 * mm
USABLE_W = PAGE_W - L_MARGIN - R_MARGIN

# Strip ANSI escape codes from terminal output
_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


# ─── TEXT / MARKDOWN HELPERS ──────────────────────────────────────────────────


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from terminal output."""
    return _ANSI.sub("", text)


def xml_escape(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraph content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_md(text: str) -> str:
    """Convert inline markdown syntax to ReportLab XML markup.

    Code spans are processed first and protected from italic/bold regex so that
    underscores inside backticks are never misread as italic markers.
    """
    text = xml_escape(text)

    # 1. Protect inline code spans before any other substitution
    _code_spans: list[str] = []

    def _protect(m: re.Match) -> str:
        tag = f'<font face="Courier" size="9">{m.group(1)}</font>'
        _code_spans.append(tag)
        return f"\x00CODE{len(_code_spans) - 1}\x00"

    text = re.sub(r"`(.+?)`", _protect, text)

    # 2. Bold-italic, bold, italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Only match _word_ when not adjacent to other word chars (avoids snake_case)
    text = re.sub(r"(?<![a-zA-Z0-9_])_([^_\n]+?)_(?![a-zA-Z0-9_])", r"<i>\1</i>", text)

    # 3. Links: [text](url) → underlined text
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"<u>\1</u>", text)

    # 4. Restore protected code spans
    for idx, span in enumerate(_code_spans):
        text = text.replace(f"\x00CODE{idx}\x00", span)

    return text


# ─── PARAGRAPH STYLES ─────────────────────────────────────────────────────────


def build_styles() -> dict:
    """Build and return all named ParagraphStyle objects for the PDF report."""
    return {
        "doc_title": ParagraphStyle(
            "doc_title",
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=colors.HexColor("#0d1117"),
            spaceAfter=6 * mm,
            alignment=1,
        ),
        "doc_subtitle": ParagraphStyle(
            "doc_subtitle",
            fontName="Helvetica",
            fontSize=12,
            textColor=colors.HexColor("#586069"),
            spaceAfter=4 * mm,
            alignment=1,
        ),
        "toc_title": ParagraphStyle(
            "toc_title",
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=colors.HexColor("#0d1117"),
            spaceAfter=5 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=colors.HexColor("#0d1117"),
            spaceAfter=4 * mm,
            spaceBefore=8 * mm,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=colors.HexColor("#24292e"),
            spaceAfter=3 * mm,
            spaceBefore=6 * mm,
        ),
        "h3": ParagraphStyle(
            "h3",
            fontName="Helvetica-BoldOblique",
            fontSize=12,
            textColor=colors.HexColor("#24292e"),
            spaceAfter=2 * mm,
            spaceBefore=4 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.HexColor("#24292e"),
            leading=15,
            spaceAfter=3 * mm,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.HexColor("#24292e"),
            leading=14,
            leftIndent=12,
            spaceAfter=1 * mm,
        ),
        "code_label": ParagraphStyle(
            "code_label",
            fontName="Courier-Bold",
            fontSize=8,
            textColor=colors.HexColor("#586069"),
            spaceAfter=1 * mm,
            spaceBefore=4 * mm,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Courier",
            fontSize=9,
            textColor=colors.HexColor("#24292e"),
            leading=13,
        ),
        "output_label": ParagraphStyle(
            "output_label",
            fontName="Courier-Bold",
            fontSize=8,
            textColor=colors.HexColor("#586069"),
            spaceAfter=1 * mm,
            spaceBefore=2 * mm,
        ),
        "output": ParagraphStyle(
            "output",
            fontName="Courier",
            fontSize=8.5,
            textColor=colors.HexColor("#333333"),
            leading=12,
        ),
        "error": ParagraphStyle(
            "error",
            fontName="Courier",
            fontSize=8.5,
            textColor=colors.HexColor("#cb2431"),
            leading=12,
        ),
        # TOC level styles
        "toc0": ParagraphStyle(
            "toc0",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            leftIndent=0,
        ),
        "toc1": ParagraphStyle(
            "toc1",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=20,
        ),
        "toc2": ParagraphStyle(
            "toc2",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            leftIndent=40,
            textColor=colors.HexColor("#586069"),
        ),
    }


# ─── FLOWABLE BUILDERS ────────────────────────────────────────────────────────


def _boxed_table(content, bg: str, border: str) -> Table:
    """Wrap a flowable in a full-width Table cell with background + border."""
    tbl = Table([[content]], colWidths=[USABLE_W])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(border)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return tbl


def make_code_cell(source: str, counter: int, styles: dict) -> list:
    """Return flowables for a code input cell."""
    items = [Paragraph(f"In&nbsp;[{counter}]:", styles["code_label"])]
    if source.strip():
        pre = Preformatted(source, styles["code"])
        items.append(_boxed_table(pre, "#f6f8fa", "#d0d7de"))
    items.append(Spacer(1, 2 * mm))
    return items


def make_output_cell(outputs: list, counter: int, styles: dict) -> list:
    """Return flowables for cell outputs (stream, text, image, error)."""
    if not outputs:
        return []

    items = [Paragraph(f"Out&nbsp;[{counter}]:", styles["output_label"])]

    for out in outputs:
        out_type = out.get("type", "")

        if out_type in ("stream", "text"):
            text = strip_ansi(out.get("text", ""))
            if text.strip():
                pre = Preformatted(text, styles["output"])
                items.append(_boxed_table(pre, "#f0f4f8", "#c8d8e8"))

        elif out_type == "image":
            try:
                img_data = base64.b64decode(out["data"])
                img = RLImage(BytesIO(img_data))
                max_w = USABLE_W * 0.85
                if img.drawWidth > max_w:
                    ratio = max_w / img.drawWidth
                    img.drawWidth = max_w
                    img.drawHeight *= ratio
                items.append(img)
            except Exception:
                pass  # Skip unreadable images

        elif out_type == "error":
            ename = out.get("ename", "Error")
            evalue = strip_ansi(out.get("evalue", ""))
            tb = "\n".join(strip_ansi(ln) for ln in out.get("traceback", []))
            error_text = f"{ename}: {evalue}\n{tb}" if tb else f"{ename}: {evalue}"
            pre = Preformatted(error_text, styles["error"])
            items.append(_boxed_table(pre, "#fdf3f3", "#f5c6c6"))

        items.append(Spacer(1, 1.5 * mm))

    return items


def _safe_para(markup: str, style) -> Paragraph:
    """Create a Paragraph; fall back to plain escaped text if markup is invalid."""
    try:
        return Paragraph(markup, style)
    except Exception:
        # Strip all tags and render as plain text
        plain = re.sub(r"<[^>]+>", "", markup)
        return Paragraph(xml_escape(plain), style)


def markdown_to_flowables(text: str, styles: dict) -> list:
    """Convert a markdown string to a list of ReportLab flowables."""
    items = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        # Blank line → small vertical space
        if not stripped.strip():
            if items and not isinstance(items[-1], Spacer):
                items.append(Spacer(1, 2 * mm))
            i += 1
            continue

        # Fenced code block
        if stripped.lstrip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if code_lines:
                pre = Preformatted("\n".join(code_lines), styles["code"])
                items.append(_boxed_table(pre, "#f6f8fa", "#d0d7de"))
                items.append(Spacer(1, 2 * mm))
            i += 1
            continue

        s = stripped.strip()

        # ATX headings (#### → h3 as lowest level)
        if s.startswith("#### "):
            items.append(_safe_para(inline_md(s[5:]), styles["h3"]))
        elif s.startswith("### "):
            items.append(_safe_para(inline_md(s[4:]), styles["h3"]))
        elif s.startswith("## "):
            items.append(_safe_para(inline_md(s[3:]), styles["h2"]))
        elif s.startswith("# "):
            items.append(_safe_para(inline_md(s[2:]), styles["h1"]))

        # Horizontal rule
        elif re.match(r"^(-{3,}|\*{3,}|_{3,})$", s):
            items.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#d0d7de"),
                    spaceAfter=3 * mm,
                )
            )

        # Unordered list
        elif re.match(r"^[-*+]\s", s):
            items.append(
                _safe_para(f"&bull;&nbsp;{inline_md(s[2:])}", styles["bullet"])
            )

        # Ordered list
        elif re.match(r"^\d+\.\s", s):
            match = re.match(r"^(\d+\.)\s+(.*)", s)
            if match:
                num, content = match.groups()
                items.append(
                    _safe_para(f"{num}&nbsp;{inline_md(content)}", styles["bullet"])
                )

        # Blockquote
        elif s.startswith("> "):
            bq_style = ParagraphStyle(
                "blockquote",
                parent=styles["body"],
                leftIndent=14,
                textColor=colors.HexColor("#586069"),
                borderPad=3,
            )
            items.append(_safe_para(inline_md(s[2:]), bq_style))

        # Regular paragraph
        else:
            items.append(_safe_para(inline_md(s), styles["body"]))

        i += 1

    return items


# ─── CUSTOM DOCUMENT TEMPLATE ─────────────────────────────────────────────────


class NotebookDocTemplate(BaseDocTemplate):
    """BaseDocTemplate with header, footer, and automatic TOC support."""

    def __init__(self, filename: str, nb_title: str, **kwargs):
        self.nb_title = nb_title
        kwargs.setdefault("pagesize", A4)
        kwargs.setdefault("leftMargin", L_MARGIN)
        kwargs.setdefault("rightMargin", R_MARGIN)
        kwargs.setdefault("topMargin", T_MARGIN)
        kwargs.setdefault("bottomMargin", B_MARGIN)
        BaseDocTemplate.__init__(self, filename, **kwargs)

        # Cover: full-height frame, no header/footer callback
        cover_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="cover",
        )
        # Content: slightly shorter frame to leave room for header + footer
        content_frame = Frame(
            self.leftMargin,
            self.bottomMargin + 12 * mm,
            self.width,
            self.height - 20 * mm,
            id="content",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[cover_frame]),
                PageTemplate(
                    id="content",
                    frames=[content_frame],
                    onPage=self._header_footer,
                ),
            ]
        )

    def _header_footer(self, canvas, doc):
        """Draw page header and footer on every content page."""
        canvas.saveState()

        # ── Header ────────────────────────────────────────────────────────────
        y_hdr = PAGE_H - T_MARGIN + 5 * mm
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor("#0d1117"))
        canvas.drawString(L_MARGIN, y_hdr, self.nb_title)

        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#586069"))
        canvas.drawRightString(PAGE_W - R_MARGIN, y_hdr, "Notebook PDF Report")

        canvas.setStrokeColor(colors.HexColor("#d0d7de"))
        canvas.setLineWidth(0.5)
        canvas.line(L_MARGIN, y_hdr - 2 * mm, PAGE_W - R_MARGIN, y_hdr - 2 * mm)

        # ── Footer ────────────────────────────────────────────────────────────
        y_ftr = B_MARGIN - 8 * mm
        canvas.line(L_MARGIN, y_ftr + 5 * mm, PAGE_W - R_MARGIN, y_ftr + 5 * mm)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#586069"))
        canvas.drawString(L_MARGIN, y_ftr, datetime.now().strftime("%Y-%m-%d"))
        canvas.drawCentredString(PAGE_W / 2, y_ftr, f"Page {canvas.getPageNumber()}")
        canvas.drawRightString(PAGE_W - R_MARGIN, y_ftr, "Generated by pdf_agent.py")

        canvas.restoreState()

    def afterFlowable(self, flowable):
        """Register heading Paragraphs as TOC entries."""
        if isinstance(flowable, Paragraph):
            sn = flowable.style.name
            if sn == "h1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
            elif sn == "h2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))
            elif sn == "h3":
                self.notify("TOCEntry", (2, flowable.getPlainText(), self.page))


# ─── TOOLS ────────────────────────────────────────────────────────────────────


@tool
def parse_notebook(notebook_path: str) -> str:
    """Parse a Jupyter Notebook (.ipynb) and return a structural summary.

    Args:
        notebook_path: Path to the .ipynb notebook file.

    Returns:
        JSON string with notebook metadata, cell counts, and heading preview.
    """
    path = Path(notebook_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {notebook_path}"})
    if path.suffix != ".ipynb":
        return json.dumps({"error": f"Not a .ipynb file: {notebook_path}"})

    with open(path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    output_count = sum(len(c.get("outputs", [])) for c in code_cells)

    # Preview the first 10 headings from markdown cells
    headings = []
    for cell in md_cells:
        for line in "".join(cell.get("source", [])).split("\n"):
            if line.startswith("#"):
                headings.append(line.strip())
            if len(headings) >= 10:
                break
        if len(headings) >= 10:
            break

    kernel = nb.get("metadata", {}).get("kernelspec", {}).get("display_name", "Python")

    return json.dumps(
        {
            "notebook_name": path.stem,
            "notebook_path": str(path.absolute()),
            "kernel": kernel,
            "total_cells": len(cells),
            "markdown_cells": len(md_cells),
            "code_cells": len(code_cells),
            "total_outputs": output_count,
            "headings_preview": headings,
            "suggested_output": f"{path.stem}.pdf",
        },
        indent=2,
    )


@tool
def generate_pdf(notebook_path: str, output_path: str) -> str:
    """Generate a professionally formatted PDF report from a Jupyter Notebook.

    Parses markdown cells, code cells, and cell outputs. Produces a PDF with:
    - Cover page (title, date, kernel, cell counts)
    - Table of contents (auto-generated from markdown headings)
    - Content pages with header (title) and footer (page numbers)

    Args:
        notebook_path: Path to the source .ipynb notebook file.
        output_path: Desired path for the output PDF (e.g., pdf_agent.pdf).

    Returns:
        Success message with the output path, or an error message.
    """
    path = Path(notebook_path)
    if not path.exists():
        return f"Error: File not found: {notebook_path}"

    try:
        with open(path) as f:
            nb = json.load(f)
    except Exception as e:
        return f"Error reading notebook: {e}"

    nb_title = path.stem.replace("_", " ").replace("-", " ")
    styles = build_styles()

    # ── Table of Contents flowable ─────────────────────────────────────────────
    toc = TableOfContents()
    toc.levelStyles = [styles["toc0"], styles["toc1"], styles["toc2"]]
    toc.dotsMinLevel = 0

    # ── Build story ───────────────────────────────────────────────────────────
    story = []
    cells = nb.get("cells", [])
    kernel = nb.get("metadata", {}).get("kernelspec", {}).get("display_name", "Python")
    md_count = sum(1 for c in cells if c.get("cell_type") == "markdown")
    code_count = sum(1 for c in cells if c.get("cell_type") == "code")

    # ── Cover Page (uses 'cover' template — no header/footer) ─────────────────
    story.append(Spacer(1, 50 * mm))
    story.append(Paragraph(nb_title, styles["doc_title"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Jupyter Notebook · PDF Report", styles["doc_subtitle"]))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y  at  %H:%M')}",
            styles["doc_subtitle"],
        )
    )
    story.append(Paragraph(f"Kernel: {kernel}", styles["doc_subtitle"]))
    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            f"{len(cells)}&nbsp;cells &nbsp;·&nbsp; "
            f"{md_count}&nbsp;markdown &nbsp;·&nbsp; "
            f"{code_count}&nbsp;code",
            styles["doc_subtitle"],
        )
    )

    # Switch to content template (header + footer) for all subsequent pages
    story.append(NextPageTemplate("content"))
    story.append(PageBreak())

    # ── Table of Contents Page ─────────────────────────────────────────────────
    story.append(Paragraph("Table of Contents", styles["toc_title"]))
    story.append(Spacer(1, 3 * mm))
    story.append(toc)
    story.append(PageBreak())

    # ── Notebook Cells ─────────────────────────────────────────────────────────
    code_counter = 0
    for cell in cells:
        cell_type = cell.get("cell_type", "raw")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            story.extend(markdown_to_flowables(source, styles))

        elif cell_type == "code":
            code_counter += 1
            if source.strip():
                story.extend(make_code_cell(source, code_counter, styles))

            # Process outputs
            raw_outputs = cell.get("outputs", [])
            processed = []
            for out in raw_outputs:
                ot = out.get("output_type", "")
                if ot == "stream":
                    text = "".join(out.get("text", []))
                    processed.append({"type": "stream", "text": text})
                elif ot == "execute_result":
                    data = out.get("data", {})
                    if "text/plain" in data:
                        processed.append(
                            {"type": "text", "text": "".join(data["text/plain"])}
                        )
                    if "image/png" in data:
                        processed.append({"type": "image", "data": data["image/png"]})
                elif ot == "display_data":
                    data = out.get("data", {})
                    if "image/png" in data:
                        processed.append({"type": "image", "data": data["image/png"]})
                    elif "text/plain" in data:
                        processed.append(
                            {"type": "text", "text": "".join(data["text/plain"])}
                        )
                elif ot == "error":
                    processed.append(
                        {
                            "type": "error",
                            "ename": out.get("ename", "Error"),
                            "evalue": strip_ansi(out.get("evalue", "")),
                            "traceback": out.get("traceback", []),
                        }
                    )
            story.extend(make_output_cell(processed, code_counter, styles))

    # ── Build PDF (two-pass for accurate TOC page numbers) ────────────────────
    doc = NotebookDocTemplate(output_path, nb_title=nb_title)
    try:
        doc.multiBuild(story)
    except Exception as e:
        return f"Error building PDF: {e}"

    return f"PDF generated successfully: {output_path}"


# ─── AGENT SETUP ──────────────────────────────────────────────────────────────

tools = [parse_notebook, generate_pdf]

SYSTEM_PROMPT = """You are a Notebook-to-PDF Agent. You convert Jupyter Notebook (.ipynb) files \
into professionally formatted PDF reports.

Your workflow:
1. Use parse_notebook to examine the notebook structure and metadata.
2. Use generate_pdf to create the formatted PDF report.
3. Report the generated PDF path to the user.

Guidelines:
- Always call parse_notebook first to understand the notebook contents.
- Then call generate_pdf with the notebook path and the desired output PDF path.
- Name the output PDF after the notebook file (replace .ipynb with .pdf).
- If the user specifies a different output name, use that instead.
- Confirm success by reporting the exact path of the generated PDF.
"""


@wrap_tool_call
def _handle_tool_error(request, handler):
    try:
        return handler(request)
    except Exception as err:
        print(f"Tool error: {err}")
        return f"Error: {err}"


_tool_retry = ToolRetryMiddleware(
    max_retries=2,
    max_delay=30,
    backoff_factor=1.5,
    tools=["parse_notebook", "generate_pdf"],
    on_failure="continue",
)


def create_notebook_agent():
    """Create and configure the Notebook-to-PDF agent."""
    llm = ChatOllama(model="minimax-m2.5:cloud", temperature=0)
    conn = sqlite3.connect("pdf_agent.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
        middleware=[_handle_tool_error, _tool_retry],
        interrupt_before=["tools"],
        name="notebook_pdf_agent",
    )
    return agent


# ─── CLI HELPERS ──────────────────────────────────────────────────────────────


def banner():
    print("=" * 62)
    print("  Notebook → PDF Agent  │  pdf_agent.py")
    print("=" * 62)
    print("  Convert any .ipynb notebook into a professional PDF report.")
    print("  Type 'quit' or 'exit' to stop.\n")


def stream_response(agent, query: str, config: dict):
    """Stream the agent response, pausing for tool-call approval."""
    current_input = {"messages": [HumanMessage(content=query)]}

    while True:
        for chunk in agent.stream(current_input, config=config, stream_mode="values"):
            latest = chunk["messages"][-1]
            if latest.content and isinstance(latest, AIMessage):
                print(f"Agent: {latest.content}")
            elif hasattr(latest, "tool_calls") and latest.tool_calls:
                print(f"  → Tools queued: {[tc['name'] for tc in latest.tool_calls]}")

        state = agent.get_state(config)
        if not state.next:
            break  # Graph finished

        last_msg = state.values["messages"][-1]
        if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            break

        # Human-in-the-loop: approve each pending tool call
        for tc in last_msg.tool_calls:
            print(f"\n  [Tool Call]  {tc['name']}  |  args: {tc.get('args', {})}")
            answer = input("  Allow? (yes/no): ").strip().lower()
            if answer not in ("yes", "y"):
                print(f"  → Cancelled '{tc['name']}'. Returning to prompt.")
                return

        current_input = Command(resume=True)


def main():
    """Main entry point."""
    banner()
    agent = create_notebook_agent()
    config = {"configurable": {"thread_id": "nb2pdf-session-1"}}

    # CLI argument mode: uv run pdf_agent.py path/to/notebook.ipynb
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
        output_name = sys.argv[2] if len(sys.argv) > 2 else "pdf_agent.pdf"
        query = (
            f"Convert the notebook '{notebook_path}' to a PDF report "
            f"named '{output_name}'."
        )
        print(f"You: {query}")
        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f"Error: {err}")
        return

    # Interactive mode
    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f"Error: {err}")


main()
