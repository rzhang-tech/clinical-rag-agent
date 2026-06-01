from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors

OUT = r"D:\project\clinical-rag-agent\Zhang_Chen_Individual contribution.pdf"

doc = SimpleDocTemplate(
    OUT,
    pagesize=letter,
    leftMargin=1.0 * inch,
    rightMargin=1.0 * inch,
    topMargin=0.9 * inch,
    bottomMargin=0.9 * inch,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleCenter",
    parent=styles["Title"],
    fontSize=16,
    alignment=TA_CENTER,
    spaceAfter=6,
)
subtitle_style = ParagraphStyle(
    "SubCenter",
    parent=styles["Normal"],
    fontSize=11,
    alignment=TA_CENTER,
    textColor=colors.grey,
    spaceAfter=18,
)
heading_style = ParagraphStyle(
    "MemberHeading",
    parent=styles["Heading2"],
    fontSize=13,
    spaceBefore=8,
    spaceAfter=6,
    textColor=colors.HexColor("#1a3d6d"),
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10.5,
    leading=14,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
)
bullet_style = ParagraphStyle(
    "Bullet",
    parent=styles["Normal"],
    fontSize=10.5,
    leading=14,
    leftIndent=18,
    bulletIndent=6,
    spaceAfter=3,
)

story = []

story.append(Paragraph("Individual Contribution Statement", title_style))
story.append(Paragraph(
    "Project: An Agentic RAG System for Medical Literature Question Answering",
    subtitle_style,
))

# Ruoyu Zhang
story.append(Paragraph("Ruoyu Zhang (ruz0002)", heading_style))
story.append(Paragraph(
    "Ruoyu was responsible for the project end-to-end, covering topic selection, "
    "system design, full code implementation, and all written reports.",
    body_style,
))
ruoyu_items = [
    "<b>Project topic selection and scoping</b>: identified the agentic medical RAG problem, "
    "defined the target benchmark (MedQA), and pivoted the project from a multimodal "
    "Med-VQA design to a text-only RAG design after finding that image-based datasets "
    "were poorly suited for retrieval evaluation.",
    "<b>System architecture and code implementation</b>: designed and implemented the full "
    "two-level LangGraph agent, including the top-level graph (history summarization, "
    "query rewrite and decomposition, fan-out, aggregation) and the per-sub-question "
    "agent subgraph (orchestrator, tool calling, routing, fallback, answer collection).",
    "<b>Retrieval and indexing pipeline</b>: hierarchical parent-child chunking algorithm; "
    "Qdrant hybrid retrieval integrating dense embeddings (all-mpnet-base-v2) and BM25 "
    "sparse search; cross-encoder reranking with ms-marco-MiniLM-L-6-v2; parent-store "
    "storage layer.",
    "<b>Agent mechanisms</b>: token-aware context compression with dynamic threshold; "
    "retrieval-key deduplication to prevent redundant searches; hard-limit fallback path; "
    "structured query decomposition via Pydantic schemas.",
    "<b>Prompt engineering</b>: authored all six domain-specific prompts (orchestrator, "
    "rewrite, summarize, compress, fallback, aggregate) with strict grounding rules and "
    "format enforcement for medical safety.",
    "<b>Evaluation and iteration</b>: built the automated MedQA evaluation harness; ran "
    "the v1 -> v2 -> v3 optimization cycle; analyzed failure modes and produced the "
    "qualitative case studies.",
    "<b>User interface</b>: Gradio chat interface with source-attribution display and "
    "streaming output.",
    "<b>Written reports</b>: sole author of the project proposal, mid-term report, and "
    "final report; produced all figures, tables, and LaTeX typesetting.",
]
for item in ruoyu_items:
    story.append(Paragraph("\u2022 " + item, bullet_style))

story.append(Spacer(1, 10))

# Xu Chen
story.append(Paragraph("Xu Chen (xzc0066)", heading_style))
story.append(Paragraph(
    "Xu was responsible for the presentation deliverables.",
    body_style,
))
xu_items = [
    "<b>Presentation slide deck</b>: designed and produced the slide deck summarizing the "
    "project motivation, approach, architecture, and experimental results.",
    "<b>Presentation delivery</b>: presented the project during class presentations.",
]
for item in xu_items:
    story.append(Paragraph("\u2022 " + item, bullet_style))

story.append(Spacer(1, 16))
story.append(Paragraph(
    "<i>Both team members acknowledge and agree to the contribution breakdown above.</i>",
    ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=9.5,
        alignment=TA_CENTER,
        textColor=colors.grey,
    ),
))

doc.build(story)
print(f"PDF written to: {OUT}")
