"""
Command Line Interface for Sakthi Finance Financial Document RAG System.
Supports: doctor, db init, inspect, ingest, ingest-dir, ask, chat, retrieve, stats, documents list, documents delete
"""

import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from ragcli.config import settings
from ragcli.database import check_db_health, init_database
from ragcli.ollama_client import OllamaClient
from ragcli.logging_config import setup_logging
from ragcli.pdf_inspector import PDFInspector
from ragcli.pdf_extractor import PDFExtractor
from ragcli.ingestion import IngestionEngine
from ragcli.retrieval import HybridRetriever
from ragcli.answer_generator import AnswerGenerator
from ragcli.repository import DocumentRepository

app = typer.Typer(
    name="ragcli",
    help="Sakthi Finance Financial Document RAG System - Terminal CLI",
    add_completion=False,
)
console = Console()


@app.command()
def doctor():
    """Inspect environment, PostgreSQL database, pgvector extension, and Ollama models."""
    setup_logging()
    console.print(Panel.fit("[bold blue]Sakthi Finance RAG - System Health Check[/bold blue]"))

    table = Table(title="Environment Diagnostics", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="dim", width=24)
    table.add_column("Status", width=12)
    table.add_column("Details")

    # 1. Python Environment
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python", "[green]OK[/green]", f"Python {py_ver}")

    # 2. Database Connectivity & pgvector
    db_health = check_db_health()
    if db_health["connected"]:
        table.add_row("PostgreSQL Connection", "[green]OK[/green]", f"Connected to {settings.db_name} on {settings.db_host}:{settings.db_port}")
        if db_health["pgvector_installed"]:
            table.add_row("pgvector Extension", "[green]OK[/green]", f"Version {db_health['pgvector_version']} installed")
        else:
            table.add_row("pgvector Extension", "[red]FAIL[/red]", "pgvector extension is NOT installed or enabled")
        
        tables_str = ", ".join(db_health["tables"]) if db_health["tables"] else "No tables found (run 'ragcli db init')"
        table.add_row("Database Tables", "[green]OK[/green]" if db_health["tables"] else "[yellow]WARN[/yellow]", tables_str)
    else:
        table.add_row("PostgreSQL Connection", "[red]FAIL[/red]", "; ".join(db_health["errors"]))
        table.add_row("pgvector Extension", "[red]FAIL[/red]", "Cannot check pgvector without database connection")

    # 3. Ollama Connectivity & Models
    ollama = OllamaClient()
    o_health = ollama.check_health()
    if o_health["available"]:
        table.add_row("Ollama Service", "[green]OK[/green]", f"Online at {settings.ollama_base_url}")
        if o_health["llm_ready"]:
            table.add_row("LLM Model (Qwen)", "[green]OK[/green]", f"Model '{settings.llm_model}' ready")
        else:
            table.add_row("LLM Model (Qwen)", "[red]FAIL[/red]", f"Model '{settings.llm_model}' not found in Ollama")

        if o_health["embedding_ready"]:
            table.add_row("Embedding Model", "[green]OK[/green]", f"Model '{settings.embedding_model}' ({settings.embedding_dim}-dim) ready")
        else:
            table.add_row("Embedding Model", "[yellow]WARN[/yellow]", f"Model '{settings.embedding_model}' not found in Ollama")
    else:
        table.add_row("Ollama Service", "[red]FAIL[/red]", f"Cannot connect: {o_health['error']}")

    console.print(table)


# Subcommand: db
db_app = typer.Typer(name="db", help="Database management commands")
app.add_typer(db_app, name="db")


@db_app.command("init")
def db_init():
    """Initialize database tables, extensions, and indexes."""
    setup_logging()
    console.print("[bold yellow]Initializing PostgreSQL database and pgvector schemas...[/bold yellow]")
    try:
        init_database()
        console.print("[bold green]Database schema initialized successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to initialize database: {e}[/bold red]")
        sys.exit(1)


@app.command()
def inspect(
    pdf_path: str = typer.Argument(..., help="Path to PDF file"),
    page: Optional[int] = typer.Option(None, "--page", "-p", help="Preview specific page"),
):
    """Inspect PDF structure, extractable text, scans, and tables before ingestion."""
    setup_logging()
    path = Path(pdf_path)
    if not path.exists():
        console.print(f"[bold red]File not found: {path}[/bold red]")
        sys.exit(1)

    inspector = PDFInspector(path)
    doc_rec, page_recs = inspector.inspect()

    console.print(Panel(f"[bold cyan]PDF Document: {doc_rec.filename}[/bold cyan]\n"
                        f"Pages: {doc_rec.page_count} | Size: {doc_rec.file_size_bytes / 1024:.1f} KB | Hash: {doc_rec.file_hash[:16]}..."))

    if page is not None:
        if page < 1 or page > doc_rec.page_count:
            console.print(f"[bold red]Page {page} out of range (1-{doc_rec.page_count})[/bold red]")
            sys.exit(1)

        extractor = PDFExtractor(path)
        _, extracted_pages = extractor.extract()
        target = next((ep for ep in extracted_pages if ep.page_number == page), None)

        if target:
            console.print(f"\n[bold yellow]--- Page {page} Extraction Preview ---[/bold yellow]")
            console.print(f"Characters: {target.record.character_count} | Tables Found: {len(target.tables)}")
            if target.tables:
                for idx, t in enumerate(target.tables, 1):
                    console.print(f"\n[bold green]Table {idx}: {t.title or 'Untitled'} (Unit: {t.units or 'Unstated'})[/bold green]")
                    console.print(Markdown(t.markdown_repr))
            console.print("\n[bold]Narrative Snippet:[/bold]")
            console.print(target.narrative_text[:800] + ("..." if len(target.narrative_text) > 800 else ""))
    else:
        table = Table(title="Page Inspection Summary", show_header=True)
        table.add_column("Page", justify="right", width=6)
        table.add_column("Characters", justify="right", width=12)
        table.add_column("Tables", justify="right", width=8)
        table.add_column("Type", width=12)
        table.add_column("Warnings")

        for pr in page_recs[:30]:  # preview first 30 pages
            page_type = "[red]Scanned[/red]" if pr.is_scanned else "[green]Native Text[/green]"
            warns = "; ".join(pr.warnings) if pr.warnings else "-"
            table.add_row(str(pr.page_number), str(pr.character_count), str(pr.table_count), page_type, warns)

        if len(page_recs) > 30:
            table.add_row("...", "...", "...", "...", f"{len(page_recs) - 30} more pages")

        console.print(table)


@app.command()
def ingest(pdf_path: str = typer.Argument(..., help="Path to PDF file to ingest")):
    """Ingest a PDF document into PostgreSQL with financial chunking and vector embeddings."""
    setup_logging()
    engine = IngestionEngine()
    try:
        res = engine.ingest_pdf(pdf_path, console=console)
        if res.get("status") == "skipped":
            console.print(f"[yellow]Document already ingested (Duplicate hash: {res['file_hash'][:16]}...)[/yellow]")
        else:
            console.print(f"[bold green]Successfully ingested {res['filename']}![/bold green]")
            console.print(f"Pages: {res['page_count']} | Total Chunks: {res['chunk_count']} "
                          f"(Tables: {res['table_chunk_count']}, Narrative: {res['narrative_chunk_count']})")
    except Exception as e:
        console.print(f"[bold red]Ingestion failed: {e}[/bold red]")
        sys.exit(1)


@app.command("ingest-dir")
def ingest_dir(dir_path: str = typer.Argument(..., help="Directory containing PDF files")):
    """Ingest all PDF documents in a directory."""
    setup_logging()
    path = Path(dir_path)
    if not path.is_dir():
        console.print(f"[bold red]Directory not found: {path}[/bold red]")
        sys.exit(1)

    pdf_files = list(path.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {path}[/yellow]")
        return

    console.print(f"[cyan]Found {len(pdf_files)} PDF files to ingest...[/cyan]")
    engine = IngestionEngine()
    for pdf in pdf_files:
        try:
            res = engine.ingest_pdf(pdf, console=console)
            console.print(f"Processed: {pdf.name} - Status: {res.get('status')}")
        except Exception as e:
            console.print(f"[red]Error processing {pdf.name}: {e}[/red]")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Search query or financial metric"),
    top_k: int = typer.Option(settings.retrieval_top_k, "--top-k", "-k", help="Number of chunks to retrieve"),
    period: Optional[str] = typer.Option(None, "--period", "-p", help="Filter by financial period (e.g. FY 2024-25)"),
):
    """Retrieve and display source chunks matching a query without calling LLM."""
    setup_logging()
    retriever = HybridRetriever()
    chunks = retriever.retrieve(query, top_k=top_k, financial_period=period)

    if not chunks:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    console.print(Panel(f"[bold green]Retrieved {len(chunks)} Chunks for Query: '{query}'[/bold green]"))
    for idx, c in enumerate(chunks, 1):
        console.print(f"\n[bold cyan]── Match {idx} | Page {c.page_start} | Type: {c.chunk_type} | Period: {c.financial_period or 'N/A'} | Unit: {c.unit or 'N/A'} ──[/bold cyan]")
        if c.section_title:
            console.print(f"[dim]Section/Table: {c.section_title}[/dim]")
        console.print(c.normalized_text[:600] + ("..." if len(c.normalized_text) > 600 else ""))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Financial question to ask"),
    period: Optional[str] = typer.Option(None, "--period", "-p", help="Financial period filter"),
):
    """Ask a one-time question and get a source-grounded answer with citations."""
    setup_logging()
    retriever = HybridRetriever()
    generator = AnswerGenerator()

    console.print(f"[cyan]Searching financial documents for:[/cyan] [bold]{question}[/bold]")
    chunks = retriever.retrieve(question, financial_period=period)

    res = generator.generate_answer(question, chunks)

    console.print()
    console.print(Panel(Markdown(f"**Answer:**\n\n{res.answer}"), title="[bold green]Sakthi Finance RAG Answer[/bold green]"))

    if res.citations:
        table = Table(title="Verified Source Citations", show_header=True, header_style="bold cyan")
        table.add_column("Doc / Page", width=25)
        table.add_column("Section / Table", width=25)
        table.add_column("Period", width=15)
        table.add_column("Unit", width=15)

        for cit in res.citations:
            table.add_row(
                f"{cit.document_name} (p.{cit.page_number})",
                cit.section_title or cit.table_id or "-",
                cit.financial_period or "-",
                cit.unit or "-",
            )
        console.print(table)
    console.print(f"[dim]Latency: {res.latency_ms:.1f}ms | Confidence: {res.confidence}[/dim]")


@app.command()
def chat():
    """Start an interactive terminal question-and-answer session with conversation context."""
    setup_logging()
    retriever = HybridRetriever()
    generator = AnswerGenerator()

    console.print(Panel.fit("[bold blue]Sakthi Finance Financial Document RAG - Interactive Chat[/bold blue]\nType 'exit' or 'quit' to end session."))

    history: list[dict[str, str]] = []

    while True:
        try:
            question = console.input("\n[bold cyan]Ask financial question > [/bold cyan]").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                console.print("[green]Session ended.[/green]")
                break

            chunks = retriever.retrieve(question)
            res = generator.generate_answer(question, chunks, conversation_history=history)

            console.print()
            console.print(Panel(Markdown(res.answer), title="[bold green]Answer[/bold green]"))

            if res.citations:
                c_summary = ", ".join(f"{c.document_name} p.{c.page_number}" for c in res.citations[:3])
                console.print(f"[dim]Citations: {c_summary}[/dim]")

            # Update history
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": res.answer})

        except (KeyboardInterrupt, EOFError):
            console.print("\n[green]Session ended.[/green]")
            break


@app.command()
def stats():
    """Display ingestion and document statistics."""
    setup_logging()
    repo = DocumentRepository()
    s = repo.get_stats()

    table = Table(title="System & Ingestion Statistics", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Indexed Documents", str(s["total_documents"]))
    table.add_row("Total Extracted Pages", str(s["total_pages"]))
    table.add_row("Total Chunks", str(s["total_chunks"]))
    for ctype, count in s["chunk_types"].items():
        table.add_row(f"  - {ctype.capitalize()} Chunks", str(count))
    table.add_row("Total Vector Embeddings", str(s["total_embeddings"]))

    console.print(table)


# Subcommand: documents
doc_app = typer.Typer(name="documents", help="Manage indexed documents")
app.add_typer(doc_app, name="documents")


@doc_app.command("list")
def doc_list():
    """List all indexed documents."""
    setup_logging()
    repo = DocumentRepository()
    docs = repo.list_documents()

    if not docs:
        console.print("[yellow]No documents indexed yet. Use 'ragcli ingest <path>' to index one.[/yellow]")
        return

    table = Table(title="Indexed Documents", show_header=True)
    table.add_column("ID", style="dim", width=36)
    table.add_column("Filename", width=30)
    table.add_column("Pages", justify="right", width=8)
    table.add_column("Chunks", justify="right", width=8)
    table.add_column("Created", width=20)

    for d in docs:
        table.add_row(d["id"], d["filename"], str(d["page_count"]), str(d["chunk_count"]), d["created_at"] or "-")

    console.print(table)


@doc_app.command("delete")
def doc_delete(document_id: str = typer.Argument(..., help="Document UUID to delete")):
    """Remove a document and all its chunks and embeddings from database."""
    setup_logging()
    repo = DocumentRepository()
    success = repo.delete_document(document_id)
    if success:
        console.print(f"[bold green]Successfully deleted document {document_id}[/bold green]")
    else:
        console.print(f"[bold red]Document {document_id} not found.[/bold red]")


def main():
    app()


if __name__ == "__main__":
    main()
