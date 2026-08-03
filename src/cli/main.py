"""
Main Command-Line Interface (CLI)
---------------------------------
Powered by Typer and Rich to provide a terminal interface for:
- Data Audit & Leakage Verification (`ekegusii-nmt audit`)
- Task Generation (`ekegusii-nmt generate-tasks`)
- Model Fine-Tuning (`ekegusii-nmt train`)
- Evaluation & Attribution Report (`ekegusii-nmt evaluate`)
"""

import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.data_processing.instruction_task_generator import InstructionTaskGenerator
from src.evaluation.resource_attribution_analyzer import ResourceAttributionAnalyzer
from src.master_corpus.integrity import DataLeakageChecker
from src.master_corpus.manager import MasterCorpusManager

app = typer.Typer(
    name="ekegusii-nmt",
    help="Resource-Aware Translation Framework for Ekegusii, Kiswahili, and English.",
    add_completion=False,
)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@app.command()
def audit() -> None:
    """Run data integrity and zero-leakage verification audit on Master Corpus."""
    console.print("[bold blue]=== Starting Master Corpus Data Leakage Audit ===[/bold blue]")
    manager = MasterCorpusManager()
    checker = DataLeakageChecker(manager)
    try:
        checker.verify_all()
        console.print("[bold green]✅ 0% Data Leakage Verified Across Splits![/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Data Audit Failed: {e}[/bold red]")
        sys.exit(1)


@app.command()
def generate_tasks() -> None:
    """Generate 6-way multilingual instruction-tuning tasks."""
    console.print("[bold blue]=== Generating 6-Way Multilingual Instruction Tasks ===[/bold blue]")
    generator = InstructionTaskGenerator()
    splits = generator.generate_all_splits()
    console.print(f"[bold green]Generated {len(splits['train']):,} Train tasks![/bold green]")


@app.command()
def evaluate() -> None:
    """Run SacreBLEU, chrF++, and Lexical evaluation attribution report."""
    console.print("[bold blue]=== Computing Resource Attribution Report ===[/bold blue]")
    analyzer = ResourceAttributionAnalyzer()
    report_df = analyzer.generate_full_attribution_report()

    table = Table(title="Resource Attribution Matrix (E0 - E8)")
    for col in report_df.columns:
        table.add_column(col)
    for _, row in report_df.iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)


if __name__ == "__main__":
    app()
