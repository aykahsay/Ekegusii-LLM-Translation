"""
Main Command-Line Interface (CLI)
---------------------------------
Powered by Typer and Rich to provide a terminal interface for:
- Data Audit & Leakage Verification (`ekegusii-nmt audit`)
- Task Generation (`ekegusii-nmt generate-tasks`)
- Resource Attribution Report (`ekegusii-nmt evaluate`)
- Model Fine-Tuning (`ekegusii-nmt train`)
- Single-Model Automatic Evaluation (`ekegusii-nmt run-eval`)
- Interactive Translation (`ekegusii-nmt translate`)
- Resource-Scheduler Preview (`ekegusii-nmt schedule-preview`)
- Ablation Analysis & Final-Model Selection (`ekegusii-nmt analyze`)
"""

import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

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
        console.print("[bold green]0% Data Leakage Verified Across Splits![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Data Audit Failed: {e}[/bold red]")
        sys.exit(1)


@app.command()
def generate_tasks() -> None:
    """Generate 6-way multilingual instruction-tuning tasks for all splits."""
    from src.cli.generate_tasks import run_generate_tasks

    console.print("[bold blue]=== Generating 6-Way Multilingual Instruction Tasks ===[/bold blue]")
    splits = run_generate_tasks()
    console.print(f"[bold green]Generated {len(splits['train']):,} Train tasks![/bold green]")


@app.command()
def evaluate() -> None:
    """Aggregate saved per-experiment results into the E0-E8 attribution matrix."""
    console.print("[bold blue]=== Computing Resource Attribution Report ===[/bold blue]")
    analyzer = ResourceAttributionAnalyzer()
    report_df = analyzer.generate_full_attribution_report()

    table = Table(title="Resource Attribution Matrix (E0 - E8)")
    for col in report_df.columns:
        table.add_column(col)
    for _, row in report_df.iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)


@app.command()
def train(
    experiment_id: str = typer.Argument(..., help="One of E1-E7 (see src.cli.train.TRAINABLE_EXPERIMENTS)."),
    model_name: str = typer.Option("qwen", help="'qwen' or 'llama'."),
) -> None:
    """Fine-tune a model (Qwen/Llama) on one experiment's resource configuration."""
    from src.cli.train import run_train

    console.print(f"[bold blue]=== Training {model_name} on {experiment_id} ===[/bold blue]")
    try:
        run_train(experiment_id, model_name)
        console.print(f"[bold green]Training complete for {model_name}/{experiment_id}![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        sys.exit(1)


@app.command(name="run-eval")
def run_eval_command(
    model_name: str = typer.Option("qwen", help="'qwen' or 'llama'."),
    source_lang: str = typer.Option("English"),
    target_lang: str = typer.Option("Ekegusii"),
    adapter_path: Optional[str] = typer.Option(None, help="Trained LoRA checkpoint path; omit for zero-shot."),
) -> None:
    """Run inference + automatic metrics for one model/direction against the test split."""
    from src.cli.evaluate import run_evaluate

    console.print(f"[bold blue]=== Evaluating {model_name}: {source_lang} -> {target_lang} ===[/bold blue]")
    results = run_evaluate(model_name, source_lang, target_lang, adapter_path)
    for key, value in results.items():
        console.print(f"  {key}: {value}")


@app.command()
def translate(
    text: str = typer.Argument(..., help="Sentence to translate."),
    source_lang: str = typer.Option("English"),
    target_lang: str = typer.Option("Ekegusii"),
    model_name: str = typer.Option("qwen", help="'qwen' or 'llama'."),
    adapter_path: Optional[str] = typer.Option(None, help="Trained LoRA checkpoint path; omit for zero-shot."),
) -> None:
    """Translate a single sentence interactively."""
    from src.cli.translate import run_translate

    result = run_translate([text], source_lang, target_lang, model_name, adapter_path)
    console.print(f"[bold green]{result[0]}[/bold green]")


@app.command(name="schedule-preview")
def schedule_preview_command(batch_size: int = typer.Option(32)) -> None:
    """Preview the resource-weighted per-direction sampling quota for one batch."""
    from src.cli.scheduler import run_schedule_preview

    quota = run_schedule_preview(batch_size)
    table = Table(title=f"Resource Schedule Preview (batch_size={batch_size})")
    table.add_column("Direction")
    table.add_column("Quota")
    for direction, count in quota.items():
        table.add_row(direction, str(count))
    console.print(table)


@app.command()
def analyze(model_key: str = typer.Option("qwen", help="'qwen' or 'llama'.")) -> None:
    """Build the ablation attribution matrix and recommend the E8 final-model source."""
    from src.cli.analyze import run_analyze

    result = run_analyze(model_key)
    table = Table(title="Ablation Attribution Matrix")
    for col in result["attribution_table"].columns:
        table.add_column(col)
    for _, row in result["attribution_table"].iterrows():
        table.add_row(*[str(val) for val in row])
    console.print(table)

    recommended = result["recommended_final_model"]
    if recommended:
        console.print(f"[bold green]Recommended E8 Final Model source: {recommended}[/bold green]")
    else:
        console.print("[yellow]No experiment results saved yet -- run some experiments first.[/yellow]")


if __name__ == "__main__":
    app()
