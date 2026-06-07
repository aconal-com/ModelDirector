"""Command-line interface for ModelDirector.

Usage:
    modeldirector select PROMPT --config CONFIG.yaml
    modeldirector select PROMPT_FILE --config CONFIG.yaml
    echo "PROMPT" | modeldirector select --config CONFIG.yaml
    modeldirector score PROMPT --config CONFIG.yaml
    modeldirector validate --config CONFIG.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from modeldirector.loader import load_config
from modeldirector.selector import ModelDirector, SelectorError


def _read_prompt(prompt: Optional[str], file: Optional[Path]) -> str:
    if file is not None:
        return file.read_text()
    if prompt is not None:
        return prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise click.UsageError(
        "Provide a prompt via --prompt, --file, or stdin."
    )


@click.group()
@click.version_option()
def cli() -> None:
    """ModelDirector - stateless AI model selection engine."""


@cli.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--prompt", "-p", default=None, help="Prompt text. Use --file or stdin if omitted.")
@click.option("--file", "-f", "file", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--pretty/--compact", default=True, help="Pretty-print the JSON output.")
def select(config_path: Path, prompt: Optional[str], file: Optional[Path], pretty: bool) -> None:
    """Pick the best model for the given prompt."""
    text = _read_prompt(prompt, file)
    cfg = load_config(config_path)
    director = ModelDirector(cfg)
    try:
        result = director.select(text)
    except SelectorError as e:
        click.echo(f"Selector failed: {e}", err=True)
        sys.exit(1)
    click.echo(json.dumps(result.model_dump(), indent=2 if pretty else None))


@cli.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--prompt", "-p", default=None)
@click.option("--file", "-f", "file", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--pretty/--compact", default=True)
def score(config_path: Path, prompt: Optional[str], file: Optional[Path], pretty: bool) -> None:
    """Print raw per-model scores without applying a policy."""
    text = _read_prompt(prompt, file)
    cfg = load_config(config_path)
    director = ModelDirector(cfg)
    try:
        scores = director.score(text)
    except SelectorError as e:
        click.echo(f"Selector failed: {e}", err=True)
        sys.exit(1)
    click.echo(json.dumps({k: v.model_dump() for k, v in scores.items()}, indent=2 if pretty else None))


@cli.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def validate(config_path: Path) -> None:
    """Validate a config file.  Exits 0 on success, 1 on error."""
    try:
        cfg = load_config(config_path)
    except Exception as e:
        click.echo(f"Invalid: {e}", err=True)
        sys.exit(1)
    click.echo(f"OK - {len(cfg.models)} candidate model(s), policy={cfg.policy.type}, "
               f"selector={cfg.selector.provider}/{cfg.selector.model}")


@cli.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def show(config_path: Path) -> None:
    """Print the resolved configuration."""
    cfg = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    raw["resolved"] = {
        "selector": cfg.selector.model_dump(),
        "policy": cfg.policy.model_dump(),
        "model_count": len(cfg.models),
    }
    click.echo(yaml.safe_dump(raw, sort_keys=False))


if __name__ == "__main__":
    cli()
