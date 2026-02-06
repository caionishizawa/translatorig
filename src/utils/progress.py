"""
Progress tracking utilities for Video Translator.
"""

from typing import Callable, Optional
from dataclasses import dataclass, field
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel

from .models import ProcessingStatus


@dataclass
class StepProgress:
    """Track progress of a single processing step."""
    name: str
    status: ProcessingStatus
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    error: Optional[str] = None


@dataclass
class ProgressTracker:
    """Track overall progress of video translation pipeline."""

    steps: dict = field(default_factory=dict)
    current_step: Optional[str] = None
    callback: Optional[Callable] = None
    console: Console = field(default_factory=Console)

    STEP_NAMES = [
        ("extraction", "Extracting Video"),
        ("transcription", "Transcribing Audio"),
        ("translation", "Translating Text"),
        ("voice_synthesis", "Synthesizing Voice"),
        ("lip_sync", "Synchronizing Lips"),
        ("rendering", "Rendering Final Video"),
    ]

    def __post_init__(self):
        """Initialize all steps."""
        for step_id, step_name in self.STEP_NAMES:
            self.steps[step_id] = StepProgress(
                name=step_name,
                status=ProcessingStatus.QUEUED
            )

    def start_step(self, step_id: str, message: str = "") -> None:
        """Mark a step as started."""
        if step_id in self.steps:
            self.steps[step_id].status = ProcessingStatus.EXTRACTING
            self.steps[step_id].message = message
            self.steps[step_id].progress = 0.0
            self.current_step = step_id
            self._notify()

    def update_step(self, step_id: str, progress: float, message: str = "") -> None:
        """Update progress of a step."""
        if step_id in self.steps:
            self.steps[step_id].progress = min(1.0, max(0.0, progress))
            if message:
                self.steps[step_id].message = message
            self._notify()

    def complete_step(self, step_id: str, message: str = "") -> None:
        """Mark a step as completed."""
        if step_id in self.steps:
            self.steps[step_id].status = ProcessingStatus.COMPLETED
            self.steps[step_id].progress = 1.0
            if message:
                self.steps[step_id].message = message
            self._notify()

    def fail_step(self, step_id: str, error: str) -> None:
        """Mark a step as failed."""
        if step_id in self.steps:
            self.steps[step_id].status = ProcessingStatus.FAILED
            self.steps[step_id].error = error
            self._notify()

    def get_overall_progress(self) -> float:
        """Calculate overall progress (0.0 to 1.0)."""
        total = sum(step.progress for step in self.steps.values())
        return total / len(self.steps) if self.steps else 0.0

    def get_status(self) -> dict:
        """Get current status as a dictionary."""
        return {
            "overall_progress": self.get_overall_progress(),
            "current_step": self.current_step,
            "steps": {
                step_id: {
                    "name": step.name,
                    "status": step.status.value,
                    "progress": step.progress,
                    "message": step.message,
                    "error": step.error,
                }
                for step_id, step in self.steps.items()
            },
        }

    def _notify(self) -> None:
        """Notify callback of progress update."""
        if self.callback:
            self.callback(self.get_status())

    def print_summary(self) -> None:
        """Print a summary of all steps."""
        table = Table(title="Processing Summary")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Progress", justify="right")
        table.add_column("Message", style="dim")

        for step_id, step in self.steps.items():
            status_color = {
                ProcessingStatus.COMPLETED: "green",
                ProcessingStatus.FAILED: "red",
                ProcessingStatus.QUEUED: "dim",
            }.get(step.status, "yellow")

            table.add_row(
                step.name,
                f"[{status_color}]{step.status.value}[/]",
                f"{step.progress * 100:.0f}%",
                step.message or step.error or "",
            )

        self.console.print(table)


def create_progress_bar() -> Progress:
    """Create a rich progress bar for the pipeline."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(),
    )
