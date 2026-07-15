"""Staged 3D animation of random search and Powell refinement."""

from __future__ import annotations

from dataclasses import dataclass
from math import log10
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D

from parameter_finder import (
    Evaluation,
    ParameterBounds,
    Parameters,
    RefinementTrace,
)


@dataclass(frozen=True)
class AnimationSettings:
    """Frame counts and rendering controls for the search animation."""

    fps: int = 24
    dpi: int = 100
    space_frames: int = 12
    population_frames: int = 18
    selection_frames: int = 18
    trajectory_frames: int = 60
    hold_frames: int = 12

    def validate(self) -> None:
        """Reject settings that cannot produce a valid animation."""
        if self.fps <= 0 or self.dpi <= 0:
            raise ValueError("animation fps and dpi must be positive")
        frame_counts = (
            self.space_frames,
            self.population_frames,
            self.selection_frames,
            self.trajectory_frames,
            self.hold_frames,
        )
        if any(count <= 0 for count in frame_counts):
            raise ValueError("every animation phase must contain at least one frame")


def write_search_animation(
    path: Path,
    random_results: list[Evaluation],
    refinements: list[RefinementTrace],
    best: Evaluation,
    bounds: ParameterBounds,
    settings: AnimationSettings,
) -> None:
    """Render bounds, 20,000 candidates, retained starts, and Powell paths."""
    settings.validate()
    if path.suffix.lower() != ".mp4":
        raise ValueError("search animation output must use the .mp4 extension")
    if not random_results:
        raise ValueError("animation needs at least one random-search result")
    if not refinements:
        raise ValueError("animation needs at least one Powell refinement trace")
    if len(refinements) > len(random_results):
        raise ValueError("refinement count exceeds random candidate count")

    retained_count = len(refinements)

    # random_results is sorted by loss. The same first N entries were passed to
    # Powell, so everything after N is the population that disappears.
    discarded_points = _evaluation_points(random_results[retained_count:])
    retained_points = np.asarray(
        [_parameter_point(trace.start.parameters) for trace in refinements],
        dtype=float,
    )

    # Powell paths are stored in natural-log coordinates. Divide by ln(10) so
    # they share the plot's log10 coordinates and physical labels.
    paths = [
        np.asarray(trace.log_path, dtype=float) / np.log(10.0)
        for trace in refinements
    ]
    best_point = _parameter_point(best.parameters)

    figure = plt.figure(figsize=(10.0, 8.0))
    axis = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(left=0.02, right=0.84, bottom=0.04, top=0.96)
    _configure_axes(axis, bounds)
    # Keep one stable viewpoint throughout the video. Parameter movement is
    # easier to compare when the coordinate frame itself does not rotate.
    axis.view_init(elev=24, azim=-62)

    neutral = "#7f8793"
    retained_color = "#f2a93b"
    best_color = "#d94b4b"

    # Separate collections let discarded candidates fade while the best 20 stay.
    cloud = axis.scatter(
        discarded_points[:, 0],
        discarded_points[:, 1],
        discarded_points[:, 2],
        s=4,
        c=neutral,
        alpha=0.0,
        depthshade=False,
        rasterized=True,
    )
    retained = axis.scatter(
        retained_points[:, 0],
        retained_points[:, 1],
        retained_points[:, 2],
        s=28,
        c=neutral,
        alpha=0.0,
        depthshade=False,
    )

    # Each Powell start keeps one colored line and one moving endpoint marker.
    color_map = plt.colormaps["turbo"]
    colors = [color_map(index / max(1, retained_count - 1)) for index in range(retained_count)]
    lines = []
    movers = []
    for color in colors:
        line, = axis.plot([], [], [], color=color, linewidth=1.6, alpha=0.9)
        mover = axis.scatter([], [], [], s=24, c=[color], depthshade=False)
        lines.append(line)
        movers.append(mover)

    # The final selected tuple appears only when trajectories reach their ends.
    winner = axis.scatter(
        [best_point[0]],
        [best_point[1]],
        [best_point[2]],
        marker="*",
        s=180,
        c=best_color,
        edgecolors="none",
        alpha=0.0,
        depthshade=False,
    )

    status = axis.text2D(0.02, 0.96, "", transform=axis.transAxes)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=neutral, label="Random candidates"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=retained_color,
                label="Best 20 starts",
            ),
            Line2D([], [], color=colors[0], linewidth=2, label="Powell trajectories"),
            Line2D(
                [],
                [],
                marker="*",
                linestyle="none",
                color=best_color,
                markersize=12,
                label="Selected result",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 0.95),
        frameon=False,
    )

    phase_1 = settings.space_frames
    phase_2 = phase_1 + settings.population_frames
    phase_3 = phase_2 + settings.selection_frames
    phase_4 = phase_3 + settings.trajectory_frames
    total_frames = phase_4 + settings.hold_frames

    def clear_paths() -> None:
        """Hide every trajectory and moving endpoint."""
        for line, mover in zip(lines, movers):
            line.set_data([], [])
            line.set_3d_properties([])
            mover._offsets3d = ([], [], [])

    def show_path_progress(progress: float) -> None:
        """Reveal the same fraction of every recorded Powell path."""
        for line, mover, trajectory in zip(lines, movers, paths):
            last_index = int(round(progress * (len(trajectory) - 1)))
            visible = trajectory[: last_index + 1]
            line.set_data(visible[:, 0], visible[:, 1])
            line.set_3d_properties(visible[:, 2])
            endpoint = visible[-1]
            mover._offsets3d = ([endpoint[0]], [endpoint[1]], [endpoint[2]])

    def update(frame: int):
        """Update one frame across the four requested visual phases."""
        if frame < phase_1:
            # Phase 1: show only the bounded 3D parameter space.
            cloud.set_alpha(0.0)
            retained.set_alpha(0.0)
            winner.set_alpha(0.0)
            clear_paths()
            status.set_text("1. Parameter space constrained by configured bounds")

        elif frame < phase_2:
            # Phase 2: fade all 20,000 random-stage candidates into the space.
            progress = (frame - phase_1 + 1) / settings.population_frames
            cloud.set_alpha(0.22 * progress)
            retained.set_color(neutral)
            retained.set_alpha(0.22 * progress)
            winner.set_alpha(0.0)
            clear_paths()
            status.set_text(f"2. Initial population: {len(random_results):,} candidates")

        elif frame < phase_3:
            # Phase 3: remove every point except the retained Powell starts.
            progress = (frame - phase_2 + 1) / settings.selection_frames
            cloud.set_alpha(0.22 * (1.0 - progress))
            retained.set_color(retained_color)
            retained.set_alpha(0.22 + 0.78 * progress)
            winner.set_alpha(0.0)
            clear_paths()
            status.set_text(f"3. Preserve the best {retained_count} Powell starting points")

        elif frame < phase_4:
            # Phase 4: draw completed Powell iteration endpoints.
            progress = (frame - phase_3 + 1) / settings.trajectory_frames
            cloud.set_alpha(0.0)
            retained.set_color(retained_color)
            retained.set_alpha(0.45)
            show_path_progress(progress)
            winner.set_alpha(1.0 if progress >= 1.0 else 0.0)
            status.set_text(f"4. Powell refinement trajectories: {progress:>5.0%}")

        else:
            # Final hold: keep completed paths and winner visible for readability.
            cloud.set_alpha(0.0)
            retained.set_alpha(0.35)
            show_path_progress(1.0)
            winner.set_alpha(1.0)
            status.set_text("Selected minimum after random search and Powell refinement")

        return (
            cloud,
            retained,
            winner,
            status,
            *lines,
            *movers,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=total_frames,
        interval=1000 / settings.fps,
        blit=False,
        repeat=False,
    )

    # Matplotlib uses the system ffmpeg executable installed with Homebrew.
    writer = FFMpegWriter(
        fps=settings.fps,
        codec="libx264",
        bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Parameter search and Powell refinement"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=writer, dpi=settings.dpi)
    plt.close(figure)


def _evaluation_points(evaluations: list[Evaluation]) -> np.ndarray:
    """Convert evaluations into an N-by-3 log10 plotting array."""
    return np.asarray(
        [_parameter_point(evaluation.parameters) for evaluation in evaluations],
        dtype=float,
    )


def _parameter_point(parameters: Parameters) -> np.ndarray:
    """Convert physical A, E_ref, and k into log10 plot coordinates."""
    return np.log10((parameters.A, parameters.E_ref, parameters.k))


def _configure_axes(axis, bounds: ParameterBounds) -> None:
    """Constrain the 3D space and label logarithmic positions physically."""
    axis.set_xlim(log10(bounds.A[0]), log10(bounds.A[1]))
    axis.set_ylim(log10(bounds.E_ref[0]), log10(bounds.E_ref[1]))
    axis.set_zlim(log10(bounds.k[0]), log10(bounds.k[1]))
    axis.set_xlabel("A (log scale)")
    axis.set_ylabel("E_ref, kJ/mol (log scale)")
    axis.set_zlabel("k (log scale)")

    _set_ticks(axis.set_xticks, axis.set_xticklabels, (1, 10, 100), bounds.A)
    _set_ticks(
        axis.set_yticks,
        axis.set_yticklabels,
        (50, 100, 300, 1000),
        bounds.E_ref,
    )
    _set_ticks(
        axis.set_zticks,
        axis.set_zticklabels,
        (10, 100, 1_000, 10_000, 100_000, 1_000_000),
        bounds.k,
    )
    axis.grid(True, linewidth=0.4, alpha=0.35)


def _set_ticks(set_positions, set_labels, candidates, limits) -> None:
    """Show physical values at their log10 positions when inside bounds."""
    values = [value for value in candidates if limits[0] <= value <= limits[1]]
    set_positions([log10(value) for value in values])
    set_labels([f"{value:g}" for value in values])
