"""
Shared figure helpers.

`save_panels` writes every axes of a composed figure as its own image, so that
each file contains exactly one plot. It re-renders from the live figure object
rather than cropping the finished PNG, so the panels keep full resolution and
each one carries its own title, labels and legend.

The panel name is taken from the axes title's "(a) ..." prefix when there is
one, so fig1's panels land as fig1_a.png ... fig1_f.png and stay in step with
the manuscript's legends.
"""
from __future__ import annotations
import re
from pathlib import Path

PANEL_RE = re.compile(r"^\s*\(([a-z])\)")


def _panel_key(ax, i: int) -> str:
    """(a)/(b)/... from the title if present, otherwise a positional index."""
    for text in (ax.get_title(), ax.get_title("left")):
        m = PANEL_RE.match(text or "")
        if m:
            return m.group(1)
    return f"p{i + 1}"


def save_panels(fig, outdir, prefix: str, dpi: int = 200, pad: float = 0.10,
                skip_empty: bool = True) -> list[Path]:
    """Save each axes of `fig` to `outdir/prefix_<panel>.png`.

    Uses the axes' tight bounding box, expanded by `pad` inches so that titles,
    tick labels and any legend anchored inside the axes are included. `pad` is
    deliberately small: a generous pad reaches across the gap between subplots
    and pulls the neighbouring panel's axis label into the crop.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # A figure-level legend (fig.legend) is not attached to any axes, so it would
    # vanish from every split panel. Give such panels their own legend for the
    # duration of the save, so each file stands on its own, then take it away
    # again to leave the composed figure exactly as it was.
    temp_legends = []
    if not fig.legends:
        pass
    else:
        for ax in fig.get_axes():
            if ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
                temp_legends.append(ax.legend(fontsize=8, frameon=False,
                                              loc="best"))

    fig.canvas.draw()                      # tight bboxes need a rendered canvas
    all_axes = list(fig.get_axes())
    suptitle = fig._suptitle
    written = []
    for i, ax in enumerate(all_axes):
        if skip_empty and not (ax.has_data() or ax.get_title() or ax.images):
            continue
        # Hide everything that does not belong to this panel. Cropping alone is
        # not enough: a neighbouring axis label sitting in the gap between
        # subplots falls inside the crop and bleeds into the saved image.
        for other in all_axes:
            other.set_visible(other is ax)
        if suptitle is not None:
            suptitle.set_visible(False)
        for lg in fig.legends:
            lg.set_visible(False)

        fig.canvas.draw()
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(pad)
        out = outdir / f"{prefix}_{_panel_key(ax, i)}.png"
        fig.savefig(out, dpi=dpi, bbox_inches=bbox, facecolor="white")
        written.append(out)

    # restore the composed figure exactly as it was
    for other in all_axes:
        other.set_visible(True)
    if suptitle is not None:
        suptitle.set_visible(True)
    for lg in fig.legends:
        lg.set_visible(True)

    for lg in temp_legends:
        lg.remove()
    return written


def maybe_split(fig, args, outdir, prefix: str, dpi: int = 200) -> None:
    """Honour a --split flag if the script's argparse namespace carries one."""
    if getattr(args, "split", False):
        paths = save_panels(fig, outdir, prefix, dpi=dpi)
        print(f"  split into {len(paths)} single-panel files -> {outdir}")
        for p in paths:
            print(f"    {p.name}")
