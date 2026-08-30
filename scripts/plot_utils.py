"""Shared Matplotlib setup for publication figures."""

import arcadia_style as acs


def set_pub_style(
    font_size: int = 8,
    title_size: int = 8,
    tick_size: int = 6,
    legend_size: int = 7,
) -> None:
    """Apply the repository style with its compact publication type scale."""
    acs.setup(
        font_size=font_size,
        title_size=title_size,
        tick_size=tick_size,
        legend_size=legend_size,
    )
