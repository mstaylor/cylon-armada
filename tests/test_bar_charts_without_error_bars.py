"""Bar charts show the mean only (CLAUDE.md chart conventions)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.container import BarContainer

from results import chart_collectives, chart_collectives_compare, notebook_generator


def _barrier_row(n, latency, std, platform=None):
    row = {"collective": "barrier", "N": str(n), "msg_size": "0",
           "latency_p50_ms": str(latency), "latency_p50_ms_std": str(std)}
    if platform:
        row["platform"] = platform
    return row


def _capture(monkeypatch, module):
    figures = {}
    monkeypatch.setattr(module, "_save",
                        lambda fig, output_dir, name, fmt, dpi: figures.setdefault(name, fig))
    return figures


def _assert_bars_without_error_bars(fig):
    ax = fig.axes[0]
    assert ax.containers and all(isinstance(c, BarContainer) and c.errorbar is None
                                 for c in ax.containers)
    assert not ax.collections
    plt.close(fig)


def test_barrier_bar_chart_has_no_error_bars_when_runs_vary(monkeypatch, tmp_path):
    figures = _capture(monkeypatch, chart_collectives)
    rows = [_barrier_row(n, 1.0 + n, 0.5) for n in (1, 2, 4)]

    chart_collectives.chart_barrier_vs_n(rows, str(tmp_path), "png", 72)

    _assert_bars_without_error_bars(figures["collectives_barrier_vs_N"])


def test_platform_barrier_bar_chart_has_no_error_bars_when_runs_vary(monkeypatch, tmp_path):
    figures = _capture(monkeypatch, chart_collectives_compare)
    rows = [_barrier_row(n, 1.0 + n, 0.5, platform)
            for n in (1, 2, 4) for platform in ("rivanna", "fargate")]

    chart_collectives_compare.chart_barrier_vs_n_compare(rows, str(tmp_path), "png", 72)

    _assert_bars_without_error_bars(figures["collectives_barrier_vs_N_rivanna_vs_fargate"])


@pytest.mark.parametrize("cell", [notebook_generator._cell_reuse_rate,
                                  notebook_generator._cmp_cell_barrier_vs_n])
def test_notebook_bar_cells_draw_no_error_bars(cell):
    source = cell()
    assert "ax.bar(" in source
    assert "yerr" not in source