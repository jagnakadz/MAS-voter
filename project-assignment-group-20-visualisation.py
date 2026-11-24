"""
Group 20 – MAS project: INTERACTIVE VISUALISATION of the voter model

Run (from the project folder):
    solara run project-assignment-group-20-visualisation.py
A browser page opens in which you can change the settings and watch the model.

This file only SHOWS the model. The model itself is imported unchanged from
project-assignment-group-20.py, and the experiment results (results/) do not use
this file. The visualisation uses Mesa's SolaraViz             [Mesa, SolaraViz]
(requires the `mesa[viz]` extra, see requirements.txt).

Annotation: [Mesa] = standard Mesa visualisation pattern; [OWN] = our own.
"""

import importlib.util
import os
from dataclasses import replace

import networkx as nx
import numpy as np
import solara
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mesa import DataCollector
from mesa.visualization import CommandConsole, SolaraViz, Slider
from mesa.visualization.utils import update_counter

# --- load the model file (its name contains hyphens, so a normal `import` fails)
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "voter_model", os.path.join(_here, "project-assignment-group-20.py"))
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)


# =============================================================================
# COLOURS AND CHART STYLE                                                [OWN]
# =============================================================================
# Opinion (left <-> right) is a diverging scale: blue = left, grey = centre,
# red = right. The four parties use steps of the same two hues, so a party's
# colour matches the colour of its voters everywhere. Party colours were checked
# for colour-blind safety (all pairs distinguishable); the lighter ones are also
# labelled directly, so colour is never the only cue.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

OPINION_CMAP = LinearSegmentedColormap.from_list(
    "left_right", ["#184f95", "#f0efec", "#a52a29"])
PARTY_COLOURS = ["#184f95", "#5598e7", "#e36462", "#a52a29"]
PARTY_NAMES = ["Left", "Centre-left", "Centre-right", "Right"]
SWITCHED_COLOUR = "#eb6834"
TRUST_COLOUR = "#1baf7a"


def new_axes(width=5.6, height=3.6):
    """An empty chart with our quiet style: light grid, no box, grey labels."""
    fig = Figure(figsize=(width, height), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def title(ax, text, subtitle=None):
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(text, loc="left", fontsize=11, color=INK, pad=8 + 11 * lines)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2,
                va="bottom")


def label_line_ends(ax, x, ends, min_gap=0.06):
    """Direct labels at the end of lines, in ink colour (not the line colour).
    `ends` = list of (y, text); labels that would overlap are pushed apart."""
    placed = []
    for y, text in sorted(ends):
        y_label = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_label)
        ax.annotate(text, (x, y_label), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)


# =============================================================================
# MODEL ADAPTER FOR THE VISUALISATION                              [OWN, Mesa]
# =============================================================================

class VoterModelViz(vm.VoterModel):
    """The unchanged voter model, plus what the visualisation needs.

    1) Mesa's sliders pass settings one by one (p=..., e=...), but our model
       takes one Params bundle, so this class builds the bundle.
    2) A DataCollector records population measures every tick for the line
       charts.                                                      [Mesa]
    3) Fixed drawing positions for the network (computed once).
    """

    def __init__(self, p=0.5, e=0.2, b=0.0, tau_c0=0.5, sigma=0.05, N=500,
                 initial_dist="uniform", network_type="small_world", seed=0):
        params = replace(vm.PARAMS, p=float(p), e=float(e), b=float(b),
                         tau_c0=float(tau_c0), sigma=float(sigma), N=int(N),
                         initial_dist=initial_dist, network_type=network_type,
                         seed=int(seed))
        super().__init__(params)

        # network drawn as a spiral in network order: in a small world most
        # friends are "next to each other" (Watts & Strogatz ring), so friends sit
        # next to each other on the spiral and local opinion groups show up as
        # stretches of one colour; the few long-distance links cross the spiral.
        # (A circle crowds 500 dots on one line; force layouts give a hairball.)
        pos = nx.spiral_layout(self.network, equidistant=True, resolution=0.6)
        self.xy = np.array([pos[v.node] for v in self.voter_list])
        self.edge_segments = (None if network_type == "complete"   # too many lines
                              else [(pos[a], pos[b]) for a, b in self.network.edges])

        reporters = {name: (lambda m, k=k: m.party_shares()[k])
                     for k, name in enumerate(PARTY_NAMES)}
        reporters["Share switched"] = lambda m: m.share_switched()
        reporters["Mean computer trust"] = lambda m: m.computer_trusts().mean()
        self.datacollector = DataCollector(model_reporters=reporters)   # [Mesa]
        self.datacollector.collect(self)

    def step(self):
        super().step()
        self.datacollector.collect(self)
        # stop "Play" at the run length used in the experiment
        self.running = self.steps < self.params.T


# =============================================================================
# CHARTS (each is a plain function, so it can also be tested without a browser)
# =============================================================================

def draw_network(model):
    fig = Figure(figsize=(5.2, 5.6), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    if model.edge_segments is not None:
        ax.add_collection(LineCollection(model.edge_segments, colors=GRID,
                                         linewidths=0.5, zorder=1))
    dots = ax.scatter(model.xy[:, 0], model.xy[:, 1], c=model.opinions(),
                      cmap=OPINION_CMAP, vmin=0, vmax=1, s=18, zorder=2,
                      edgecolors=MUTED, linewidths=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    subtitle = ("neighbours on the spiral are friends; grey lines = friendships"
                if model.edge_segments is not None
                else "complete network: everyone knows everyone (lines hidden)")
    title(ax, "Voters in the network", f"each dot is a voter, coloured by opinion\n{subtitle}")
    bar = fig.colorbar(dots, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                       ticks=[0, 0.5, 1])
    bar.ax.set_xticklabels(["0 left", "0.5", "1 right"], color=INK_2, fontsize=8.5)
    bar.outline.set_visible(False)
    return fig


def draw_histogram(model):
    fig, ax = new_axes()
    bins = np.linspace(0, 1, 41)
    counts, _ = np.histogram(model.opinions(), bins=bins)
    centres = (bins[:-1] + bins[1:]) / 2
    ax.bar(centres, counts, width=bins[1] - bins[0], color=OPINION_CMAP(centres),
           edgecolor=SURFACE, linewidth=0.8)
    # party borders and names (borders = halfway between party positions)
    positions = model.params.party_positions
    for left, right in zip(positions[:-1], positions[1:]):
        ax.axvline((left + right) / 2, color=MUTED, linestyle=(0, (3, 3)), linewidth=0.8)
    top = max(counts.max(), 1) * 1.18
    for pos, name in zip(positions, PARTY_NAMES):
        ax.text(pos, top * 0.97, name, ha="center", va="top", fontsize=8.5, color=INK_2)
    ax.set(xlim=(0, 1), ylim=(0, top))
    ax.set_xlabel("opinion (0 = left, 1 = right)", color=INK_2, fontsize=9)
    ax.set_ylabel("number of voters", color=INK_2, fontsize=9)
    title(ax, "Opinions now", "dashed lines = borders between parties")
    return fig


def draw_trust(model):
    fig, ax = new_axes()
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.scatter(model.opinions(), model.computer_trusts(), c=model.opinions(),
               cmap=OPINION_CMAP, vmin=0, vmax=1, s=16, alpha=0.85,
               edgecolors=MUTED, linewidths=0.3)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.04, 1.04))
    ax.set_xlabel("opinion (0 = left, 1 = right)", color=INK_2, fontsize=9)
    ax.set_ylabel("trust in the computer", color=INK_2, fontsize=9)
    sub = "each dot is a voter"
    if model.params.p == 0:
        sub += " – computer never used (p = 0), so trust stays at its start value"
    title(ax, "Who trusts the computer?", sub)
    return fig


def draw_lines(model, names, colours, chart_title, subtitle):
    fig, ax = new_axes()
    df = model.datacollector.get_model_vars_dataframe()
    ticks = np.arange(len(df))
    for name, colour in zip(names, colours):
        ax.plot(ticks, df[name], color=colour, linewidth=2, label=name)
    label_line_ends(ax, ticks[-1],
                    [(df[n].iloc[-1], f"{n} {df[n].iloc[-1]:.0%}") for n in names])
    ax.set(ylim=(0, 1), xlim=(0, max(ticks[-1], 10) * 1.3))   # room for labels
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("tick", color=INK_2, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2,
              ncol=len(names))
    title(ax, chart_title, subtitle)
    return fig


def draw_party_shares(model):
    return draw_lines(model, PARTY_NAMES, PARTY_COLOURS, "Vote share per party",
                      "share of voters closest to each party")


def draw_switching_and_trust(model):
    return draw_lines(model, ["Share switched", "Mean computer trust"],
                      [SWITCHED_COLOUR, TRUST_COLOUR], "Party switching and trust",
                      "switched = party now differs from party at tick 0")


# =============================================================================
# SOLARA COMPONENTS (re-drawn after every step)                          [Mesa]
# =============================================================================

def _component(draw):
    @solara.component
    def Chart(model):
        update_counter.get()            # [Mesa] re-draw when the model steps
        solara.FigureMatplotlib(draw(model), format="png", bbox_inches="tight")
    return Chart


@solara.component
def HowToRead(model):
    update_counter.get()
    p = model.params
    solara.Markdown(
        f"**Tick {model.steps} of {p.T}.** In one tick every voter asks either the "
        f"computer (chance p = {p.p:.2f}) or a random friend, and moves towards the "
        f"answer only if it is within e = {p.e:.2f} of their own opinion. "
        f"The computer reports the mean opinion of {p.k} random voters "
        f"+ bias b = {p.b:.2f}.\n\n"
        "**Change settings:** move the sliders, then press **Reset**.")


model_params = {
    "p": Slider("Computer use p", 0.5, 0.0, 1.0, 0.05),
    "e": Slider("Openness e", 0.2, 0.05, 0.5, 0.05),
    "b": Slider("Computer bias b", 0.0, 0.0, 0.3, 0.05),
    "tau_c0": Slider("Starting computer trust", 0.5, 0.0, 1.0, 0.05),
    "sigma": Slider("Noise on human messages", 0.05, 0.0, 0.2, 0.01),
    "N": Slider("Number of voters", 500, 50, 1000, 50),
    "initial_dist": {"type": "Select", "value": "uniform",
                     "values": ["uniform", "bimodal"], "label": "Starting opinions"},
    "network_type": {"type": "Select", "value": "small_world",
                     "values": ["small_world", "complete"], "label": "Network"},
    "seed": {"type": "InputText", "value": 0, "label": "Random seed"},
}

# all charts shown on the page
COMPONENTS = [HowToRead,
              _component(draw_network),
              _component(draw_histogram),
              _component(draw_party_shares),
              _component(draw_trust),
              _component(draw_switching_and_trust),
              CommandConsole]          # [Mesa] type Python commands, e.g. model.share_switched()
TITLE = "Group 20 – Voters, people and computers"

page = SolaraViz(                                                       # [Mesa]
    VoterModelViz(),
    components=COMPONENTS,
    model_params=model_params,
    play_interval=150,
    name=TITLE,
)
