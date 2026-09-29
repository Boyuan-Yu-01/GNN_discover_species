"""Plain logs, progress-safe console output and PNG plots; no video output."""
import math
import os
import sys
import textwrap
import traceback
from pathlib import Path
from collections import Counter
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from tqdm import tqdm
from .molecule import Molecule

class TrainingLogger:
    """Control console/file detail while keeping tqdm bars readable.

    Level 0 is silent. Level 1 prints stage objectives and enables progress
    bars. Level 2 also writes the existing detailed messages to console and log.
    """

    def __init__(self, path, append=False, level=2):
        if type(level) is not int or level not in (0, 1, 2):
            raise ValueError("log level must be 0, 1, or 2")
        self.level = level
        self.path = Path(path)
        self.file = None
        if level == 0:
            self.path.unlink(missing_ok=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("a" if append else "w", encoding="utf-8")

    @property
    def progress_enabled(self):
        # Cluster jobs can retain stage/log messages while suppressing the
        # redraw-heavy tqdm output in a Slurm text log.
        return self.level >= 1 and os.environ.get("SELF_UPDATING_SHOW_PROGRESS_BARS", "1") != "0"

    def write(self, message):
        if self.level == 0:
            return
        tqdm.write(message, file=sys.stdout)
        self.file.write(message + "\n")
        self.file.flush()

    def info(self, message):
        if self.level == 2:
            self.write(message)

    def objective(self, message):
        if self.level >= 1:
            self.write(f"\n=== {message} ===")

    def stage(self, label, status, **details):
        """Write one stable, compact lifecycle record to the master log.

        ``pipeline.log`` uses these records as its complete chronology.  Child
        training and exploration logs may contain richer diagnostics, but they
        must not be needed to determine the active stage or its outcome.
        """
        if self.level < 1:
            return
        suffix = "".join(f" | {key}={value}" for key, value in details.items()
                         if value is not None)
        self.objective(f"{label} | {status}{suffix}")

    def exception(self, message):
        if self.level == 2:
            self.write(message + "\n" + traceback.format_exc().rstrip())
        elif self.level == 1:
            self.objective(message)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class MoleculeDrawing:
    """One chemical coordinate calculation for both PNG renderers."""

    @staticmethod
    def positions(graph_record):
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        mol = Chem.RWMol()
        indices = {}
        for node in graph_record["nodes"]:
            atom = Chem.Atom(node["symbol"])
            atom.SetNoImplicit(True)
            if node.get("formal_charge") is not None:
                atom.SetFormalCharge(node["formal_charge"])
            indices[node["id"]] = mol.AddAtom(atom)
        orders = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        for edge in graph_record["edges"]:
            mol.AddBond(indices[edge["source"]], indices[edge["target"]], orders[edge["order"]])
        # Preserve explicit bonds/H. Layout never sanitizes or assigns radicals.
        rdDepictor.Compute2DCoords(mol)
        points = np.array([[mol.GetConformer().GetAtomPosition(i).x,
                            mol.GetConformer().GetAtomPosition(i).y] for i in indices.values()])
        points -= (points.max(axis=0) + points.min(axis=0)) / 2
        points /= max(float(np.abs(points).max()), 1e-8)
        return dict(zip(indices, points))


class ExplorationMedia:
    """Draw all distinct successful structures as PNG panels."""

    def __init__(self, symbols, valency, max_atoms, config):
        # Use a noninteractive canvas: files can be rendered without a GUI.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        self.plt, self.nx = plt, nx
        self.symbols, self.valency = symbols, valency
        self.max_atoms, self.config = max_atoms, config

    def layout(self, record):
        return MoleculeDrawing.positions(record["graph"])

    def draw(self, ax, record, graph_record, positions, caption=""):
        import numpy as np
        ax.clear()
        candidate = record["category"] == "new_candidate"
        accent = "#c76b13" if candidate else "#16847a"
        ax.set_facecolor("#fff8ef" if candidate else "#f2faf8")
        for spine in ax.spines.values():
            spine.set_color(accent)
        # Draw bond order with parallel strokes; single/double/triple remain explicit.
        for edge in graph_record["edges"]:
            a, b = positions[edge["source"]], positions[edge["target"]]
            delta = b-a
            normal = np.array([-delta[1], delta[0]]) / max(float(np.linalg.norm(delta)), 1e-9)
            order = edge["order"]
            for i in range(order):
                shift = normal * (i-(order-1)/2) * 0.045
                ax.plot([a[0]+shift[0], b[0]+shift[0]],
                        [a[1]+shift[1], b[1]+shift[1]], color="#66717c", lw=1.8, zorder=1)
        colors = {"C": "#566273", "H": "#bcdcff", "O": "#e16c68"}
        for node in graph_record["nodes"]:
            x, y = positions[node["id"]]
            ax.scatter([x], [y], s=290, c=colors[node["symbol"]],
                       edgecolors="white", linewidths=1, zorder=2)
            ax.text(x, y, node["symbol"], ha="center", va="center", fontsize=9,
                    color="white" if node["symbol"] == "C" else "#172b3a", zorder=3)
        c, h, o = record["generated_C_H_O"]
        frequency = (record["frequency_overall"] if record["requested_C_H_O"] is None
                     else record["frequency_within_requested_composition"])
        name = "Candidate" if candidate else record["species_key"]
        ax.set_title(f"{record['structure_id']}  {name} | C={c} H={h} O={o}"
                     f"\nCount: {record['occurrences']}  |  {frequency:.1%}",
                     fontsize=9, color=accent, pad=8)
        ax.set_xlabel(caption, fontsize=8, labelpad=6)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def save(self, records, output, summary, log):
        # Candidates appear first for convenient inspection; IDs still match JSON.
        records = sorted(records, key=lambda r: (r["category"] != "new_candidate",
                                                 -r["occurrences"], r["structure_id"]))
        positions = [self.layout(r) for r in records]
        columns = 5
        rows = max(1, math.ceil(len(records)/columns))
        fig, axes = self.plt.subplots(rows, columns, figsize=(18, rows*3.8+1.2), squeeze=False)
        try:
            for index, ax in enumerate(axes.flat):
                if index < len(records):
                    self.draw(ax, records[index], records[index]["graph"], positions[index])
                else:
                    ax.set_axis_off()
            fig.suptitle(
                f"{self.config.get('search_method', 'sampling').upper()} exploration | "
                f"{len(records)} distinct structures from {summary['total_attempts']:,} attempts"
                "\nOrange: new candidates (unvalidated) | Teal: reference matches | C: grey, H: blue, O: red",
                fontsize=15, y=0.99)
            fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=3.5, w_pad=1.2)
            fig.savefig(output / "final_structures.png", dpi=self.config.get("plot_dpi", 150))
        finally:
            self.plt.close(fig)
        log.info(f"Saved final structure plot: {output / 'final_structures.png'}")



class ComparisonPlot:
    """Draw accepted (green), unvalidated (red), then given (blue) graphs."""

    COLORS = {"accepted": ("#D5F2D8", "#246B35"),
              "unvalidated": ("#FADADD", "#A52632"),
              "given": ("#D9EAFE", "#245B96")}

    def __init__(self, config):
        self.config = config
        regular = font_manager.findfont("DejaVu Sans")
        bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
        self.font = ImageFont.truetype(regular, 14)
        self.small = ImageFont.truetype(regular, 12)
        self.bold = ImageFont.truetype(bold, 16)
        self.atom_font = ImageFont.truetype(bold, 16)
        self.heading = ImageFont.truetype(bold, 22)
        self.width, self.height = 360, 320

    @staticmethod
    def formula(graph):
        counts = Counter(d["symbol"] for _, d in graph.nodes(data=True))
        order = [s for s in ("C", "H") if s in counts] + sorted(set(counts) - {"C", "H"})
        return "".join(s + (str(counts[s]) if counts[s] != 1 else "") for s in order)

    def panel(self, record):
        graph = Molecule.make_graph(record["graph"]["nodes"], record["graph"]["edges"])
        matched = record["accepted"]
        background, accent = self.COLORS[record["display_group"]]
        canvas = Image.new("RGB", (self.width, self.height), background)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, self.width - 1, self.height - 1), outline=accent, width=2)
        names = " / ".join(record["matched_species"]) if matched else record["display_group"].capitalize()
        if record["display_group"] == "given":
            names = "Given: " + record.get("species_key", names)
        title = record["structure_id"] + "  " + names
        lines = textwrap.wrap(title, width=34)
        y = 10
        for line in lines:
            draw.text((10, y), line, fill=accent, font=self.bold)
            y += 21
        label = self.formula(graph) + "   count: " + str(record.get("occurrences", 0))
        draw.text((10, y + 2), label, fill="#253B4A", font=self.font)
        y += 29
        positions = MoleculeDrawing.positions(record["graph"])
        # Use one equal scale for both axes so the diagram is not distorted.
        top, bottom = y + 14, self.height - 22
        scale = min((self.width - 58) / 2.5, (bottom - top) / 2.5)
        points = {i: np.array([self.width / 2 + float(p[0]) * scale,
                              (top + bottom) / 2 - float(p[1]) * scale])
                  for i, p in positions.items()}
        for u, v, data in graph.edges(data=True):
            a, b = points[u], points[v]
            delta = b - a
            normal = np.array([-delta[1], delta[0]]) / max(float(np.linalg.norm(delta)), 1e-8)
            for stroke in range(data["order"]):
                shift = normal * (stroke - (data["order"] - 1) / 2) * 4
                draw.line((tuple(a + shift), tuple(b + shift)), fill="#66717C", width=2)
        colors = {"C": "#566273", "H": "#BCDCFF", "O": "#E16C68",
                  "N": "#8B9EEA", "He": "#C7A9E5", "Ar": "#C7A9E5"}
        for index, data in graph.nodes(data=True):
            x, yy = points[index]
            r = 13
            draw.ellipse((x-r, yy-r, x+r, yy+r), fill=colors.get(data["symbol"], "#C6CED5"),
                         outline="white", width=1)
            draw.text((x, yy), data["symbol"], anchor="mm", font=self.atom_font,
                      fill="white" if data["symbol"] == "C" else "#172B3A")
        return canvas

    def header(self, canvas, title, summary):
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), title, fill="#183C4C", font=self.heading)
        draw.text((12, 43),
                  f"{summary['total_structures']} structures; {summary['display_group_counts']['accepted']} green; "
                  f"{summary['display_group_counts']['unvalidated']} red; "
                  f"{summary['display_group_counts']['given']} blue.",
                  fill="#183C4C", font=self.font)
        draw.text((12, 65), "Green: accepted. Red: unvalidated. Blue: given. Within each color: most frequent first.",
                  fill="#183C4C", font=self.small)
        draw.text((12, 82), "Charge/electronic state is not inferred; multiple matching reference keys are shown together.",
                  fill="#183C4C", font=self.small)

    def save(self, records, output, summary, log, filename="final_structures.png",
             title="Molecular validation results"):
        columns = self.config["overview_columns"]
        overview_size = (240, 214)
        header_height = 108
        count = len(records)
        overview = Image.new("RGB", (columns * overview_size[0],
                             header_height + max(1, math.ceil(count / columns)) * overview_size[1]), "white")
        self.header(overview, title, summary)
        # Remove generated pages left by earlier versions; no new pages are made.
        pages = output / "pages"
        if pages.is_dir():
            for old in pages.glob("structures_*.png"):
                old.unlink()
            if not any(pages.iterdir()):
                pages.rmdir()
        for index, record in enumerate(records):
            panel = self.panel(record)
            small = panel.resize(overview_size, Image.Resampling.LANCZOS)
            overview.paste(small, ((index % columns) * overview_size[0],
                                  header_height + (index // columns) * overview_size[1]))
            panel.close()
            small.close()
            if (index + 1) % 100 == 0 or index == count - 1:
                log(f"Rendered {index+1}/{count} structures in {filename}")
        overview.save(output / filename)
        overview.close()


class Reporter:
    """Save evidence and training labels separately; plots never assign labels."""

    @staticmethod
    def display_records(groups, method=None):
        """Return validation-display records for all methods or one method.

        A structure can occur in both PUCT and direct sampling. Per-method plots
        keep it in both files, but count only that method's appearances so the
        ordering answers which structures each generator produced most often.
        """
        display_order = ["accepted", "unvalidated", "given"]
        records = []
        for group, rows in groups.items():
            display_group = "unvalidated" if group in ("rejected", "unknown") else group
            for row in rows:
                observations = row["observations"]
                if method is not None:
                    observations = [item for item in observations if item.get("method") == method]
                    if not observations:
                        continue
                records.append({**row,
                    "accepted": row["validation"]["decision"] == "accepted",
                    "display_group": display_group,
                    "occurrences": sum(item.get("occurrences", 0) for item in observations),
                    "observations": observations})
        records.sort(key=lambda row: (display_order.index(row["display_group"]),
                                      -row["occurrences"], row["structure_id"]))
        return records

    @staticmethod
    def display_summary(records, scope, method=None):
        display_order = ["accepted", "unvalidated", "given"]
        return {
            "method": method,
            "total_structures": len(records),
            "display_group_counts": {group: sum(record["display_group"] == group for record in records)
                                     for group in display_order},
            "display_group_order": display_order,
            "within_group_order": "method-specific occurrences descending; structure_id ascending for ties",
            "scope": scope,
        }

    @staticmethod
    def validation(groups, output, columns=10, plots=True, log=print):
        from .storage import Storage
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        Storage.write(output / 'species_groups.json', groups)
        records = Reporter.display_records(groups)
        summary = Reporter.display_summary(records, 'External validation decisions; reference absence alone is unknown.')
        summary['validation_group_counts'] = {key: len(rows) for key, rows in groups.items()}
        summary['within_group_order'] = 'all-method occurrences descending; structure_id ascending for ties'
        summary['training_labels'] = dict(Counter(row['training_label'] for rows in groups.values() for row in rows))
        Storage.write(output/'summary.json',summary)
        if plots:
            ComparisonPlot({'overview_columns':columns}).save(records,output,summary,log)
        return summary

    @staticmethod
    def method_validation(groups, output, method, columns=10, plots=True, log=print):
        """Save a method-specific three-category plot after validation finishes."""
        from .storage import Storage
        if method not in ("puct", "direct"):
            raise ValueError("method must be 'puct' or 'direct'")
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        records = Reporter.display_records(groups, method)
        summary = Reporter.display_summary(
            records, 'Validation evidence for this generator only; reference absence alone is unknown.', method)
        Storage.write(output / 'classified_summary.json', summary)
        if plots:
            ComparisonPlot({'overview_columns':columns}).save(
                records, output, summary, log, filename='classified_structures.png',
                title='PUCT validation results' if method == 'puct' else 'Direct sampling validation results')
        return summary
