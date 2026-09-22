"""Stage III: compare generated graphs with the FFCMII reference workbook.

Run from self_improving/stageIII. Classes first, editable settings below, then
direct calls. The workbook and Stage II outputs are read only.
"""
import ast
import csv
import hashlib
import io
import json
import math
import posixpath
import textwrap
import time
import zipfile
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


class WorkbookGraphs:
    """Read the workbook's graph arrays using Python's built-in XLSX/XML support.

    No Excel installation, openpyxl, or network connection is needed. Only cached
    cell values are read; this reader never modifies or recalculates the workbook.
    """

    def __init__(self, path, sheet_name):
        self.path = Path(path)
        data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.records = []
        self.buckets = {}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            rows = self.read_rows(archive, sheet_name)
        required = {"Species key", "All node symbols JSON", "All bonds JSON",
                    "Base atoms JSON", "Base formal charges JSON"}
        header = None
        for row_number, row in rows:
            if required <= set(row.values()):
                header = {value: column for column, value in row.items()}
                continue
            if header is None:
                continue
            name = row.get(header["Species key"])
            if not name:
                continue
            if any(r["species_key"] == name for r in self.records):
                raise ValueError(f"Duplicate reference key: {name}")
            symbols = json.loads(row[header["All node symbols JSON"]])
            bonds = json.loads(row[header["All bonds JSON"]])
            base = json.loads(row[header["Base atoms JSON"]])
            charges = json.loads(row[header["Base formal charges JSON"]])
            if len(base) != len(charges) or symbols[:len(base)] != base:
                raise ValueError(f"{name}: inconsistent base atoms/formal charges")
            if any(type(charge) is not int for charge in charges):
                raise ValueError(f"{name}: formal charges must be integers")
            if any(symbol != "H" for symbol in symbols[len(base):]):
                raise ValueError(f"{name}: appended atoms must be explicit H")
            charges += [0] * (len(symbols) - len(base))
            graph = self.make_graph(
                [{"id": i, "symbol": symbol, "formal_charge": charges[i]}
                 for i, symbol in enumerate(symbols)],
                [{"source": u, "target": v, "order": order} for u, v, order in bonds])
            reference = {
                "species_key": name, "graph": graph, "workbook_row": row_number,
                "chemical_name": row.get(header.get("Chemical name", ""), ""),
                "state_notes": row.get(header.get("State / representation notes", ""), ""),
            }
            self.records.append(reference)
            self.buckets.setdefault(self.graph_key(graph), []).append(reference)
        if header is None or not self.records:
            raise ValueError(f"No graph definitions found in {path}, sheet {sheet_name}")

    @staticmethod
    def read_rows(archive, sheet_name):
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next((s for s in workbook.findall("s:sheets/s:sheet", ns)
                      if s.get("name") == sheet_name), None)
        if sheet is None:
            raise ValueError(f"Workbook has no sheet named {sheet_name}")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in relationships
                      if r.get("Id") == sheet.get(rel_ns + "id"))
        location = (target.lstrip("/") if target.startswith("/") else
                    posixpath.normpath(posixpath.join("xl", target)))
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in item.findall(".//s:t", ns))
                       for item in shared.findall("s:si", ns)]
        worksheet = ET.fromstring(archive.read(location))
        result = []
        for row in worksheet.findall("s:sheetData/s:row", ns):
            cells = {}
            for cell in row.findall("s:c", ns):
                address = cell.get("r")
                if not address:
                    raise ValueError("Workbook cell is missing its address")
                column = "".join(c for c in address if c.isalpha())
                kind = cell.get("t")
                value = cell.find("s:v", ns)
                if kind == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//s:t", ns))
                elif value is None:
                    continue
                elif kind == "s":
                    text = strings[int(value.text)]
                elif kind == "e":
                    raise ValueError(f"Spreadsheet error at {address}: {value.text}")
                else:
                    text = value.text
                cells[column] = text
            result.append((int(row.get("r")), cells))
        return result

    @staticmethod
    def make_graph(nodes, edges):
        graph = nx.Graph()
        for node in nodes:
            index, symbol = node["id"], node["symbol"]
            if type(index) is not int or index < 0 or index in graph or not isinstance(symbol, str):
                raise ValueError("Invalid or duplicate atom node")
            charge = node.get("formal_charge")
            if charge is not None and type(charge) is not int:
                raise ValueError("Formal charge must be an integer when present")
            graph.add_node(index, symbol=symbol, formal_charge=charge)
        for bond in edges:
            u, v, order = bond["source"], bond["target"], bond["order"]
            if (u not in graph or v not in graph or u == v or graph.has_edge(u, v)
                    or type(order) is not int or order not in (1, 2, 3)):
                raise ValueError("Invalid or duplicate bond")
            graph.add_edge(u, v, order=order)
        if not graph or not nx.is_connected(graph):
            raise ValueError("Species graph must be nonempty and connected")
        return graph

    @staticmethod
    def graph_key(graph):
        return tuple(sorted(Counter(d["symbol"] for _, d in graph.nodes(data=True)).items())), graph.number_of_edges()

    @staticmethod
    def same_atom(predicted, reference):
        # Stage II currently has no charge field. Missing charge means unknown,
        # never an invented neutral assignment. If supplied, it must match.
        return (predicted["symbol"] == reference["symbol"]
                and (predicted.get("formal_charge") is None
                     or predicted["formal_charge"] == reference["formal_charge"]))

    def match(self, graph):
        return [record for record in self.buckets.get(self.graph_key(graph), [])
                if nx.is_isomorphic(graph, record["graph"], node_match=self.same_atom,
                                    edge_match=nx.algorithms.isomorphism.categorical_edge_match("order", None))]


class ComparisonPlot:
    """Draw green FFCMII-only matches, red unmatched, then blue training species."""

    COLORS = {"ffcmii_only": ("#D5F2D8", "#246B35"),
              "not_in_ffcmii": ("#FADADD", "#A52632"),
              "training": ("#D9EAFE", "#245B96")}

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
        graph = WorkbookGraphs.make_graph(record["graph"]["nodes"], record["graph"]["edges"])
        matched = record["in_ffcmii_reference"]
        background, accent = self.COLORS[record["display_group"]]
        canvas = Image.new("RGB", (self.width, self.height), background)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, self.width - 1, self.height - 1), outline=accent, width=2)
        names = " / ".join(record["ffcmii_species_keys"]) if matched else "No FFCMII match"
        if record["display_group"] == "training":
            names = "Training: " + record.get("species_key", names)
        title = record["structure_id"] + "  " + names
        lines = textwrap.wrap(title, width=34)
        y = 10
        for line in lines:
            draw.text((10, y), line, fill=accent, font=self.bold)
            y += 21
        label = self.formula(graph) + "   count: " + str(record.get("occurrences", 0))
        draw.text((10, y + 2), label, fill="#253B4A", font=self.font)
        y += 29
        positions = nx.spring_layout(graph, seed=42)
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
                  f"{summary['total_structures']} structures; {summary['display_group_counts']['ffcmii_only']} green; "
                  f"{summary['display_group_counts']['not_in_ffcmii']} red; "
                  f"{summary['display_group_counts']['training']} blue.",
                  fill="#183C4C", font=self.font)
        draw.text((12, 65), "Green: FFCMII, outside training. Red: outside FFCMII. Blue: training set (last).",
                  fill="#183C4C", font=self.small)
        draw.text((12, 82), "Charge/electronic state is not inferred; multiple matching reference keys are shown together.",
                  fill="#183C4C", font=self.small)

    def save(self, records, output, summary, log):
        columns = self.config["overview_columns"]
        overview_size = (240, 214)
        header_height = 108
        count = len(records)
        overview = Image.new("RGB", (columns * overview_size[0],
                             header_height + max(1, math.ceil(count / columns)) * overview_size[1]), "white")
        self.header(overview, "Stage III: FFCMII structure comparison", summary)
        per_page = self.config["structures_per_page"]
        page_columns = self.config["page_columns"]
        pages = output / "pages"
        pages.mkdir(exist_ok=True)
        for old in pages.glob("structures_*.png"):
            old.unlink()
        page = None
        for index, record in enumerate(records):
            if index % per_page == 0:
                remaining = min(per_page, count-index)
                page = Image.new("RGB", (page_columns * self.width,
                       header_height + math.ceil(remaining / page_columns) * self.height), "white")
                self.header(page, f"Stage III: FFCMII comparison, page {index // per_page + 1}", summary)
            panel = self.panel(record)
            small = panel.resize(overview_size, Image.Resampling.LANCZOS)
            overview.paste(small, ((index % columns) * overview_size[0],
                                  header_height + (index // columns) * overview_size[1]))
            local = index % per_page
            page.paste(panel, ((local % page_columns) * self.width,
                              header_height + (local // page_columns) * self.height))
            if local == per_page-1 or index == count-1:
                filename = pages / f"structures_{index // per_page + 1:03d}.png"
                page.save(filename)
                log(f"Rendered {index+1}/{count} structures: {filename}")
                page.close()
        overview.save(output / "final_structures.png")
        overview.close()


class StageIIIComparison:
    """Compare saved Stage II structures and write a separate Stage III report."""

    def __init__(self, config):
        self.config = dict(config)

    @staticmethod
    def compact_writer(path):
        tree = ast.parse(Path(path).read_text())
        scope = {}
        nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
        return scope["CompactJSON"]

    @staticmethod
    def compare(records, reference):
        result = []
        ids = set()
        for record in records:
            if record.get("termination") != "STOP":
                raise ValueError("Stage III expects completed structures from unique_structures.json")
            identity = record["structure_id"]
            if identity in ids:
                raise ValueError(f"Duplicate structure ID: {identity}")
            ids.add(identity)
            graph = WorkbookGraphs.make_graph(record["graph"]["nodes"], record["graph"]["edges"])
            matches = reference.match(graph)
            # Stage II saved membership against the reference used for that run,
            # including training exclusions. Preserve this historical label.
            in_training = record.get("in_species_training_reference",
                                     record.get("category") == "reference_match")
            if type(in_training) is not bool:
                raise ValueError("Training-set membership must be a boolean")
            group = "training" if in_training else ("ffcmii_only" if matches else "not_in_ffcmii")
            result.append({
                **record,
                "in_training_set": in_training,
                "display_group": group,
                "in_ffcmii_reference": bool(matches),
                "ffcmii_species_keys": [r["species_key"] for r in matches],
                "ffcmii_workbook_rows": [r["workbook_row"] for r in matches],
                "ffcmii_state_notes": {r["species_key"]: r["state_notes"] for r in matches},
                "match_basis": "element_and_bond_order_isomorphism; check formal charge only when predicted",
                "electronic_state_resolved": False,
                "all_predicted_formal_charges_present": all(
                    d.get("formal_charge") is not None for _, d in graph.nodes(data=True)),
            })
        # Stable sorting retains the original order within each group.
        order = {"ffcmii_only": 0, "not_in_ffcmii": 1, "training": 2}
        return sorted(result, key=lambda r: order[r["display_group"]])

    def run(self):
        cfg = self.config
        for name in ("overview_columns", "structures_per_page", "page_columns"):
            if type(cfg[name]) is not int or cfg[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        reference = WorkbookGraphs(cfg["reference_workbook"], cfg["reference_sheet"])
        source = Path(cfg["predictions_path"])
        source_bytes = source.read_bytes()
        original = json.loads(source_bytes)
        if not isinstance(original, list):
            raise ValueError("Predictions must be Stage II's unique_structures.json list")
        records = self.compare(original, reference)
        output = Path(cfg["output_folder"])
        if output.resolve() == source.parent.resolve():
            raise ValueError("Stage III output must have its own folder, separate from Stage II inputs")
        output.mkdir(parents=True, exist_ok=True)
        writer = self.compact_writer(cfg["compact_json_source"])
        matched = [r for r in records if r["in_ffcmii_reference"]]
        keys = {key for r in matched for key in r["ffcmii_species_keys"]}
        summary = {
            "reference_entries": len(reference.records), "total_structures": len(records),
            "ffcmii_matched_structures": len(matched),
            "unmatched_structures": len(records)-len(matched),
            "display_group_counts": {group: sum(r["display_group"] == group for r in records)
                                     for group in ComparisonPlot.COLORS},
            "display_group_order": ["ffcmii_only", "not_in_ffcmii", "training"],
            "training_membership_source": "Stage II in_species_training_reference; fallback category=reference_match",
            "stageII_new_candidates_matching_ffcmii": sum(r.get("category") == "new_candidate" for r in matched),
            "reference_keys_matched_by_connectivity": sorted(keys),
            "reference_keys_not_matched": [r["species_key"] for r in reference.records if r["species_key"] not in keys],
            "match_scope": "Graph connectivity; formal charges checked only if predicted. Electronic states unresolved.",
            "plot_order": [r["structure_id"] for r in records],
        }
        provenance = {**cfg, "reference_sha256": reference.sha256,
                      "predictions_sha256": hashlib.sha256(source_bytes).hexdigest()}
        with (output / "comparison.log").open("w") as handle:
            def log(message):
                line = time.strftime("%H:%M:%S") + " | " + message
                print(line, flush=True)
                handle.write(line + "\n")
                handle.flush()
            log(f"Read {len(reference.records)} FFCMII reference entries and {len(records)} generated structures.")
            log(f"FFCMII matches={len(matched)}; unmatched={len(records)-len(matched)}; "
                f"Stage II new candidates now matched={summary['stageII_new_candidates_matching_ffcmii']}")
            log(f"Plot groups (green, red, blue): {summary['display_group_counts']}")
            writer.write(output / "config.json", provenance)
            writer.write(output / "summary.json", summary)
            writer.write(output / "compared_structures.json", records)
            writer.write(output / "ffcmii_matches.json", matched)
            with (output / "comparison.csv").open("w", newline="") as csv_file:
                table = csv.writer(csv_file)
                table.writerow(["plot_order", "structure_id", "formula", "in_ffcmii_reference",
                                "ffcmii_species_keys", "stageII_category", "occurrences", "state_resolved",
                                "in_training_set", "display_group"])
                for i, record in enumerate(records, 1):
                    graph = WorkbookGraphs.make_graph(record["graph"]["nodes"], record["graph"]["edges"])
                    table.writerow([i, record["structure_id"], ComparisonPlot.formula(graph),
                                    record["in_ffcmii_reference"], "; ".join(record["ffcmii_species_keys"]),
                                    record.get("category", ""), record.get("occurrences", 0), False,
                                    record["in_training_set"], record["display_group"]])
            ComparisonPlot(cfg).save(records, output, summary, log)
            log(f"FINISHED: {output / 'final_structures.png'}")
        return summary


# ==========================================
# PARAMETERS — paths relative to the working directory
# ==========================================
PREDICTIONS_PATH = "../stageII/output/unique_structures.json"
REFERENCE_WORKBOOK = "../FFCMII_species_reference.xlsx"
REFERENCE_SHEET = "Graph definitions"
COMPACT_JSON_SOURCE = "../stageI/compact_json.py"
OUTPUT_FOLDER = "output"
# The overview contains every structure; pages provide larger, readable panels.
OVERVIEW_COLUMNS = 10
STRUCTURES_PER_PAGE = 50
PAGE_COLUMNS = 5

# ==========================================
# RUN COMPARISON — direct calls, no main guard
# ==========================================
comparison_config = {
    "predictions_path": PREDICTIONS_PATH, "reference_workbook": REFERENCE_WORKBOOK,
    "reference_sheet": REFERENCE_SHEET, "compact_json_source": COMPACT_JSON_SOURCE,
    "output_folder": OUTPUT_FOLDER, "overview_columns": OVERVIEW_COLUMNS,
    "structures_per_page": STRUCTURES_PER_PAGE, "page_columns": PAGE_COLUMNS,
}
comparison = StageIIIComparison(comparison_config)
comparison.run()
