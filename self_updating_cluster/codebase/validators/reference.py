"""Reference graph loading and matching; no workbook mutation."""
import hashlib
import io
import json
import posixpath
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import networkx as nx
from .base import Validator, ValidationResult
from ..molecule import Molecule, SpeciesInput

class GraphReferences:
    """Shared graph validation and matching, independent of the source file format."""

    make_graph = staticmethod(Molecule.make_graph)

    def __init__(self, records):
        self.records = []
        self.buckets = {}
        for name, record in records.items():
            graph = Molecule.graph(record)
            entry = {"species_key": name, "graph": graph,
                     "chemical_name": record.get("chemical_name", ""),
                     "state_notes": record.get("state_notes", "Graph reference; electronic state unresolved.")}
            self.records.append(entry)
            self.buckets.setdefault(self.graph_key(graph), []).append(entry)

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


class WorkbookGraphs(GraphReferences):
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
        required = {"Species key", "All node symbols JSON", "All bonds JSON", "Base atoms JSON"}
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
            charges = (json.loads(row[header["Base formal charges JSON"]])
                       if "Base formal charges JSON" in header else [None] * len(base))
            if len(base) != len(charges) or symbols[:len(base)] != base:
                raise ValueError(f"{name}: inconsistent base atoms/formal charges")
            if any(charge is not None and type(charge) is not int for charge in charges):
                raise ValueError(f"{name}: formal charges must be integers")
            if any(symbol != "H" for symbol in symbols[len(base):]):
                raise ValueError(f"{name}: appended atoms must be explicit H")
            charges += [0 if "Base formal charges JSON" in header else None] * (len(symbols) - len(base))
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



class ReferenceValidator(Validator):
    """Match a CSV, workbook, or JSON graph dictionary; absence remains unknown.

    Any unmatched-as-negative policy belongs to DatasetManager, not this class.
    Match scope preserves the previous Stage III element/bond-order rules.
    """
    name='reference'
    version=1

    def __init__(self, reference_path, sheet_name='Graph definitions'):
        self.path=Path(reference_path)
        self.sheet_name=sheet_name
        self.source_hash=hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.path.suffix.lower()=='.xlsx':
            self.reference=WorkbookGraphs(self.path,sheet_name)
        elif self.path.suffix.lower() in ('.csv', '.json'):
            self.reference=GraphReferences(SpeciesInput.read(self.path))
        else:
            raise ValueError('Reference must be a .csv, .xlsx, or .json graph dictionary')

    def normalized_records(self):
        """Return portable graph records for the run-local JSON input snapshot."""
        output={}
        for item in self.reference.records:
            graph=item['graph']
            output[item['species_key']]={
                'nodes':[{'id':int(index),'symbol':data['symbol'],
                          'formal_charge':data.get('formal_charge')}
                         for index,data in sorted(graph.nodes(data=True))],
                'edges':[{'source':int(u),'target':int(v),'order':data['order']}
                         for u,v,data in sorted(graph.edges(data=True))],
                'chemical_name':item.get('chemical_name',''),
                'state_notes':item.get('state_notes',''),
            }
        return output

    def configuration(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.source_hash:
            raise RuntimeError('Reference file changed; create a new ReferenceValidator')
        return {'reference_path':str(self.path.resolve()),'reference_sha256':self.source_hash,
                'sheet_name':self.sheet_name,'match_rule':'elements_and_bond_orders; charges_if_predicted'}

    def cache_payload(self, record):
        return Molecule.identity_payload(record['graph'])

    def submit(self, record, workdir):
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.source_hash:
            raise RuntimeError('Reference file changed; create a new ReferenceValidator')
        graph=Molecule.graph(record['graph'])
        # The imported graph references cannot establish a supplied spin state.
        if record['graph'].get('spin_multiplicity') is not None or record['graph'].get('total_charge') is not None:
            return ValidationResult(record['structure_id'],'completed','unknown',
                'This reference matcher does not resolve total-charge/spin metadata',self.name,
                {'unmatched_reference':False,'electronic_state_resolved':False})
        matches=self.reference.match(graph)
        return ValidationResult(record['structure_id'],'completed','accepted' if matches else 'unknown',
            'Reference graph match' if matches else 'No matching graph in the supplied reference',self.name,
            {'matched_species':[m['species_key'] for m in matches],
             'unmatched_reference':not bool(matches),'electronic_state_resolved':False,
             'state_notes':{m['species_key']:m['state_notes'] for m in matches}})
