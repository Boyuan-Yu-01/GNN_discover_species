"""Readable JSON: pack short fields by default; use pack=False for expanded objects."""
import json
from pathlib import Path


class CompactJSON:
    """Preserve normal JSON data while avoiding one line per scalar field."""

    @staticmethod
    def dumps(data, width=120, pack=True):
        return CompactJSON._format(data, 0, width, pack) + "\n"

    @staticmethod
    def _format(value, level, width, pack=False):
        # json.dumps handles quotes, backslashes, Unicode and JSON booleans/null.
        inline = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if not isinstance(value, (dict, list, tuple)) or not value:
            return inline
        # Optional denser layout: short nested containers can share a line.
        if pack and len(inline) + 2 * level <= width:
            return inline
        children = value.values() if isinstance(value, dict) else value
        flat = all(not isinstance(child, (dict, list, tuple)) for child in children)
        sequence = isinstance(value, (list, tuple))
        # Numeric vectors (including long run/index lists) never become one
        # number per line. The width is a readability hint, not a hard limit.
        if sequence and flat and all(not isinstance(child, str) for child in value):
            return inline
        # Keep short bond tables such as [[0, 1, 1], [1, 2, 2]] together too.
        scalar_rows = sequence and all(
            isinstance(child, (list, tuple))
            and all(not isinstance(item, (dict, list, tuple)) for item in child)
            for child in value)
        if scalar_rows and len(inline) + 2 * level <= width:
            return inline
        if flat and len(inline) + 2 * level <= width:
            return inline
        indent = "  " * level
        child_indent = "  " * (level + 1)
        # Pack long lists of species names/IDs into lines instead of giving
        # every short string a separate line.
        if sequence and flat:
            lines = []
            line = child_indent
            for index, child in enumerate(value):
                token = json.dumps(child, ensure_ascii=False, allow_nan=False)
                if index < len(value) - 1:
                    token += ","
                separator = " " if line != child_indent else ""
                if line != child_indent and len(line) + len(separator) + len(token) > width:
                    lines.append(line)
                    line = child_indent
                    separator = ""
                line += separator + token
            lines.append(line)
            return "[\n" + "\n".join(lines) + "\n" + indent + "]"
        if isinstance(value, dict):
            lines = []
            for key, child in value.items():
                # Let the standard encoder convert numeric keys just as json.dump does.
                encoded_key = json.dumps({key: None}, ensure_ascii=False).rsplit(": ", 1)[0][1:]
                lines.append(child_indent + encoded_key + ": "
                             + CompactJSON._format(child, level + 1, width, pack))
            if pack:
                return "{\n" + CompactJSON._pack_lines(lines, child_indent, width) + "\n" + indent + "}"
            return "{\n" + ",\n".join(lines) + "\n" + indent + "}"
        lines = [child_indent + CompactJSON._format(child, level + 1, width, pack)
                 for child in value]
        if pack:
            return "[\n" + CompactJSON._pack_lines(lines, child_indent, width) + "\n" + indent + "]"
        return "[\n" + ",\n".join(lines) + "\n" + indent + "]"

    @staticmethod
    def _pack_lines(items, indent, width):
        """Share lines for short fields/atoms/bonds; keep multiline records separate."""
        lines = []
        pending = ""
        for index, item in enumerate(items):
            token = item + ("," if index < len(items) - 1 else "")
            if "\n" in token:
                if pending:
                    lines.append(pending)
                    pending = ""
                lines.append(token)
            elif pending and len(pending) + 1 + len(token) - len(indent) <= width:
                pending += " " + token[len(indent):]
            else:
                if pending:
                    lines.append(pending)
                pending = token
        if pending:
            lines.append(pending)
        return "\n".join(lines)

    @staticmethod
    def write(path, data, width=120, pack=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CompactJSON.dumps(data, width, pack), encoding="utf-8")
        return path
