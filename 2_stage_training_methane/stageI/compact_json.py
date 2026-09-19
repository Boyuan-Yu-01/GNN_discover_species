"""Readable JSON: short lists and individual atom/bond records stay on one line."""
import json
from pathlib import Path


class CompactJSON:
    """Preserve normal JSON data while avoiding one line per scalar field."""

    @staticmethod
    def dumps(data, width=120):
        return CompactJSON._format(data, 0, width) + "\n"

    @staticmethod
    def _format(value, level, width):
        # json.dumps handles quotes, backslashes, Unicode and JSON booleans/null.
        inline = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if not isinstance(value, (dict, list, tuple)) or not value:
            return inline
        children = value.values() if isinstance(value, dict) else value
        flat = all(not isinstance(child, (dict, list, tuple)) for child in children)
        if flat and len(inline) + 2 * level <= width:
            return inline
        indent = "  " * level
        child_indent = "  " * (level + 1)
        if isinstance(value, dict):
            lines = []
            for key, child in value.items():
                # Let the standard encoder convert numeric keys just as json.dump does.
                encoded_key = json.dumps({key: None}, ensure_ascii=False).rsplit(": ", 1)[0][1:]
                lines.append(child_indent + encoded_key + ": "
                             + CompactJSON._format(child, level + 1, width))
            return "{\n" + ",\n".join(lines) + "\n" + indent + "}"
        lines = [child_indent + CompactJSON._format(child, level + 1, width)
                 for child in value]
        return "[\n" + ",\n".join(lines) + "\n" + indent + "]"

    @staticmethod
    def write(path, data, width=120):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CompactJSON.dumps(data, width), encoding="utf-8")
        return path
