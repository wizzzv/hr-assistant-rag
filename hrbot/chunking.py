from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str
    text: str

    @property
    def source(self):
        return f"{self.doc_title} / {self.section}"


def split_document(path):
    """Cut one markdown file into chunks, one chunk per '## section'."""
    doc_title = path.stem
    sections = []  # [heading, [lines]]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sections.append([line[3:].strip(), []])
        elif line.startswith("# "):
            doc_title = line[2:].strip()
        elif sections:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            chunks.append(Chunk(path.stem, doc_title, heading, body))
    return chunks


def load_chunks(folder):
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        chunks.extend(split_document(path))
    return chunks
