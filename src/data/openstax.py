import _io
from io import StringIO
import os
from lxml import etree
from git import Repo
from copy import deepcopy
from mathml_to_latex.converter import MathMLToLaTeX

_M2L = MathMLToLaTeX()
_MATHML_NS = "http://www.w3.org/1998/Math/MathML"


def clone_repo(repo_url: str, raw_data_dir: str) -> None:
    if not os.path.isdir(raw_data_dir):
        os.makedirs(raw_data_dir)
        Repo.clone_from(url=repo_url, to_path=raw_data_dir)


def _strip_mathml_ns_inplace(el: etree._Element) -> None:
    for n in el.xpath(".//m:*", namespaces=dict(m=_MATHML_NS)):
        q = etree.QName(n.tag)
        n.tag = q.localname
        if n.attrib:
            for k in list(n.attrib):
                qk = etree.QName(k)
                if qk.namespace == _MATHML_NS:
                    n.attrib[qk.localname] = n.attrib.pop(k)


def _normalize_module_for_mathml(m_root: etree._Element) -> etree._Element:
    mod_clean = deepcopy(m_root)
    _strip_mathml_ns_inplace(mod_clean)
    etree.cleanup_namespaces(mod_clean)
    return mod_clean


def _collection_extract(collection_path: str, modules_path: str, output_dir: str) -> None:
    tree = etree.parse(collection_path)
    root = tree.getroot()

    ns = {p if p else "col": uri for p, uri in root.nsmap.items()}
    module_elems = tree.xpath("//col:module[@document]", namespaces=ns)

    for mod in module_elems:
        mid = mod.get("document")

        m_tree = etree.parse(os.path.join(modules_path, mid, "index.cnxml"))
        m_root = m_tree.getroot()

        m_root_clean = _normalize_module_for_mathml(m_root)

        mod.append(m_root_clean)

    collection_name = os.path.basename(collection_path)
    tree.write(os.path.join(output_dir, collection_name), encoding="utf-8", xml_declaration=True)


def raw_to_xml(raw_data_dir: str, output_dir: str) -> None:
    collections_dir = os.path.join(raw_data_dir, "collections")
    modules_dir = os.path.join(raw_data_dir, "modules")

    collections = [
        os.path.join(collections_dir, f)
        for f in os.listdir(collections_dir)
        if f.endswith(".xml")
    ]

    os.makedirs(output_dir, exist_ok=True)

    for collection in collections:
        _collection_extract(collection_path=collection, modules_path=modules_dir, output_dir=output_dir)


def _localname(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _ancestor_count(el: etree._Element, local: str) -> int:
    return sum(1 for a in el.iterancestors() if _localname(a.tag) == local)


def _render_inline(el: etree._Element, parse_cfg: dict[str, object]) -> str:
    """Render inline content inside a <title> using existing handlers."""
    buf = StringIO()
    # write any leading text
    if el.text and el.text.strip():
        buf.write(el.text.strip() + " ")
    # walk children (emphasis, math, sub/sup, etc.)
    for ch in el:
        _walk_subtree(ch, buf, parse_cfg=parse_cfg)
    return " ".join(buf.getvalue().split())  # collapse whitespace


def _default_handler(el: etree._Element, out: _io.TextIOWrapper, newline: bool = False) -> None:
    if el.text and el.text.strip():
        out.write(el.text.strip() + ("\n" if newline else " "))


def _default_tail_handler(el: etree._Element, out: _io.TextIOWrapper, newline: bool = False) -> None:
    if el.tail and el.tail.strip():
        out.write(el.tail.strip() + ("\n" if newline else " "))


def _math_handler(el: etree._Element, out: _io.TextIOWrapper) -> None:
    string = etree.tostring(el, encoding="unicode", with_tail=False)
    latex = _M2L.convert(string).strip()
    out.write("\\( " + latex + " \\) ")


def _equation_handler(el: etree._Element, out: _io.TextIOWrapper) -> None:
    string = etree.tostring(el, encoding="unicode", with_tail=False)
    latex = _M2L.convert(string).strip()
    out.write("\n\\[ " + latex + " \\]\n")


def _media_handler(el, out):
    alt = (el.get("alt") or "").strip()
    out.write("\n" + (f"[IMAGE: {alt}]" if alt else "[IMAGE]") + "\n")
    if el.tail and el.tail.strip():
        out.write(el.tail.strip() + " ")


def _sub_handler(el: etree._Element, out: _io.TextIOWrapper) -> None:
    txt = ("".join(el.itertext()) or "").strip()
    out.write(f"_{{{txt}}}")
    if el.tail and el.tail.strip():
        out.write(el.tail.strip() + " ")

def _sup_handler(el: etree._Element, out: _io.TextIOWrapper) -> None:
    txt = ("".join(el.itertext()) or "").strip()
    out.write(f"^{{{txt}}}")
    if el.tail and el.tail.strip():
        out.write(el.tail.strip() + " ")

def _emphasis_handler(el, out):
    effect = (el.get("effect") or "").strip().lower()
    if effect == "italics":
        cmd = r"\textit"
    elif effect == "bold":
        cmd = r"\textbf"
    else:
        raise ValueError(f"Unsupported <emphasis> effect: {effect}")
    txt = ("".join(el.itertext()) or "").strip()
    out.write(f"{cmd}{{{txt}}}")
    if el.tail and el.tail.strip():
        tail = el.tail.strip()
        if tail and tail[0].isalnum():
            out.write(" ")
        out.write(tail + " ")


def _item_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    # start a new bullet on its own line
    out.write("\n- ")
    # item text
    if el.text and el.text.strip():
        out.write(el.text.strip() + " ")
    # keep nested markup (math, emphasis, sub/sup, etc.)
    for child in el:
        _walk_subtree(child, out, parse_cfg=parse_cfg)
    # write tail right away so we don't lose punctuation/spacing
    if el.tail and el.tail.strip():
        out.write(el.tail.strip() + " ")


def _list_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    # blank line before the list block
    for child in el:
        _walk_subtree(child, out, parse_cfg=parse_cfg)
    # newline after the list block
    out.write("\n")
    # keep tail text (e.g., paragraph continues)
    if el.tail and el.tail.strip():
        out.write(el.tail.strip() + " ")


def _title_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    parent_local = _localname(el.getparent().tag) if el.getparent() is not None else None
    sec_depth = _ancestor_count(el, "section")
    sc_depth  = _ancestor_count(el, "subcollection")  # Unit=1, Chapter=2 (in your books)

    text = _render_inline(el, parse_cfg)
    if not text:
        return

    # decide LaTeX-y command based on real hierarchy
    if parent_local == "subcollection":
        # md:title inside a subcollection
        if sc_depth == 1:                 # Unit
            cmd = r"\part*"
            text = f"Unit: {text}"
        elif sc_depth == 2:               # Chapter
            cmd = r"\chapter*"
        else:
            cmd = r"\section*"            # deeper subcollections (unlikely)
    elif parent_local == "document":
        # Module title; if no chapter context (e.g., Preface at top), promote to chapter
        cmd = r"\section*" if sc_depth >= 1 else r"\chapter*"
    elif parent_local == "section":
        # Section headings inside a module, by depth
        if   sec_depth == 1: cmd = r"\subsection*"
        elif sec_depth == 2: cmd = r"\subsubsection*"
        else:                cmd = r"\paragraph*"
    else:
        # Fallback (rare)
        cmd = r"\section*"

    out.write(f"\n\n{cmd}{{{text}}}\n\n")


def _footnote_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    out.write("(" + "".join(el.itertext()).strip() + ")")


def _glossary_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    out.write("\n\nGlossary:\n")
    for def_el in el:
        if _localname(def_el.tag) != "definition":
            continue
        term = ""
        meaning = ""
        for child in def_el:
            ln = _localname(child.tag)
            if ln == "term":
                term = "".join(child.itertext()).strip()
            elif ln == "meaning":
                meaning = "".join(child.itertext()).strip()
        if term or meaning:
            out.write(f"{term} - {meaning}\n")


def _newline_handler(el: etree._Element, out: _io.TextIOWrapper) -> None:
    out.write("\n")


def _para_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    # blank line before paragraph
    out.write("\n")
    # paragraph text (before children)
    if el.text and el.text.strip():
        out.write(el.text.strip() + " ")
    # inline content inside the paragraph
    for child in el:
        _walk_subtree(child, out, parse_cfg=parse_cfg)
    # newline after paragraph
    out.write("\n")


def _exercise_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    # blank line before the exercise block
    out.write("\n")

    for child in el:
        ln = _localname(child.tag)
        if ln == "problem":
            out.write("Problem:\n")
            if child.text and child.text.strip():
                out.write(child.text.strip() + " ")
            for ch in child:
                _walk_subtree(ch, out, parse_cfg=parse_cfg)
            out.write("\n")  # end of problem
        elif ln == "solution":
            out.write("Solution:\n")
            if child.text and child.text.strip():
                out.write(child.text.strip() + " ")
            for ch in child:
                _walk_subtree(ch, out, parse_cfg=parse_cfg)
            out.write("\n")  # end of solution

    # optional blank line after the exercise block (keeps things airy)
    out.write("\n")


def _example_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    out.write("\nExample:\n")
    if el.text and el.text.strip():
        out.write(el.text.strip() + " ")
    for ch in el:
        _walk_subtree(ch, out, parse_cfg=parse_cfg)
    out.write("\n")


def _table_handler(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    summary = (el.get("summary") or "").strip()
    if not summary:
        cap = el.find(".//caption")
        if cap is not None:
            summary = " ".join(("".join(cap.itertext())).split())
    label = f"[TABLE: {summary}]" if summary else "[TABLE]"
    out.write("\n" + label + "\n")


def _walk_subtree(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    tag = _localname(el.tag)

    if tag in parse_cfg.get("skip_tags", set()):
        _default_tail_handler(el, out, newline=False)
        return

    if tag not in parse_cfg["ignore_content_tags"]:
        match tag:
            case "math":
                _math_handler(el, out)
                _default_tail_handler(el, out, newline=False)
                return
            case "equation":
                _equation_handler(el, out)
                _default_tail_handler(el, out, newline=False)
                return
            case "media":
                _media_handler(el, out)
                return
            case "sub":
                _sub_handler(el, out)
                return
            case "sup":
                _sup_handler(el, out)
                return
            case "emphasis":
                _emphasis_handler(el, out)
                return
            case "item":
                _item_handler(el, out, parse_cfg)
                return
            case "list":
                _list_handler(el, out, parse_cfg)
                return
            case "title":
                _title_handler(el, out, parse_cfg)
                _default_tail_handler(el, out, newline=False)
                return
            case "footnote":
                _footnote_handler(el, out, parse_cfg)
                _default_tail_handler(el, out, newline=False)
                return
            case "glossary":
                _glossary_handler(el, out, parse_cfg)
                _default_tail_handler(el, out, newline=False)
                return
            case "newline":
                _newline_handler(el, out)
                _default_tail_handler(el, out, newline=False)
                return
            case "para":
                _para_handler(el, out, parse_cfg)
                _default_tail_handler(el, out, newline=False)
                return
            case "exercise":
                _exercise_handler(el, out, parse_cfg)
                _default_tail_handler(el, out, newline=False)
                return
            case "example":
                _example_handler(el, out, parse_cfg)
                _default_tail_handler(el, out, newline=False)
                return
            case "table":
                _table_handler(el, out, parse_cfg)
                _default_tail_handler(el, out, newline=False)
                return
        _default_handler(el, out, newline=(tag not in parse_cfg["no_newline_tags"]))

    # else:
    #     print(f"*{_localname(el.tag)}*", el.text)

    for child in el:
        _walk_subtree(child, out, parse_cfg=parse_cfg)

    if tag not in parse_cfg["ignore_content_tags"]:
        _default_tail_handler(el, out, newline=(tag not in parse_cfg["no_newline_tags"]))


def xml_to_txt(xml_file: str, parse_cfg: dict[str, object], output_dir: str) -> None:
    tree = etree.parse(xml_file)
    root = tree.getroot()
    ns = {p if p else "col": uri for p, uri in root.nsmap.items()}

    subs = tree.xpath("//col:subcollection", namespaces=ns)

    os.makedirs(output_dir, exist_ok=True)
    file_base = os.path.splitext(os.path.basename(xml_file))[0]
    save_path = os.path.join(output_dir, file_base + ".txt")

    with open(save_path, "w", encoding="utf-8") as file:
        for sc in subs:
            title = sc.xpath("string(md:title)", namespaces=ns).strip()
            if title in parse_cfg["parse_topics"]:
                _walk_subtree(el=sc, out=file, parse_cfg=parse_cfg)
