import _io
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


def _default_handler(el: etree._Element, out: _io.TextIOWrapper, newline: bool = False) -> None:
    if el.text and el.text.strip():
        out.write(el.text.strip() + ("\n" if newline else " "))


def _default_tail_handler(el: etree._Element, out: _io.TextIOWrapper, newline: bool = False) -> None:
    if el.tail and el.tail.strip():
        out.write(el.tail.strip() + ("\n" if newline else " "))


def _math_handler(el: etree._Element, out: _io.TextIOWrapper) -> None:
    string = etree.tostring(el, encoding="unicode", with_tail=False)
    latex = _M2L.convert(string).strip()
    out.write("\\( " + latex + " \\)")


def _equation_handler(el: etree._Element, out: _io.TextIOWrapper) -> None:
    string = etree.tostring(el, encoding="unicode", with_tail=False)
    latex = _M2L.convert(string).strip()
    out.write("\n\\[ " + latex + " \\]\n")


def _walk_subtree(el: etree._Element, out: _io.TextIOWrapper, parse_cfg: dict[str, object]) -> None:
    tag = _localname(el.tag)

    if tag not in parse_cfg["ignore_content_tags"]:
        match tag:
            case "math":
                _math_handler(el, out)
                return
            case "equation":
                _equation_handler(el, out)
                return

        _default_handler(el, out, newline=(tag not in parse_cfg["no_newline_tags"]))

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
