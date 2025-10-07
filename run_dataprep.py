from src.data.openstax import clone_repo, raw_to_xml, xml_to_txt
from src.utils.helpers import load_yaml


def openstax_dataprep(cfg: dict):
    clone_repo(repo_url=cfg["git_repo"], raw_data_dir=cfg["raw_data_path"])
    raw_to_xml(raw_data_dir=cfg["raw_data_path"], output_dir=cfg["extracted_xml_dir"])
    xml_to_txt(
        xml_file="data/extracted/openstax/university-physics-volume-1.collection.xml",
        parse_cfg=cfg["parse_cfg"],
        output_dir=cfg["processed_txt_dir"],
    )


if __name__ == "__main__":
    cfg = load_yaml("config/data.yaml")
    openstax_dataprep(cfg=cfg["openstax"])
