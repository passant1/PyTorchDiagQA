"""
CSV 导出模块
将知识图谱导出为 nodes.csv 和 edges.csv
"""
import csv
import json
import os

from utils.path_utils import get_data_path


def export_to_csv(kg_path: str = None, output_dir: str = None):
    """
    将知识图谱导出为 CSV 文件

    Args:
        kg_path: 知识图谱 JSON 文件路径
        output_dir: 输出目录
    """
    if kg_path is None:
        kg_path = get_data_path("pytorch_kg.json")
    if output_dir is None:
        output_dir = os.path.dirname(kg_path)

    if not os.path.exists(kg_path):
        print(f"[CSV Export] 知识图谱文件不存在: {kg_path}")
        return

    with open(kg_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # 导出 nodes.csv
    nodes_path = os.path.join(output_dir, "nodes.csv")
    node_fields = ["id", "name", "label", "description", "aliases", "url", "command", "code"]

    with open(nodes_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields, extrasaction='ignore')
        writer.writeheader()
        for node in nodes:
            row = {
                "id": node.get("id", ""),
                "name": node.get("name", ""),
                "label": node.get("label", ""),
                "description": node.get("description", ""),
                "aliases": "|".join(node.get("aliases", [])),
                "url": node.get("url", ""),
                "command": node.get("command", ""),
                "code": node.get("code", ""),
            }
            writer.writerow(row)

    print(f"[CSV Export] 节点已导出: {nodes_path} ({len(nodes)} 行)")

    # 导出 edges.csv
    edges_path = os.path.join(output_dir, "edges.csv")
    edge_fields = ["source", "relation", "target", "evidence", "source_doc"]

    with open(edges_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=edge_fields, extrasaction='ignore')
        writer.writeheader()
        for edge in edges:
            row = {
                "source": edge.get("source", ""),
                "relation": edge.get("relation", ""),
                "target": edge.get("target", ""),
                "evidence": edge.get("evidence", ""),
                "source_doc": edge.get("source_doc", ""),
            }
            writer.writerow(row)

    print(f"[CSV Export] 边已导出: {edges_path} ({len(edges)} 行)")

    return nodes_path, edges_path


if __name__ == "__main__":
    # 直接运行此脚本即可导出 CSV
    export_to_csv()
