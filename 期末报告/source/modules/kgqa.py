"""
知识图谱问答模块
基于 PyTorch 错误诊断知识图谱进行结构化查询
继承 KBQA.py 的思想：意图识别 → 图谱查询 → 结构化答案
"""
import json
import os

from utils.path_utils import get_data_path


class KnowledgeGraphQA:
    """
    知识图谱问答引擎
    支持根据实体、关系类型、意图进行图谱查询
    """

    def __init__(self, kg_path: str = None):
        """
        初始化知识图谱问答引擎

        Args:
            kg_path: 知识图谱 JSON 文件路径
        """
        if kg_path is None:
            kg_path = get_data_path("pytorch_kg.json")

        self.kg_path = kg_path
        self.graph = self._load_graph()

        # 构建索引
        self._build_index()

    def _load_graph(self) -> dict:
        """加载知识图谱"""
        if not os.path.exists(self.kg_path):
            print(f"[KGQA] 知识图谱文件不存在: {self.kg_path}")
            return {"nodes": [], "edges": []}

        with open(self.kg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self):
        """构建节点索引，加快查询速度"""
        self.node_by_id = {}
        self.node_by_name = {}
        self.node_by_label = {}

        for node in self.graph.get("nodes", []):
            nid = node.get("id", "")
            nname = node.get("name", "")
            nlabel = node.get("label", "")

            if nid:
                self.node_by_id[nid] = node
            if nname:
                # 支持别名查询
                self.node_by_name[nname.lower()] = node
                for alias in node.get("aliases", []):
                    self.node_by_name[alias.lower()] = node
            if nlabel:
                if nlabel not in self.node_by_label:
                    self.node_by_label[nlabel] = []
                self.node_by_label[nlabel].append(node)

        # 构建邻接表
        self.adj_out = {}  # source -> [(target, relation, edge_props)]
        self.adj_in = {}   # target -> [(source, relation, edge_props)]

        for edge in self.graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            relation = edge.get("relation", "")

            if source not in self.adj_out:
                self.adj_out[source] = []
            self.adj_out[source].append((target, relation, edge))

            if target not in self.adj_in:
                self.adj_in[target] = []
            self.adj_in[target].append((source, relation, edge))

    def get_node(self, identifier: str) -> dict:
        """
        根据标识符获取节点

        Args:
            identifier: 节点 ID、名称或别名

        Returns:
            dict or None: 节点信息
        """
        # 先按 ID 查找
        if identifier in self.node_by_id:
            return self.node_by_id[identifier]
        # 按名称查找（大小写不敏感）
        if identifier.lower() in self.node_by_name:
            return self.node_by_name[identifier.lower()]
        return None

    def get_out_edges(self, node_id: str) -> list:
        """获取节点的所有出边"""
        return self.adj_out.get(node_id, [])

    def get_in_edges(self, node_id: str) -> list:
        """获取节点的所有入边"""
        return self.adj_in.get(node_id, [])

    def get_related_by_relation(self, node_id: str, relation: str) -> list:
        """
        按关系类型查询相关节点

        Args:
            node_id: 源节点 ID
            relation: 关系类型

        Returns:
            list of dict: 相关节点列表
        """
        results = []
        for target, rel, edge in self.get_out_edges(node_id):
            if rel == relation:
                target_node = self.get_node(target)
                if target_node:
                    results.append({
                        "node": target_node,
                        "relation": rel,
                        "evidence": edge.get("evidence", ""),
                        "source_doc": edge.get("source_doc", ""),
                    })
        return results

    def query(self, entity: str, intent: str = "QUERY_ALL") -> dict:
        """
        根据实体和意图查询知识图谱

        Args:
            entity: 实体名称（问题类型、API、报错等）
            intent: 意图类型

        Returns:
            dict: 结构化查询结果
        """
        node = self.get_node(entity)
        if node is None:
            return {"found": False, "message": f"知识图谱中未找到实体: {entity}"}

        result = {
            "found": True,
            "entity": node,
            "intent": intent,
            "apis": [],
            "causes": [],
            "solutions": [],
            "commands": [],
            "concepts": [],
            "errors": [],
            "examples": [],
            "doc_pages": [],
            "related": [],
        }

        node_id = node.get("id", "")

        # 查询所有出边，按关系类型分类
        for target, relation, edge in self.get_out_edges(node_id):
            target_node = self.get_node(target)
            if target_node is None:
                continue

            entry = {
                "node": target_node,
                "evidence": edge.get("evidence", ""),
                "source_doc": edge.get("source_doc", ""),
            }

            if relation == "HAS_API":
                result["apis"].append(entry)
            elif relation == "HAS_CAUSE":
                result["causes"].append(entry)
            elif relation == "HAS_SOLUTION":
                result["solutions"].append(entry)
            elif relation == "CHECK_BY":
                result["commands"].append(entry)
            elif relation == "HAS_EXAMPLE":
                result["examples"].append(entry)
            elif relation == "MENTIONED_IN":
                result["doc_pages"].append(entry)
            elif relation == "RELATED_TO":
                result["related"].append(entry)
            elif relation == "HAS_PARAMETER":
                result["concepts"].append(entry)
            elif relation == "HAS_ERROR":
                result["errors"].append(entry)

        # 根据意图过滤结果
        if intent == "QUERY_CAUSE":
            result["primary"] = result["causes"]
        elif intent == "QUERY_SOLUTION":
            result["primary"] = result["solutions"]
        elif intent == "QUERY_COMMAND":
            result["primary"] = result["commands"]
        elif intent == "QUERY_EXPLAIN":
            result["primary"] = [{"node": node}] + result["concepts"]

        return result

    def format_answer(self, query_result: dict, use_llm: bool = False) -> str:
        """
        将图谱查询结果格式化为结构化答案

        Args:
            query_result: query() 的返回结果
            use_llm: 是否使用大模型润色（由调用方处理）

        Returns:
            str: 格式化的答案文本
        """
        if not query_result.get("found"):
            return f"[知识图谱] {query_result.get('message', '未找到相关信息')}"

        entity = query_result["entity"]
        intent = query_result.get("intent", "QUERY_ALL")

        lines = []

        # 问题类型
        entity_label = entity.get("label", "")
        entity_name = entity.get("name", "")
        entity_desc = entity.get("description", "")

        if entity_label == "Problem":
            lines.append(f"【问题类型】\n{entity_name}")
        elif entity_label == "Error":
            lines.append(f"【错误信息】\n{entity_name}")
        elif entity_label == "API":
            lines.append(f"【相关 API】\n{entity_name}")
        elif entity_label == "Concept":
            lines.append(f"【概念说明】\n{entity_name}")

        # 描述
        if entity_desc:
            lines.append(f"\n【问题说明】\n{entity_desc}")

        # 相关 API
        apis = query_result.get("apis", [])
        if apis:
            lines.append("\n【相关 API】")
            for a in apis:
                api_node = a["node"]
                api_desc = api_node.get("description", "")
                lines.append(f"  - {api_node['name']}")
                if api_desc:
                    lines.append(f"    {api_desc}")

        # 原因
        causes = query_result.get("causes", [])
        if causes:
            lines.append("\n【可能原因】")
            for i, c in enumerate(causes):
                c_node = c["node"]
                lines.append(f"  {i + 1}. {c_node.get('description', c_node['name'])}")
                if c.get("source_doc"):
                    lines.append(f"     [来源: {c['source_doc']}]")

        # 解决方法
        solutions = query_result.get("solutions", [])
        if solutions:
            lines.append("\n【解决建议】")
            for i, s in enumerate(solutions):
                s_node = s["node"]
                lines.append(f"  {i + 1}. {s_node.get('description', s_node['name'])}")
                if s.get("source_doc"):
                    lines.append(f"     [来源: {s['source_doc']}]")

        # 检查命令
        commands = query_result.get("commands", [])
        if commands:
            lines.append("\n【检查命令】")
            for c in commands:
                lines.append(f"  $ {c['node'].get('command', c['node']['name'])}")
                if c["node"].get("description"):
                    lines.append(f"    {c['node']['description']}")

        # 参数 / 概念说明
        concepts = query_result.get("concepts", [])
        if concepts:
            lines.append("\n【参数与概念】")
            for c in concepts:
                c_node = c["node"]
                desc = c_node.get("description", "")
                lines.append(f"  - {c_node['name']}: {desc}")
                if c.get("source_doc"):
                    lines.append(f"    [来源: {c['source_doc']}]")

        # 相关知识
        related = query_result.get("related", [])
        if related:
            lines.append("\n【相关信息】")
            for r in related:
                r_node = r["node"]
                lines.append(f"  - {r_node['name']}: {r_node.get('description', '')}")

        # 代码示例
        examples = query_result.get("examples", [])
        if examples:
            lines.append("\n【代码示例】")
            for e in examples:
                code = e["node"].get("code", "")
                desc = e["node"].get("description", "")
                if desc:
                    lines.append(f"  # {desc}")
                if code:
                    for code_line in code.strip().split('\n'):
                        lines.append(f"  {code_line}")
                    lines.append("")

        # 知识来源
        doc_pages = query_result.get("doc_pages", [])
        if doc_pages:
            lines.append("\n【知识来源】")
            for d in doc_pages:
                d_node = d["node"]
                url = d_node.get("url", "")
                title = d_node.get("name", "")
                lines.append(f"  - {title}")
                if url:
                    lines.append(f"    {url}")

        return "\n".join(lines)

    def stats(self) -> dict:
        """返回图谱统计信息"""
        node_count = len(self.graph.get("nodes", []))
        edge_count = len(self.graph.get("edges", []))

        # 按类型统计节点
        label_counts = {}
        for node in self.graph.get("nodes", []):
            label = node.get("label", "Unknown")
            label_counts[label] = label_counts.get(label, 0) + 1

        # 按类型统计关系
        relation_counts = {}
        for edge in self.graph.get("edges", []):
            rel = edge.get("relation", "Unknown")
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "node_types": label_counts,
            "relation_types": relation_counts,
        }

    def get_all_nodes(self) -> list:
        """获取所有节点（用于前端可视化）"""
        return self.graph.get("nodes", [])

    def get_all_edges(self) -> list:
        """获取所有边（用于前端可视化）"""
        return self.graph.get("edges", [])
