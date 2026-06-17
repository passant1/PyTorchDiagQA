"""
PyTorch 智能报错诊断问答系统 - 主入口
支持桌面 GUI 模式和命令行模式
"""
import sys
import os
import argparse

# 确保项目根目录在 sys.path 中
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
# 支持 vendor/ 本地包（离线运行）
_VENDOR = os.path.join(_BASE, "vendor")
if os.path.isdir(_VENDOR):
    sys.path.append(_VENDOR)


def run_gui():
    """启动桌面 GUI"""
    from frontend.desktop_app import main
    main()


def run_cli():
    """命令行交互模式"""
    from qa_engine import get_engine

    print("=" * 60)
    print(">>> PyTorch 智能报错诊断问答系统 (命令行模式)")
    print("=" * 60)
    print("支持的问题类型:")
    print("  - CUDA 不可用 / CUDA out of memory")
    print("  - PyTorch 安装版本问题")
    print("  - DataLoader 参数和多进程问题")
    print("  - 模型保存与加载问题")
    print("  - state_dict 不匹配")
    print("  - torch.load 安全问题")
    print("  - CrossEntropyLoss 输入问题")
    print("=" * 60)
    print("输入 'exit' 或 'quit' 退出")
    print("输入 'stats' 查看系统统计")
    print("输入 'build' 构建知识图谱")
    print()

    engine = get_engine()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "退出"):
            print("再见!")
            break

        if user_input.lower() == "stats":
            stats = engine.stats()
            print("\n【系统统计】")
            kg = stats["kg_stats"]
            print(f"  知识图谱: {kg['total_nodes']} 节点, {kg['total_edges']} 条边")
            print(f"  节点类型: {kg['node_types']}")
            print(f"  关系类型: {kg['relation_types']}")
            print(f"  FAQ 条目: {stats['faq_count']}")
            print(f"  大模型: {'可用' if stats['llm_available'] else '离线模式'}")
            if stats['llm_available']:
                print(f"  模型: {stats['model_name']}")
            print()
            continue

        if user_input.lower() == "build":
            print("正在构建知识图谱...")
            try:
                from kg_builder.graph_builder import build_graph
                from kg_builder.export_csv import export_to_csv
                graph = build_graph()
                export_to_csv()
                print(f"[OK] 完成: {len(graph['nodes'])} 节点, {len(graph['edges'])} 条边\n")
            except Exception as e:
                print(f"[ERROR] 构建失败: {e}\n")
            continue

        # 诊断
        print("[...] 正在分析...")
        result = engine.ask(user_input)
        print("\n" + result["answer"])
        print(f"\n--- [意图: {result['intent']} | 实体: {result['matched_entity']} | 来源: {result['source']} | 置信度: {result['confidence']:.2f}] ---\n")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="PyTorch 智能报错诊断问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py              # 启动桌面 GUI
  python main.py --cli        # 命令行交互模式
  python main.py --build      # 构建知识图谱后退出
  python main.py --ask "CUDA out of memory 怎么办？"
        """
    )
    parser.add_argument("--cli", action="store_true", help="命令行交互模式")
    parser.add_argument("--gui", action="store_true", help="桌面 GUI 模式（默认）")
    parser.add_argument("--build", action="store_true", help="仅构建知识图谱")
    parser.add_argument("--ask", type=str, help="单次问答查询")
    parser.add_argument("--server", action="store_true", help="启动 FastAPI 后端服务")

    args = parser.parse_args()

    if args.server:
        from backend.app import run_server
        run_server()
    elif args.build:
        print("正在构建知识图谱...")
        from kg_builder.graph_builder import build_graph
        from kg_builder.export_csv import export_to_csv
        graph = build_graph()
        export_to_csv()
        print(f"[OK] 完成: {len(graph['nodes'])} 节点, {len(graph['edges'])} 条边")
    elif args.ask:
        from qa_engine import get_engine
        engine = get_engine()
        result = engine.ask(args.ask)
        print(result["answer"])
    elif args.cli:
        run_cli()
    elif args.gui:
        print("正在启动桌面 GUI...")
        run_gui()
    elif getattr(sys, 'frozen', False):
        # 打包后的 exe 默认启动 server 模式
        from backend.app import run_server
        run_server()
    else:
        # 开发环境默认启动 server 模式
        from backend.app import run_server
        run_server()


if __name__ == "__main__":
    main()
