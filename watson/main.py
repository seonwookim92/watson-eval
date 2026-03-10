import asyncio
import argparse
import json
import os

from core.parsers.factory import ParserFactory
from core.utils.chunking import chunk_documents
from core.pipeline.graph import create_pipeline
from core.pipeline.nodes import close_mcp_client, get_mcp_client
from core.config import config


def draw_graph(graph_data_json: str):
    """Visualizes the knowledge graph using Pyvis."""
    from pyvis.network import Network

    data = json.loads(graph_data_json)

    net = Network(height="800px", width="100%", bgcolor="#1a1a1a", font_color="white", directed=True)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)

    for node in data.get("nodes", []):
        full_uri = node["id"]
        label = node.get("label", full_uri.split("/")[-1])
        display_label = label if len(label) < 30 else label[:27] + "..."

        types = node.get("types", [])
        type_str = ", ".join(types)
        type_display = f"[{type_str}]" if type_str else ""
        title = f"{type_display} {label}"
        if "description" in node:
            title += f"\nDescription: {node['description']}"

        other_details = [f"{k}: {v}" for k, v in node.items() if k not in ["id", "label", "types", "description"]]
        if other_details:
            title += "\n\n--- Properties ---\n" + "\n".join(other_details)

        color = "#00ffcc"
        if any("Malware" in t for t in types):
            color = "#ff5555"
        elif any("Actor" in t or "Identity" in t for t in types):
            color = "#ffac33"
        elif any("Address" in t or "IP" in label or "Port" in label for t in types):
            color = "#55ff55"
        elif any("File" in t or "Hash" in t for t in types):
            color = "#33ccff"
        elif any("Action" in t or "Observation" in t for t in types):
            color = "#cc33ff"

        net.add_node(node["id"], label=display_label, title=title, color=color, shadow=True)

    for edge in data.get("edges", []):
        net.add_edge(edge["from"], edge["to"], label=edge["label"], color="#aaaaaa", width=1)

    print(f"[*] Graph: {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges.")

    net.set_options("""
    var options = {
      "interaction": { "hover": true, "navigationButtons": true, "search": true },
      "nodes": { "font": { "size": 14, "face": "Tahoma" } },
      "edges": { "color": { "inherit": true }, "smooth": { "type": "continuous" } }
    }
    """)

    output_html = "ontology_graph.html"
    net.show(output_html, notebook=False)
    print(f"[*] Visualization saved to {os.path.abspath(output_html)}")


async def main():
    parser = argparse.ArgumentParser(description="Cyber Ontology Analyzer")
    parser.add_argument("--input", type=str, required=True, help="Input file path or URL")
    parser.add_argument("--schema", type=str, default="uco", choices=["uco", "stix", "all"],
                        help="Ontology schema to use (default: uco)")
    parser.add_argument("--paraphrase", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--chunk_size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=config.DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose_graph", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Generate interactive HTML graph")
    parser.add_argument("--prune", action="store_true", help="Remove disconnected graph components")
    parser.add_argument("--prune_threshold", type=int, default=1)
    parser.add_argument("--summarize", action="store_true", help="Add descriptions to extracted entities")
    parser.add_argument("--output", type=str, default=None, help="Save graph output to file")
    parser.add_argument("--eval-output", type=str, default=None,
                        help="Save SRO triples + entity extractions as JSON (for evaluation)")

    args = parser.parse_args()

    # Apply schema before MCP client is created
    if args.schema != "uco":
        config.set_schema(args.schema)

    print(f"[*] Processing: {args.input}  (schema: {args.schema})")

    docs = ParserFactory.load_content(args.input)
    if not docs:
        print("[!] No content loaded. Exiting.")
        return

    chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if args.verbose:
        print(f"[*] Text split into {len(chunks)} chunks.")

    pipeline = create_pipeline()

    initial_state = {
        "documents": chunks,
        "processed_text": "",
        "ontology_output": "",
        "source_info": args.input,
        "use_paraphrasing": args.paraphrase == "on",
        "verbose_graph": args.verbose_graph,
        "verbose": args.verbose,
        "include_summaries": args.summarize,
        "current_chunk_index": 0,
        "accumulated_graph": "",
        "sro_triples": [],
        "entity_extractions": [],
    }

    try:
        mcp_client = await get_mcp_client()

        if args.verbose:
            print("[*] Resetting knowledge graph state...")
        await mcp_client.call_tool("reset_graph", {})

        final_output = await pipeline.ainvoke(initial_state)

        if args.prune:
            if args.verbose:
                print(f"[*] Pruning islands (size <= {args.prune_threshold})...")
            await mcp_client.call_tool("prune_islands", {"min_size": args.prune_threshold})
            viz_blocks = await mcp_client.call_tool("visualize_graph", {"verbose": args.verbose_graph})
            final_output["ontology_output"] = "".join(
                [b.text if hasattr(b, "text") else str(b) for b in viz_blocks]
            )

        print("\n" + "=" * 50)
        print("--- Ontology Graph ---")
        print("=" * 50)
        print(final_output["ontology_output"])
        print("=" * 50)

        if args.output:
            output_path = os.path.join(os.getcwd(), args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Source: {args.input}\n\n")
                f.write("=== Ontology Graph ===\n")
                f.write(final_output["ontology_output"])
            print(f"[+] Graph saved to {args.output}")

        # Save evaluation-friendly JSON
        if args.eval_output:
            eval_data = {
                "source": args.input,
                "schema": args.schema,
                "sro_triples": final_output.get("sro_triples", []),
                "entity_extractions": final_output.get("entity_extractions", []),
            }
            eval_path = os.path.join(os.getcwd(), args.eval_output)
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
            print(f"[+] Evaluation output saved to {args.eval_output}")
            print(f"    SRO triples: {len(eval_data['sro_triples'])}")
            print(f"    Entities:    {len(eval_data['entity_extractions'])}")

        if args.visualize:
            blocks = await mcp_client.call_tool("get_graph_data", {})
            graph_json = "".join([
                b.text if hasattr(b, "text") else (b["text"] if isinstance(b, dict) else str(b))
                for b in blocks
            ])
            draw_graph(graph_json)

    finally:
        print("[*] Cleaning up...")
        await close_mcp_client()


if __name__ == "__main__":
    asyncio.run(main())
