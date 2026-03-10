"""
Batch evaluation runner.

For each sample in a dataset:
1. Reset the MCP graph state
2. Run the ontology mapping pipeline
3. Collect sro_triples and entity_extractions
4. Compare against ground truth using the provided matcher
5. Aggregate precision / recall / F1
"""

from typing import Optional, List

from langchain_core.documents import Document

from core.pipeline.graph import create_pipeline
from core.pipeline.nodes import get_mcp_client, close_mcp_client
from core.utils.chunking import chunk_documents
from core.eval.loaders import load_ctinexus, load_ctikg
from core.eval.metrics import evaluate_triples, evaluate_entities, aggregate_metrics
from core.config import config


class EvalRunner:
    def __init__(self, verbose: bool = False, matcher=None):
        """
        matcher: any matcher from core.eval.matchers.
                 Defaults to JaccardMatcher(0.5) when None.
        """
        self.verbose = verbose
        self.matcher = matcher
        self.pipeline = create_pipeline()

    async def run(
        self,
        dataset_type: str,
        data_path: str,
        limit: Optional[int] = None,
        output_format: str = "eval",
        schema: str = "uco",
    ) -> object:
        if dataset_type == "ctinexus":
            samples = load_ctinexus(data_path)
        elif dataset_type == "ctikg":
            samples = load_ctikg(data_path)
        else:
            raise ValueError(f"Unknown dataset: '{dataset_type}'. Choose 'ctinexus' or 'ctikg'.")

        if limit:
            samples = samples[:limit]

        matcher_name = type(self.matcher).__name__ if self.matcher else "JaccardMatcher"
        print(f"[*] Loaded {len(samples)} samples from '{dataset_type}' | matcher: {matcher_name}")

        mcp_client = await get_mcp_client()

        sample_results: List[dict] = []
        triple_metrics_list: List[dict] = []
        entity_metrics_list: List[dict] = []

        try:
            for i, sample in enumerate(samples):
                print(f"[{i + 1}/{len(samples)}] {sample['id']}", end=" ... ", flush=True)

                await mcp_client.call_tool("reset_graph", {})

                docs = [Document(page_content=sample["text"])]
                chunks = chunk_documents(
                    docs,
                    chunk_size=config.DEFAULT_CHUNK_SIZE,
                    chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
                )

                initial_state = {
                    "documents": chunks,
                    "processed_text": "",
                    "ontology_output": "",
                    "source_info": sample["id"],
                    "use_paraphrasing": False,
                    "verbose_graph": False,
                    "verbose": self.verbose,
                    "include_summaries": False,
                    "current_chunk_index": 0,
                    "accumulated_graph": "",
                    "sro_triples": [],
                    "entity_extractions": [],
                }

                try:
                    result = await self.pipeline.ainvoke(initial_state)

                    predicted_triples  = result.get("sro_triples",       [])
                    predicted_entities = result.get("entity_extractions", [])

                    if output_format == "extract":
                        # Baseline-compatible format: list of raw extractions
                        sample_results.append({
                            "file": sample["id"] + ".json",
                            "text": sample["text"],
                            "ontology": schema,
                            "extracted_entities": [
                                {
                                    "name": e.get("name", ""),
                                    "class": e.get("ontology_class_short", "UcoObject"),
                                }
                                for e in predicted_entities
                            ],
                            "extracted_triplets": [
                                {
                                    "subject": t.get("subject", ""),
                                    "relation": t.get("relation", ""),
                                    "relation_class": t.get("relation_class", ""),
                                    "object": t.get("object", ""),
                                }
                                for t in predicted_triples
                            ],
                        })
                        print("done")
                    else:
                        # Evaluation format: metrics per sample
                        triple_m = await evaluate_triples(
                            predicted_triples,
                            sample["ground_truth_triples"],
                            matcher=self.matcher,
                        )
                        triple_metrics_list.append(triple_m)

                        sample_result = {
                            "id":                 sample["id"],
                            "predicted_triples":  predicted_triples,
                            "gold_triples":       sample["ground_truth_triples"],
                            "triple_metrics":     triple_m,
                            "predicted_entities": predicted_entities,
                            "gold_entities":      sample.get("ground_truth_entities", []),
                        }

                        if sample.get("ground_truth_entities"):
                            entity_m = await evaluate_entities(
                                predicted_entities,
                                sample["ground_truth_entities"],
                                matcher=self.matcher,
                            )
                            sample_result["entity_metrics"] = entity_m
                            entity_metrics_list.append(entity_m)

                        if sample.get("tactic"):
                            sample_result["tactic"] = sample["tactic"]

                        print(
                            f"P={triple_m['precision']:.3f}  "
                            f"R={triple_m['recall']:.3f}  "
                            f"F1={triple_m['f1']:.3f}"
                        )
                        sample_results.append(sample_result)

                except Exception as e:
                    print(f"ERROR: {e}")
                    if output_format == "extract":
                        sample_results.append({
                            "file": sample["id"] + ".json",
                            "text": sample["text"],
                            "ontology": schema,
                            "error": str(e),
                            "extracted_entities": [],
                            "extracted_triplets": [],
                        })
                    else:
                        sample_results.append({"id": sample["id"], "error": str(e)})

        finally:
            await close_mcp_client()

        if output_format == "extract":
            return sample_results

        output = {
            "dataset":        dataset_type,
            "matcher":        matcher_name,
            "num_samples":    len(samples),
            "triple_metrics": aggregate_metrics(triple_metrics_list),
            "samples":        sample_results,
        }
        if entity_metrics_list:
            output["entity_metrics"] = aggregate_metrics(entity_metrics_list)

        return output
