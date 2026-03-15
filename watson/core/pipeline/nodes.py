import asyncio
import json
import re
import urllib.parse

from langchain_core.messages import SystemMessage, HumanMessage
from core.pipeline.state import GraphState
from core.config import config
from core.mcp.client import MCPClient


def get_llm(provider=None):
    p = provider or config.LLM_PROVIDER
    cfg = config.get_provider_config(p)
    
    if p == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": cfg["model"], "api_key": cfg["api_key"],
                  "model_kwargs": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        return ChatOpenAI(**kwargs)
    elif p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=cfg["model"], google_api_key=cfg["api_key"])
    elif p == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=cfg["model"], anthropic_api_key=cfg["api_key"])
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg["model"], base_url=cfg["base_url"], num_ctx=config.OLLAMA_NUM_CTX)


async def paraphrasing_node(state: GraphState) -> GraphState:
    """Rewrites chunk text into clean S-P-O form for better ontology extraction."""
    if not state.get("use_paraphrasing", False):
        state["processed_text"] = state["documents"][state["current_chunk_index"]].page_content
        return state

    llm = get_llm()
    text = state["documents"][state["current_chunk_index"]].page_content

    system_prompt = (
        "You are an ontology specialist. Rewrite the following text to be 'Ontology-friendly'.\n"
        "Rules:\n"
        "1. Identify and keep all domain-specific proper nouns.\n"
        "2. Replace all pronouns (he, she, it, they, this, that) with their corresponding explicit subjects.\n"
        "3. Simplify sentences into clear Subject-Predicate-Object (S-P-O) triplets.\n"
        "4. Remove flowery language and filler words, but PRESERVE the semantic meaning and intent.\n"
        "5. Output only the paraphrased text."
    )

    if state.get("verbose"):
        print(f"\n[VERBOSE: Paraphrasing Chunk {state['current_chunk_index'] + 1}/{len(state['documents'])}]")
        print("-" * 50)
        print(f"Original: {text[:200]}...")

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ])

    state["processed_text"] = response.content

    if state.get("verbose"):
        print(f"Paraphrased: {response.content[:200]}...")
        print("-" * 50)
    return state


_mcp_client = None


async def get_mcp_client():
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
        await _mcp_client.connect()
    return _mcp_client


async def close_mcp_client():
    global _mcp_client
    if _mcp_client:
        await _mcp_client.disconnect()
        _mcp_client = None


async def ontology_mapping_node(state: GraphState) -> GraphState:
    """
    Extracts entities and relationships from text, maps them to the loaded ontology,
    and also captures natural-language SRO triples for evaluation.
    """
    mcp_client = await get_mcp_client()
    text = state["processed_text"]
    llm = get_llm()

    try:
        few_shot_example = (
            "Example Input: 'Attacker A sent a phishing email with Malware X (hash 123). "
            "Malware X targets Windows, connects to 1.1.1.1 on port 443.'\n"
            "Example Output:\n"
            "{\n"
            "  \"entities\": [\n"
            "    {\"id\": \"Attacker_A\", \"types\": [\"https://ontology.unifiedcyberontology.org/uco/identity/Organization\"], \"description\": \"Threat group responsible for the phishing campaign.\"},\n"
            "    {\"id\": \"Malware_X\", \"types\": [\"https://ontology.unifiedcyberontology.org/uco/tool/MaliciousTool\"], \"description\": \"Malware variant targeting Windows systems.\"},\n"
            "    {\"id\": \"Action_Phishing\", \"types\": [\"https://ontology.unifiedcyberontology.org/uco/action/Action\"], \"description\": \"Email delivery of the malware attachment.\"},\n"
            "    {\"id\": \"Action_C2\", \"types\": [\"https://ontology.unifiedcyberontology.org/uco/action/Action\"], \"description\": \"C2 communication event.\"},\n"
            "    {\"id\": \"IPv4_1.1.1.1\", \"types\": [\"https://ontology.unifiedcyberontology.org/uco/observable/IPv4Address\"], \"description\": \"C2 server IP address.\"}\n"
            "  ],\n"
            "  \"properties\": [\n"
            "    {\"subj\": \"Action_Phishing\", \"pred\": \"performer\", \"obj\": \"Attacker_A\"},\n"
            "    {\"subj\": \"Action_Phishing\", \"pred\": \"result\", \"obj\": \"Malware_X\"},\n"
            "    {\"subj\": \"Malware_X\", \"pred\": \"name\", \"obj\": \"Malware X\"},\n"
            "    {\"subj\": \"Malware_X\", \"pred\": \"hashValue\", \"obj\": \"123\"},\n"
            "    {\"subj\": \"Action_C2\", \"pred\": \"performer\", \"obj\": \"Malware_X\"},\n"
            "    {\"subj\": \"Action_C2\", \"pred\": \"object\", \"obj\": \"IPv4_1.1.1.1\"},\n"
            "    {\"subj\": \"Action_C2\", \"pred\": \"port\", \"obj\": \"443\"}\n"
            "  ],\n"
            "  \"sro_triples\": [\n"
            "    {\"subject\": \"Attacker A\", \"relation\": \"sent phishing email containing\", \"relation_class\": \"result\", \"object\": \"Malware X\"},\n"
            "    {\"subject\": \"Malware X\", \"relation\": \"targets\", \"relation_class\": \"object\", \"object\": \"Windows\"},\n"
            "    {\"subject\": \"Malware X\", \"relation\": \"connects to\", \"relation_class\": \"communicatesWith\", \"object\": \"1.1.1.1\"}\n"
            "  ]\n"
            "}\n"
        )

        summary_rule = ""
        if state.get("include_summaries"):
            summary_rule = (
                "1. MANDATORY Documentation: Every entity MUST be listed in 'entities' "
                "with a 'types' list and a 'description' field.\n"
            )

        extraction_prompt = (
            "Extract entities and the FULL chain of relationships from the text.\n"
            "Output THREE sections:\n"
            "  - entities: typed entities mapped to ontology class URIs\n"
            "  - properties: ontology-mapped subject-predicate-object triples\n"
            "  - sro_triples: natural-language Subject-Relation-Object triples (short verb phrases)\n"
            "Rules:\n"
            f"{summary_rule}"
            "2. DO NOT create isolated nodes. Use Action entities to link performers, objects, and results.\n"
            "3. Technical details (port, hash, IP) MUST be attached as properties, not standalone entities.\n"
            "4. Co-reference: reuse the EXACT same entity ID when the same entity reappears.\n"
            "5. Every Action node must have a 'performer' and an 'object' or 'result' link.\n"
            "6. sro_triples must use the original entity names (not IDs) and natural verb phrases.\n"
            "7. Each sro_triple must include a 'relation_class' field: the short ontology ObjectProperty name "
            "that best maps to this relation (e.g., performer, object, result, communicatesWith, uses, targets).\n"
            f"\n{few_shot_example}\n"
            f"Text to Analyze: {text}"
        )

        if state.get("verbose"):
            print(f"\n[VERBOSE: Mapping Chunk {state['current_chunk_index'] + 1}/{len(state['documents'])}]")
            print("-" * 50)
            print(f"Input Text: {text[:200]}...")

        response = await llm.ainvoke([
            SystemMessage(content=(
                "You are a professional cyber security analyst and ontology mapper. "
                "Build a structured knowledge graph and also extract plain natural-language SRO triples."
            )),
            HumanMessage(content=extraction_prompt)
        ])

        if state.get("verbose"):
            print("\n[VERBOSE: Raw LLM Output]")
            print(response.content)
            print("-" * 50)

        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if not match:
            state["current_chunk_index"] += 1
            return state

        data = json.loads(match.group(0))
        entities = data.get("entities", [])
        properties = data.get("properties", [])
        chunk_sro = data.get("sro_triples", [])

        if state.get("verbose"):
            print(f"Extracted Entities: {[e['id'] for e in entities]}")
            print(f"Extracted Relations: {len(properties)} ontology triples, {len(chunk_sro)} SRO triples")

        # ── Entity creation ──────────────────────────────────────────────────
        created_ids = set()
        entity_extractions_chunk = []

        async def resolve_class(cls_uri: str) -> str:
            query_clean = str(cls_uri).strip().lower()

            known_map = {
                "malware": "https://ontology.unifiedcyberontology.org/uco/tool/MaliciousTool",
                "action": "https://ontology.unifiedcyberontology.org/uco/action/Action",
                "ipv4address": "https://ontology.unifiedcyberontology.org/uco/observable/IPv4Address",
                "file": "https://ontology.unifiedcyberontology.org/uco/observable/File",
                "threatactor": "https://ontology.unifiedcyberontology.org/uco/identity/Organization",
                "organization": "https://ontology.unifiedcyberontology.org/uco/identity/Organization",
            }

            try:
                search_resp = await mcp_client.call_tool("search_classes", {"query": query_clean})
                res_text = "".join([b.text for b in search_resp if hasattr(b, "text")])

                matches = re.findall(
                    r'(\d+)\. ([^ ]+) \(Sim: ([\d\.]+)\)\n\s+URI: (https?://[^\n]+)', res_text
                )
                if matches:
                    for idx, name, sim, uri in matches:
                        if uri == cls_uri or name.lower() == query_clean:
                            return uri
                    query_parts = query_clean.split("/")[-1].split("#")[-1].replace("_", " ").lower().split()
                    for idx, name, sim, uri in matches:
                        if all(part in name.lower() for part in query_parts):
                            return uri
                    idx, name, sim, uri = matches[0]
                    if float(sim) > 0.6:
                        return uri
            except Exception:
                pass

            for key, uri in known_map.items():
                if key in query_clean:
                    return uri

            return "https://ontology.unifiedcyberontology.org/uco/core/UcoObject"

        INSTANCE_BASE = "http://example.org/entities/"

        def get_safe_uri(eid: str) -> str:
            if str(eid).startswith("http"):
                return str(eid)
            return INSTANCE_BASE + urllib.parse.quote(str(eid).replace(" ", "_"))

        for ent in entities:
            resolved_types = []
            for t in ent.get("types", []):
                rt = await resolve_class(t)
                if rt:
                    resolved_types.append(rt)

            if not resolved_types:
                resolved_types = ["https://ontology.unifiedcyberontology.org/uco/core/UcoObject"]

            await mcp_client.call_tool("create_entity", {
                "entity_id": ent["id"],
                "class_uris": resolved_types,
            })
            created_ids.add(ent["id"])

            # Capture for evaluation output
            short_class = resolved_types[0].split("/")[-1]
            entity_extractions_chunk.append({
                "name": ent["id"].replace("_", " "),
                "ontology_class_short": short_class,
                "ontology_class_uri": resolved_types[0],
            })

            subj_uri = get_safe_uri(ent["id"])
            for attr_name in ["name", "description"]:
                if attr_name in ent and ent[attr_name]:
                    pred_uri = "https://ontology.unifiedcyberontology.org/uco/core/" + attr_name
                    await mcp_client.call_tool("set_property", {
                        "entity_uri": subj_uri,
                        "property_uri": pred_uri,
                        "value": str(ent[attr_name]),
                    })

        # ── Property mapping ─────────────────────────────────────────────────
        async def resolve_pred_and_facet(pred: str) -> tuple[str, str]:
            query_clean = str(pred).strip().lower()

            DIRECT_PROPS = {
                "https://ontology.unifiedcyberontology.org/uco/action/performer",
                "https://ontology.unifiedcyberontology.org/uco/action/object",
                "https://ontology.unifiedcyberontology.org/uco/action/result",
                "https://ontology.unifiedcyberontology.org/uco/core/name",
                "https://ontology.unifiedcyberontology.org/uco/core/description",
                "https://ontology.unifiedcyberontology.org/uco/core/specVersion",
            }

            mapping = {
                "hasTarget":       ("https://ontology.unifiedcyberontology.org/uco/action/object", None),
                "hasSubject":      ("https://ontology.unifiedcyberontology.org/uco/action/performer", None),
                "hasResult":       ("https://ontology.unifiedcyberontology.org/uco/action/result", None),
                "performer":       ("https://ontology.unifiedcyberontology.org/uco/action/performer", None),
                "object":          ("https://ontology.unifiedcyberontology.org/uco/action/object", None),
                "result":          ("https://ontology.unifiedcyberontology.org/uco/action/result", None),
                "name":            ("https://ontology.unifiedcyberontology.org/uco/core/name", None),
                "description":     ("https://ontology.unifiedcyberontology.org/uco/core/description", None),
                "communicatesWith":("https://ontology.unifiedcyberontology.org/uco/observable/connectedTo", None),
                "fileName":        ("https://ontology.unifiedcyberontology.org/uco/observable/fileName",
                                    "https://ontology.unifiedcyberontology.org/uco/observable/FileFacet"),
                "hashValue":       ("https://ontology.unifiedcyberontology.org/uco/observable/hashValue",
                                    "https://ontology.unifiedcyberontology.org/uco/observable/ContentDataFacet"),
                "ipAddressValue":  ("https://ontology.unifiedcyberontology.org/uco/observable/addressValue",
                                    "https://ontology.unifiedcyberontology.org/uco/observable/IPAddressFacet"),
                "addressValue":    ("https://ontology.unifiedcyberontology.org/uco/observable/addressValue",
                                    "https://ontology.unifiedcyberontology.org/uco/observable/DigitalAddressFacet"),
            }

            if pred in mapping:
                return mapping[pred]

            try:
                search_resp = await mcp_client.call_tool("search_properties", {"query": pred})
                res_text = "".join([b.text for b in search_resp if hasattr(b, "text")])

                uri_match = re.search(r'\(https?://[^\)]+\)', res_text)
                if not uri_match:
                    return None, None

                found_uri = uri_match.group(0)[1:-1]

                if found_uri in DIRECT_PROPS or "uco/core/" in found_uri or "uco/action/" in found_uri:
                    return found_uri, None

                found_facet = None
                if "uco/observable/" in found_uri:
                    facet_match = re.search(r'TIP: This property belongs to \[([^\]]+)\]', res_text)
                    if facet_match:
                        facet_name = facet_match.group(1)
                        f_search = await mcp_client.call_tool("search_classes", {"query": facet_name})
                        f_text = "".join([b.text for b in f_search if hasattr(b, "text")])
                        f_uri_match = re.search(r'URI: (https?://[^\n]+)', f_text)
                        if f_uri_match:
                            found_facet = f_uri_match.group(1).strip()

                return found_uri, found_facet
            except Exception:
                pass

            return None, None

        for prop in properties:
            subj_id = prop.get("subj") or prop.get("subject")
            pred_raw = prop.get("pred") or prop.get("predicate") or prop.get("property")
            obj_val = prop.get("obj") or prop.get("object") or prop.get("value")

            if not subj_id or not pred_raw or obj_val is None:
                continue

            subj_uri = get_safe_uri(subj_id)
            pred_uri, facet_uri = await resolve_pred_and_facet(pred_raw)

            if not pred_uri:
                continue

            if isinstance(obj_val, str) and not str(obj_val).startswith("http"):
                looks_like_entity = (obj_val in created_ids) or (
                    "_" in str(obj_val) and "." not in str(obj_val) and len(str(obj_val)) < 50
                )
                if looks_like_entity:
                    obj_val = get_safe_uri(obj_val)

            if facet_uri:
                if state.get("verbose"):
                    print(f"Attaching Facet: {subj_uri} --[{facet_uri}]--> {pred_uri}: {obj_val}")
                await mcp_client.call_tool("attach_component", {
                    "entity_uri": subj_uri,
                    "component_class": facet_uri,
                    "connection_prop": "https://ontology.unifiedcyberontology.org/uco/core/hasFacet",
                    "attributes": {pred_uri: obj_val},
                })
            else:
                if state.get("verbose"):
                    print(f"Setting Property: {subj_uri} --({pred_uri})--> {obj_val}")
                await mcp_client.call_tool("set_property", {
                    "entity_uri": subj_uri,
                    "property_uri": pred_uri,
                    "value": obj_val,
                })

        # ── Accumulate evaluation outputs ────────────────────────────────────
        current_sro = state.get("sro_triples") or []
        state["sro_triples"] = current_sro + chunk_sro

        current_ents = state.get("entity_extractions") or []
        seen_names = {e["name"] for e in current_ents}
        state["entity_extractions"] = current_ents + [
            e for e in entity_extractions_chunk if e["name"] not in seen_names
        ]

    except Exception as e:
        print(f"Extraction error: {e}")

    # ── Visualize current graph state ────────────────────────────────────────
    try:
        verbose = state.get("verbose_graph", False)
        viz = await mcp_client.call_tool("visualize_graph", {"verbose": verbose})

        viz_text = ""
        for block in viz:
            if hasattr(block, "text"):
                viz_text += block.text
            elif isinstance(block, dict) and "text" in block:
                viz_text += block["text"]
            elif hasattr(block, "content"):
                viz_text += str(block.content)

        state["accumulated_graph"] = viz_text
    except Exception as e:
        print(f"Visualization error: {e}")

    state["current_chunk_index"] += 1
    return state


def next_step_router(state: GraphState):
    if state["current_chunk_index"] < len(state["documents"]):
        return "paraphrase"
    return "visualize"


async def visualization_node(state: GraphState) -> GraphState:
    state["ontology_output"] = state["accumulated_graph"]
    return state
