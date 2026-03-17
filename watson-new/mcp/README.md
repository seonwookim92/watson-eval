# Universal Ontology MCP

**The Intelligent Bridge between Unstructured Data and High-Fidelity Knowledge Graphs.**

Universal Ontology MCP is a powerful tool designed for AI assistants to explore, navigate, and populate complex ontologies. It transforms raw text into structured relationships while adhering to strict semantic standards.

## Why Universal Ontology MCP?

Existing ontology tools often struggle with semantic ambiguity and rigid keyword matching. This MCP solves these problems by providing:

- **Dual-Mode Class Discovery**: Two complementary search strategies — fast embedding-based search (`search_classes`) and LLM-driven hierarchical navigation (`drill_into_classes`) — cover each other's failure modes.
- **Embedding Cache**: Ontology embeddings are computed once and cached to disk. Subsequent startups skip re-encoding entirely, dramatically reducing load time for large ontologies. Cache is automatically invalidated when ontology files change.
- **Batch Embedding Encoding**: All class and property embeddings are computed in a single batched pass at startup rather than one-by-one, making initial load significantly faster.
- **Balanced Semantic Search**: Uses state-of-the-art embeddings (default: `all-MiniLM-L6-v2`) with a **50/50 weighted scoring** between class names and descriptions for intuitive matching.
- **Context-Aware Property Recommendation**: The engine analyzes **Domain/Range constraints** of your entities and recommends the most valid `DatatypeProperty` or `ObjectProperty`, preventing schema violations.
- **Advanced Facet Discovery**: Locates related Facet classes through four complementary strategies — direct property range, `rdfs:subPropertyOf` chain, SHACL `sh:node` resolution, and name heuristics.
- **Structured Validation Output**: `validate_entity` returns structured JSON detailing exactly which properties are missing or exceed limits, along with actionable fix instructions the AI can act on immediately.
- **Duplicate-Safe Entity Creation**: `create_entity` detects pre-existing entities and reports which types were added, skipped (already assigned), or invalid — no silent overwrites.
- **Automatic Property Normalization**: Detects and normalizes non-standard types like `owl:DataProperty` into the standard `owl:DatatypeProperty` for consistent mapping.
- **Proactive Schema Guidance**: Identifies mandatory fields and expected entity types for ObjectProperties in real-time.
- **Component-Based Modeling**: Simplifies complex modeling (like UCO Facets) by recommending relevant components for any given class.
- **ASCII Graph Visualization**: Preview your knowledge graph structure directly in the chat interface before exporting.
- **Built-in SHACL Validation**: Ensures data integrity with schema constraint checking before export.
- **Session Control & Sanitization**: Surgically remove entities, reset the session, prune disconnected islands, and enjoy automatic URI sanitization.
- **Multi-Format Ontology Support**: Loads `.ttl`, `.owl`, `.rdf`, `.n3`, `.nt`, `.trig`, `.jsonld` files simultaneously with automatic format detection.

---

## Supported Ontology Formats

The engine automatically detects and loads all recognized ontology files under `ONTOLOGY_DIR` recursively.

| Extension | Format | Description |
|-----------|--------|-------------|
| `.ttl` | Turtle | Human-readable RDF syntax. Used by UCO, CASE, schema.org |
| `.owl` | OWL/RDF-XML | Protégé-native format. Used by STIX 2.1 OWL, TAC, TAL |
| `.rdf` | RDF/XML | Generic RDF/XML format |
| `.n3` | Notation3 | Extended Turtle superset |
| `.nt` | N-Triples | Line-based RDF format |
| `.trig` | TriG | Named graph RDF format |
| `.jsonld` | JSON-LD | JSON-based RDF format |

These are all different **serialization formats** of the same underlying RDF/OWL data model — the engine merges them into a single graph regardless of format.

---

## Ontology Examples

### UCO (Turtle / `.ttl`)

[Unified Cyber Ontology](https://github.com/ucoProject/UCO) uses Turtle format. Point `ONTOLOGY_DIR` to the `ontology/` subdirectory:

```
UCO/ontology/          ← point here
├── core/
├── observable/
├── action/
└── ...
```

### STIX 2.1 (OWL / `.owl`)

The [tac-ontology](https://github.com/oasis-open/tac-ontology) repository provides a formal OWL formalization of STIX 2.1. For pure STIX 2.1 CTI work, point to the `stix/` subdirectory:

```
tac-ontology/stix/     ← point here for STIX 2.1 only
├── core-objects/
│   ├── sdo/           ← AttackPattern, ThreatActor, Malware, ...
│   ├── sco/           ← IPv4Address, File, NetworkTraffic, ...
│   └── sro/           ← Relationship, Sighting
├── meta-objects/
├── bundle-object/
└── vocabularies/
```

> **Note on ATT&CK TTPs**: MITRE ATT&CK Techniques (e.g. T1059.001) are instances of `stix:AttackPattern`, with the T-number stored in `external_references`. The full ATT&CK framework is officially distributed as STIX 2.1 data.

---

## Class Discovery Strategy

Two tools cover complementary failure modes:

| Situation | Tool |
|-----------|------|
| You know the concept name | `search_classes("Vulnerability")` — one concept per call |
| Search scores are all below 0.5 | `drill_into_classes("vulnerability")` — read descriptions, pick branch, repeat |
| Want a structural overview | `show_class_tree()` or `show_class_tree('<uri>', depth=3)` |

**Important**: Always pass **one concept per query** to `search_classes` or `search_properties`. Multi-concept queries (e.g. `"Vulnerability Software Malware Exploit"`) dilute the embedding vector and return poor results. Run one call per concept, in parallel if possible.

### `drill_into_classes` — LLM-Driven Hierarchical Navigation

When `search_classes` returns no confident match (all scores < 0.5), use `drill_into_classes` to navigate the class tree yourself:

```
1. drill_into_classes("vulnerability")
   → Shows root classes with full descriptions. Read and pick the best branch.

2. drill_into_classes("vulnerability", "https://.../UcoObject")
   → Shows UcoObject's subclasses. Pick the next branch.

3. Repeat until you reach the right class or a leaf node.

4. Confirm with get_class_details('<uri>') before creating entities.
```

Unlike `search_classes`, this tool computes **no embeddings** — you (the LLM) read the descriptions and decide. This leverages LLM language understanding, which outperforms small embedding models for domain-specific terminology.

---

## Intelligent Tools

### Schema Inspection

- **`get_ontology_summary`**: Quick high-level overview of the loaded schema (class count, property count, SHACL shapes, namespaces).
- **`search_classes(query, limit)`**: Embedding + keyword hybrid search. **One concept per query.**
- **`search_properties(query)`**: Embedding + keyword hybrid search for properties. **One concept per query.**
- **`drill_into_classes(query, class_uri?)`**: LLM-driven top-down navigation. Fallback when `search_classes` scores are all below 0.5. No embeddings — shows raw names and descriptions for your judgment.
- **`show_class_tree(class_uri?, depth)`**: Displays the class hierarchy as an indented tree for structural orientation.
- **`get_class_hierarchy(class_uri)`**: Returns the inheritance path from root to a specified class.
- **`list_subclasses(class_uri)`**: Lists immediate subclasses of a class.
- **`list_root_classes`**: Lists top-level classes with no parents.
- **`get_class_details(class_uri)`**: Detailed usage instructions, SHACL-derived property requirements, and connectivity rules for a class.
- **`list_available_facets(class_uri)`**: Semantically ranked list of applicable Facet/Component classes.

### Smart Recommendation

- **`recommend_attribute(entity_uri, query, value?, context?)`**: Recommends the best `DatatypeProperty` for an entity based on its class hierarchy, user intent, actual data value, and type inference (Date, Integer, Boolean, String).
  - Traverses related Facets automatically to find hidden attributes.
  - Boosts candidates whose range matches the inferred data type.
- **`recommend_relation(subject_uri, object_uri, query, context?)`**: Recommends the best `ObjectProperty` to connect two entities by analyzing Domain/Range constraints and structural paths.
  - Discovers direct, Facet-bridged, and inverse relations.
  - Applies small score adjustments (boost for direct, penalty for inverse).
- **`recommend_property(subject_type_uri, object_type_uri, predicate, context?)`**: Recommends up to 3 schema property URIs for a type pair and returns structured JSON.
  - Considers 4 directions: `S -> O` object property, `O -> S` object property, `S -> literal` data property, `O -> literal` data property.
  - Uses an OpenAI-compatible LLM over a constrained candidate list and retries up to 3 times if the returned JSON is invalid.

### Graph Construction

- **`create_entity(entity_id, class_uris)`**: Creates a multi-typed entity. Detects pre-existing entities and reports added, duplicate-skipped, and invalid class URIs separately.
- **`set_property(entity_uri, property_uri, value)`**: Sets a property value. Values starting with `http` are automatically treated as URI links.
- **`attach_component(entity_uri, component_class, connection_prop, attributes)`**: Groups metadata into a Facet/Component node. Uses UUID-based URIs to prevent collisions.
- **`remove_entity(entity_uri)`**: Deletes an entity and all its incoming/outgoing triples.
- **`reset_graph`**: Resets all instance data while reloading the ontology schema.

### Graph Analysis & Export

- **`validate_entity(entity_uri)`**: Validates an entity against SHACL constraints. Returns structured JSON with error type, missing property details, and specific fix instructions the AI can act on immediately.
- **`visualize_graph(verbose?)`**: ASCII tree preview of the current instance graph. `verbose=True` shows full URIs.
- **`get_graph_data`**: Returns the current graph as JSON (nodes + edges) for programmatic use.
- **`get_raw_triplets`**: Debug tool — returns all triples in the graph as plain text.
- **`prune_islands(min_size?)`**: Removes disconnected entity clusters smaller than `min_size`.
- **`export_graph(filename?)`**: Exports all instance data to a Turtle (`.ttl`) file.

---

## Architecture

```
universal-ontology-mcp/
├── main.py                  # Entry point — runs the FastMCP server
├── requirements.txt
├── settings_template.json   # MCP client configuration template
└── mcp_server/
    ├── engine.py            # Core engine: ontology loading, embedding cache,
    │                        # schema cache, SHACL extraction, semantic search,
    │                        # facet discovery, candidate relation finding
    ├── server.py            # FastMCP tool definitions (all @mcp.tool() handlers)
    └── config.py            # Persona instructions, environment variable defaults
```

### Engine Internals

- **Embedding Cache**: On startup, checks `.embedding_cache.pkl` in the ontology directory. If the cache key (MD5 of all file paths + sizes + mtimes) matches, embeddings are restored from disk — no re-encoding. On cache miss, all embeddings are batch-encoded in a single pass and saved.
- **SHACL Integration**: `_build_schema_cache()` extracts `sh:NodeShape` constraints, resolves `sh:node → sh:targetClass` chains for facet discovery, and augments property domain lists from shape definitions.
- **Facet Discovery** (`get_related_facets`): Four strategies in order — (1) direct hasFacet-like property range, (2) `rdfs:subPropertyOf` chain, (3) SHACL `sh:node` resolution, (4) shape path name heuristic.

---

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your ontology directory (the engine will recursively load all recognized files):
   ```bash
   export ONTOLOGY_DIR="/path/to/your/ontology/folder"
   ```

---

## MCP Configuration

Add this configuration to your MCP-compatible client (e.g., Claude Desktop, Claude Code, Gemini CLI).

### Claude Desktop / Claude Code / Gemini CLI

```json
{
  "mcpServers": {
    "universal-ontology-mcp": {
      "command": "python",
      "args": ["/absolute/path/to/universal-ontology-mcp/main.py"],
      "env": {
        "ONTOLOGY_DIR": "/absolute/path/to/your/ontology/folder",
        "EMBEDDING_MODE": "remote",
        "EMBEDDING_API_URL": "http://192.168.100.2:8082/v1/embeddings",
        "EMBEDDING_API_KEY": "",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "PROPERTY_RECOMMENDER_BASE_URL": "https://api.openai.com/v1",
        "PROPERTY_RECOMMENDER_API_KEY": "your-api-key",
        "PROPERTY_RECOMMENDER_MODEL": "your-openai-compatible-model"
      }
    }
  }
}
```

### OpenCode

```json
{
  "mcp": {
    "universal-ontology-mcp": {
      "type": "local",
      "command": ["python", "/absolute/path/to/universal-ontology-mcp/main.py"],
      "environment": {
        "ONTOLOGY_DIR": "/absolute/path/to/your/ontology/folder",
        "EMBEDDING_MODE": "remote",
        "EMBEDDING_API_URL": "http://192.168.100.2:8082/v1/embeddings",
        "EMBEDDING_API_KEY": "",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "PROPERTY_RECOMMENDER_BASE_URL": "https://api.openai.com/v1",
        "PROPERTY_RECOMMENDER_API_KEY": "your-api-key",
        "PROPERTY_RECOMMENDER_MODEL": "your-openai-compatible-model"
      }
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ONTOLOGY_DIR` | `./ontology` | Path to directory containing ontology files (recursive) |
| `EMBEDDING_MODE` | `local` | `local` uses SentenceTransformer in-process, `remote` uses an OpenAI-compatible embedding endpoint |
| `EMBEDDING_API_URL` | `http://192.168.100.2:8082/v1/embeddings` | Remote embedding endpoint used when `EMBEDDING_MODE=remote` |
| `EMBEDDING_API_KEY` | `""` | Optional API key for the remote embedding endpoint |
| `EMBEDDING_TIMEOUT_SECONDS` | `60` | Timeout for remote embedding requests |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for semantic search |
| `PROPERTY_RECOMMENDER_BASE_URL` | `https://api.openai.com/v1` | Base URL for the OpenAI-compatible API used by `recommend_property` |
| `PROPERTY_RECOMMENDER_API_KEY` | `""` | API key for the property recommender LLM. Leave empty only if your endpoint does not require auth |
| `PROPERTY_RECOMMENDER_MODEL` | `""` | Model name used by `recommend_property`. If unset, the tool returns `isSuccess: false` |
| `PROPERTY_RECOMMENDER_TIMEOUT_SECONDS` | `30` | Request timeout for the property recommender LLM |

---

## License

This project is licensed under the [MIT License](LICENSE).
