# LangChain Graph + LLM Prompt Inventory

## 1) Graph composition

From `kg_extractor/workflow/graph.py`

- `workflow = StateGraph(ExtractionState)`
- Nodes
  - `preprocess` -> `preprocess_input_node`
  - `chunking` -> `chunking_node`
  - `paraphrasing` -> `paraphrasing_node`
  - `triplet` -> `triplet_node`
  - `uco_type` -> `uco_type_node`
  - `graph_build` -> `graph_build_node`
  - `intra_report_resolution` -> `intra_report_resolution_node`
  - `resolution` -> `entity_resolution_node`
  - `execution` -> `execution_node`
  - `consolidation` -> `consolidation_node`
- Entry point
  - `set_entry_point("preprocess")`
- Directed edges
  - `preprocess -> chunking`
  - `chunking -> paraphrasing`
  - `paraphrasing -> triplet`
  - `triplet -> uco_type`
  - `uco_type -> graph_build`
  - `graph_build -> intra_report_resolution`
  - `intra_report_resolution -> resolution`
  - `resolution -> execution`
  - `execution -> consolidation`
  - `consolidation -> END`
- Build call
  - `return workflow.compile()`

## 2) Prompts used in active nodes

### `preprocess_input_node`
- No direct LLM prompt in this node.

---

### `chunking_node`
- No direct prompt in this node.
- It calls `Chunker.process_file(...)`, which can use `Chunker._semantic_chunking`.
- Prompt used in semantic mode (`_semantic_chunking` in `kg_extractor/core/chunker.py`):

```text
You are a semantic text splitter. Your goal is to find the best point to end a text chunk so that context is preserved (e.g., at the end of a paragraph or a complete thought).

I will provide a block of text. Identify the most appropriate sentence to END the chunk within this block. 
Return ONLY the exact text of that last sentence. DO NOT paraphrase. 

TEXT BLOCK:
{buffer}
```

---

### `paraphrasing_node`
- No prompt directly in node function; all prompts are inside `Paraphraser` helpers.
- Prompt in `Paraphraser.process_chunks` (`prompt_para` in `kg_extractor/core/paraphraser.py`):

```text
You are a paraphraser for knowledge‑graph extraction. ONLY PRINT OUT THE PARAPHRASED TEXT.
Read the input text below and apply the following transformations **in the order given**:
1. Replace every markdown table with one or more plain‑text sentences that convey the same information, **without any markdown characters** (|, –, :, etc.).
2. Remove all markdown formatting (headings, bold/italics, code fences, back‑ticks, bullet lists, etc.) so the result is pure plain text.
3. Replace **every** pronoun AND every vague or underspecified reference with the explicit noun phrase it refers to. This includes:
   - Personal pronouns: he/she/it/they, his/her/its/their, him/her/them.
   - Demonstrative pronouns: this/that/these/those when used without a following noun.
   - Collective/quantifying expressions that do not name the actual referents:
     "both campaigns" → write out the actual names (e.g., "Operation Bran Flakes and Operation Raisin Bran")
     "the former" / "the latter" → replace with the actual named entity
     "each method" / "all three groups" / "the two organizations" → list the actual names
     "such tactics" / "these operations" / "the above techniques" / "the following" → name them explicitly
     "the first" / "the second" when referring back to named items → use the actual name
   Always look back in the text to find the exact noun(s) being referenced and substitute them in full.
4. Delete any rhetorical questions, exclamation marks, and filler phrases such as “in fact”, “as you can see”, “indeed”, etc.; keep only factual statements.
5. Reconstruct the text into **SIMPLE sentences** with a clear **Subject, Predicate, and Object (SPO)** structure.
6. Split any long or complex sentences into multiple simple sentences.
7. Group related simple sentences into logical paragraphs. Each paragraph should contain multiple SPO sentences that describe a single coherent idea or section (scenario, data model, motivations, campaigns, relationships, etc.).
8. Insert a blank line **ONLY** between these logical paragraphs, not between every sentence.
9. Preserve the original ordering of information; do not add or remove content, only rewrite it to satisfy the rules above.
The output must be **plain text only** and must contain **no markdown**, **no pronouns**, and **no underspecified collective references**. Related SPO sentences must be grouped into paragraphs, with blank lines only between paragraphs.
{retry_hint}
{prior_draft_hint}

INPUT TEXT:
{chunk_text}
```

- Prompt in `_ask_reference_replacement` (`kg_extractor/core/paraphraser.py`):

```text
Resolve one reference mention for CTI paraphrasing.
Return ONLY JSON: {"replacement":"..."}.

Rules:
1. replacement must be the exact explicit referent phrase for the marked mention.
2. replacement must be specific and concrete (named entity or precise noun phrase).
3. replacement must not be a pronoun, demonstrative, or vague placeholder.
4. Do not add extra facts.

Original Source Text:
{original_text}

Current Paraphrased Text:
{annotated}
{feedback_line}
```

- Prompt in `_validate_reference_replacement`:

```text
Validate one reference replacement for CTI paraphrasing.
Return ONLY JSON: {"valid": true/false, "reason":"..."}.

Validation checklist:
1. "{replacement}" is the correct referent for "{mention_text}" in context.
2. Meaning remains faithful to the original source text.
3. No new facts are introduced.
4. The rewritten sentence remains semantically natural.

Original Source Text:
{original_text}

Before Replacement:
{before_text}

After Replacement:
{after_text}
```

- Prompt in `_validate_contextual_quality`:

```text
You are a strict but practical CTI paraphrasing validator.
Return ONLY JSON: {"pass": true/false, "reason":"..."}.

Check2/3 policy:
1. Evaluate markdown/table artifacts, demonstratives (this/that/these/those), vague references, and question/exclamation marks with context.
2. Do NOT fail only due to regex-style false positives.
   - "that" as conjunction/relative marker can be valid.
   - "this/that + noun" determiner usage can be valid.
   - punctuation inside filenames/paths/identifiers can be valid.
3. Fail only if remaining issues materially harm clarity for KG extraction.
4. Semantic checks are mandatory: meaning and ordering preserved; simple SPO-friendly sentences; coherent paragraphs; no added facts.

Heuristic detector signals (for your review, not auto-fail):
- demonstratives={hard_report['demonstratives']}
- vague_refs={hard_report['vague_refs']}
- markdown_hits={hard_report['markdown_hits']}
- table_tags={hard_report['table_tags']}
- question_or_exclaim={hard_report['question_or_exclaim']}
- filler_phrases={hard_report['filler_phrases']}

Original Text:
{original_text}

Paraphrased Text:
{candidate_text}
```

---

### `triplet_node`
- System prompt:

```text
You are a Cyber Threat Intelligence (CTI) information extraction specialist.
Your task: extract factual (Subject, Predicate, Object) triplets from the TARGET PASSAGE.

RULES:
1. Extract ONLY from the TARGET PASSAGE — use CONTEXT solely to resolve ambiguous references (e.g., pronouns, 'the group', 'the malware', 'the campaign').
2. Subject and Object must be specific named entities: threat actors, malware families, IP addresses, domains, URLs, file hashes, vulnerabilities (CVE-IDs), campaigns, organizations, tools, or similar concrete CTI entities.
3. Predicate must be a concise verb phrase describing the relationship (e.g., 'uses', 'targets', 'communicates_with', 'exploits', 'downloads', 'attributed_to', 'drops', 'connects_to', 'was_observed_at').
4. Do NOT extract meta-statements about the report itself (e.g., 'report describes', 'section discusses', 'authors note').
5. If the TARGET PASSAGE is too short or contains no extractable CTI relationships, return an empty array.

OUTPUT FORMAT — return ONLY a JSON array, no explanation:
[{"subject": "APT28", "predicate": "uses", "object": "X-Agent"},
 {"subject": "X-Agent", "predicate": "connects_to", "object": "185.220.101.5"}]
```

- User prompt template:

```text
[CONTEXT — use only to resolve ambiguous references, do NOT extract from here]
{parent_chunk}

[TARGET PASSAGE — extract all CTI triplets from this text]
{text}
```

---

### `uco_type_node`
- In `_infer_search_queries` (helper used by this node):

System prompt:

```text
You are a cybersecurity ontology expert.
For each entity extracted from a CTI report, write a concise SEMANTIC TYPE DESCRIPTION (2–6 words) that captures WHAT KIND OF THING the entity is.
This description will be used to search a cybersecurity ontology, so it must describe the category, not repeat the name.

Rules:
- Describe the TYPE, not the name.  '10.1.1.1' → 'IPv4 address'.
- Use domain-specific terms: 'malware executable', 'threat actor group',
  'Windows registry key', 'network traffic log', 'file hash', 'URL', etc.
- If the name itself is a well-known concept (PowerShell, Active Directory),
  describe what TYPE of software/service it is.
- Return ONLY a JSON object mapping entity name to its type description.

Example output:
{"10.1.1.1": "IPv4 address",
 "svchostext": "malicious Windows executable",
 "mkd.conf": "configuration file",
 "APT28": "threat actor group",
 "PowerShell": "Windows scripting shell"}
```

User prompt template:

```text
Produce the semantic type description for each entity:

{entity blocks}

Return JSON: {"entity_name": "type description", ...}
```

- In entity resolution batch step (main in node):

System prompt:

```text
You are a UCO (Unified Cyber Ontology) 1.4.0 ontology expert.
For each entity, select the single most specific UCO class URI from the provided candidates.

RULES:
1. Choose the most specific (deepest in the class hierarchy) applicable UCO class.
2. Use the entity name and its relationship context triplets to inform your decision.
3. If no candidate is a good fit, use:
   'https://ontology.unifiedcyberontology.org/uco/core/UcoObject'
4. Output ONLY a JSON object mapping entity names to UCO class URIs — no explanation.

Example output:
{"APT28": "https://ontology.unifiedcyberontology.org/uco/identity/Organization",
 "X-Agent": "https://ontology.unifiedcyberontology.org/uco/malware/MaliciousCode"}
```

User prompt template:

```text
Assign the best UCO class URI to each entity below.

{entity blocks}

Return ONLY a JSON object: {"entity_name": "class_uri", ...}
```

---

### `graph_build_node`
- System prompt:

```text
You are a UCO (Unified Cyber Ontology) 1.4.0 ontology expert.
For each CTI relationship predicate, select the single most appropriate UCO property URI from the provided candidates.

RULES:
1. Choose the property whose semantic meaning best matches the predicate in the given entity-type context.
2. If no candidate is a reasonable fit, return an empty string "" for that predicate.
3. Output ONLY a JSON object mapping predicate text to URI (or "").

Example:
{"uses": "https://ontology.unifiedcyberontology.org/uco/core/object",
 "targets": "https://ontology.unifiedcyberontology.org/uco/action/object",
 "unknown_pred": ""}
```

- User prompt template:

```text
Assign the best UCO property URI to each predicate below.

{predicate blocks}

Return ONLY a JSON object: {"predicate": "uri_or_empty", ...}
```

---

### `intra_report_resolution_node`
- Prompt used when comparing two candidates (no system message):

```text
Determine if these two entities from the SAME CTI report are the EXACT SAME real-world entity.
Answer ONLY YES or NO.

Entity A:
Type: {ia["type"]}
Name: {ia["name"]}
Description: {ia["description"]}

Entity B:
Type: {ib["type"]}
Name: {ib["name"]}
Description: {ib["description"]}
```

---

### `entity_resolution_node`
- Prompt used when comparing extracted node vs database candidate (no system message):

```text
Compare the following two entities of type '{item["entity_type"]}' and determine if they represent the EXACT SAME real-world entity.

Entity A (Extracted):
Name: {item["name"]}
Description: {item["description"]}

Entity B (Existing in Database):
Name: {candidate["name"]}
Description: {candidate["description"]}

Are they the same entity? Answer ONLY 'YES' or 'NO'.
```

---

### `execution_node`
- No direct LLM prompt in this node.

---

### `consolidation_node`
- No direct LLM prompt in node function.
- It calls `Consolidator.consolidate(...)` (`kg_extractor/core/consolidator.py`).

Prompts used there:

- `system_prompt`:

```text
You are an Ontology Linker specializing in Unified Cyber Ontology (UCO) 1.4.0. Search the Neo4j database and merge nodes/edges that are identical or semantically equivalent.
The database contains pre-populated MITRE ATT&CK entities (marked with `mitre_id` and specific STIX IDs, mapped to UCO classes).

Your primary goal is to:
1. Anchor new extractions to existing MITRE entities.
2. Consolidate new extractions that refer to the same entity.

WORKFLOW:
STEP 1: List all named nodes: `MATCH (n) WHERE n.name IS NOT NULL AND n.is_report_root IS NULL RETURN labels(n) AS lbl, n.name AS name, n.mitre_id as mitre_id, n.id as stix_id ORDER BY name`
STEP 2: Identify merging candidates.
STEP 3: MERGE nodes using this EXACT Cypher structure (NO APOC):
   MATCH (ref {id: 'stix-id-or-mitre-id'}), (new {name: 'fuzzy-name'})
   WHERE id(ref) <> id(new)
   SET ref += properties(new)
   WITH ref, new
   MATCH (new)-[r]->(x) WHERE id(x) <> id(ref)
   MERGE (ref)-[new_r:REL_TYPE_HERE]->(x) SET new_r = r
   WITH ref, new
   MATCH (x)-[r]->(new) WHERE id(x) <> id(ref)
   MERGE (x)-[new_r:REL_TYPE_HERE]->(ref) SET new_r = r
   DETACH DELETE new

STRICT RULES:
1. DO NOT USE APOC.
2. ALWAYS prioritize merging INTO a node that has a `mitre_id`.
3. For UCO, many entities have Facets. If merging nodes, ensure all properties are preserved.
4. CRITICAL: NEVER merge, modify, or delete any node that has the property `is_report_root = true`.
   These are Report anchor nodes. They must remain intact with all their edges.
```

- Initial user message:

```text
Start by listing all named nodes in the database, then identify and merge exact duplicates and semantic equivalents following the workflow above.
```

- Continuation message in loop:

```text
Continue consolidation. Use execute_neo4j_query tool calls, or reply COMPLETED when done.
```

- Validation retry message:

```text
Validation failed after your consolidation attempt. Issues found:
{issue_summary}
Please resolve these remaining issues. REMEMBER: NO APOC, NO exists().
```

---

## 3) Legacy prompts (not part of the active `create_workflow` graph)

These functions exist in `kg_extractor/workflow/nodes.py` but are not wired into `create_workflow()`:
- `extraction_node_old`
- `extraction_node_cyber_old`
- `extraction_node_monolithic`
- `chunking_node_old`

They contain their own LLM prompts (e.g., agentic MCP tool-call prompts and MCP cyber extraction prompts).
