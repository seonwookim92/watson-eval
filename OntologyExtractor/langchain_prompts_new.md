# LangGraph composition and prompts (current code)

## 1) LangGraph composition

File: `extractor/pipeline.py`

- State: `PipelineState(TypedDict)`
  - `error: Optional[str]`
- Graph entry point: `pre_processing`
- Nodes registered:
  - `pre_processing`
  - `chunking`
  - `paraphrasing`
  - `triplet_extraction`
  - `ioc_detection`
  - `type_matching`
  - `internal_entity_resolution`
  - `existing_entity_resolution`
  - `data_insert`
- Edges:
  - `pre_processing -> chunking`
  - `chunking -> paraphrasing`
  - `paraphrasing -> triplet_extraction`
  - `triplet_extraction -> ioc_detection`
  - `ioc_detection -> type_matching`
  - `type_matching -> internal_entity_resolution`
  - `internal_entity_resolution -> existing_entity_resolution`
  - `existing_entity_resolution -> data_insert`
  - `data_insert -> END`

Mermaid:
```mermaid
graph LR
    pre_processing --> chunking --> paraphrasing --> triplet_extraction --> ioc_detection --> type_matching --> internal_entity_resolution --> existing_entity_resolution --> data_insert --> END
```

Run is invoked by:
```python
graph = self._build_graph()
graph.invoke({"error": None})
```

## 2) Exact prompts used per node (verbatim prompt text)

All prompts are built by functions in `extractor/prompts.py`.

### Node 2: Chunking (`node_chunking`)

`chunk_prompt(block)`
```text
You are a semantic text splitter. Your goal is to find the best point to end a
text chunk so that context is preserved (e.g., at the end of a paragraph or a
complete thought).

I will provide a block of text. Identify the most appropriate sentence to END
the chunk within this block.
Return ONLY the exact text of that last sentence. DO NOT PARAPHRASE.

TEXT BLOCK:
{block}
```

Called at:
- `self.llm.chat_text(chunk_prompt(block))`

### Node 3: Paraphrasing (`node_paraphrasing`)

`paraphrase_prompt(chunk)`
```text
You are an expert CTI (Cyber Threat Intelligence) analyst.
Rewrite the following text in English according to these rules:
1. Remove all figurative language, filler phrases, and rhetorical expressions.
2. Describe all facts using Subject-Predicate-Object sentence structure.
3. Replace all pronouns, nominal phrases, and anaphoric references with the
   explicit proper nouns or named entities they refer to.
4. Preserve every technical detail from the original without omission.

Original text:
{chunk}
```

`paraphrase_verify_prompt(original, paraphrased)`
```text
You are a strict quality checker for CTI text processing.
Review the following paraphrased text against the original and answer in JSON:

{
  "has_figurative_language": <true/false>,
  "has_pronouns_or_nominal_phrases": <true/false>,
  "missing_technical_details": <true/false>,
  "issues": "<brief description or empty string>"
}

Original:
{original}

Paraphrased:
{paraphrased}
```

`paraphrase_retry_prompt(original, previous, issues)`
```text
You are an expert CTI analyst. A previous paraphrasing attempt failed quality checks.
Improve the paraphrased text based on the failure reasons.

Original text:
{original}

Previous paraphrase attempt:
{previous}

Failure reasons:
{issues}

Apply the same rules:
1. Remove all figurative language and filler phrases.
2. Use Subject-Predicate-Object sentence structure throughout.
3. Replace all pronouns and nominal references with explicit proper nouns or named entities.
4. Preserve every technical detail from the original.
```

Callsites:
- Initial attempt: `self.llm.chat_text(paraphrase_prompt(original))`
- Retry attempt: `self.llm.chat_text(paraphrase_retry_prompt(original, current, issues))`
- Quality check every attempt: `self.llm.chat_json(paraphrase_verify_prompt(original, current))`

### Node 4: Triplet Extraction (`node_triplet_extraction`)

`triplet_prompt(chunk, sentence)`
```text
You are a Knowledge Graph construction expert specializing in CTI (Cyber Threat Intelligence).

Given the context paragraph and the target sentence, extract all Subject-Predicate-Object
(SPO) triplets that represent factual relationships relevant to cybersecurity.

Rules:
- Subject and Object must be named entities or specific technical terms (not pronouns).
- Predicate must be a concise verb phrase describing the relationship.
- Extract only factual, objective relationships. Omit opinions or speculation.
- Return JSON in the following format:
{
  "triplets": [
    {"subject": "...", "predicate": "...", "object": "..."},
    ...
  ]
}

Context paragraph:
{chunk}

Target sentence:
{sentence}
```

Callsite:
- `self.llm.chat_json(triplet_prompt(chunk_text, sentence))`

### Node 5: IoC Detection & Rearm (`node_ioc_detection`)

`ioc_prompt(candidate, context)`
```text
You are a CTI IoC analyst. Determine if the following string is a genuine IoC
(IoC) such as an IP address, domain, URL, file hash, CVE ID,
email address, or registry key.

If it is an IoC, also provide the rearmed (defanged reversed) version
(e.g., "hxxp" -> "http", "192[.]168[.]1[.]1" -> "192.168.1.1").

Return JSON:
{
  "is_ioc": <true/false>,
  "rearmed_value": "<rearmed string or original if not IoC>"
}

String to evaluate: {candidate}
Context: {context}
```

Callsite:
- `self.llm.chat_json(ioc_prompt(candidates[0], value))`

### Node 6: Triplet Type Matching (`node_type_matching` and helpers)

`type_match_select_prompt(entity, context, candidates)`
```text
You are an ontology expert. Given the entity below, find the most specific
class in the ontology that best describes it.

IMPORTANT: Choose the most specific (leaf-level) class that still accurately
describes the entity. Do NOT select a broader parent class if a more specific
subclass is a better fit.

Entity: {entity}
Context: {context}

Available classes from ontology search:
{candidates_text}

Return JSON:
{
  "class_uri": "<URI from the list above, or empty string if none fit>",
  "class_name": "<name>",
  "confidence": <0.0-1.0>
}
```

`data_property_check_prompt(subject_class_uri, predicate, obj, data_properties)`
```text
You are an ontology expert. Determine whether the following predicate should be
modeled as a DataProperty (literal value on subject node) or an ObjectProperty
(relationship to another entity node).

Subject class: {subject_class_uri}
Predicate (natural language): {predicate}
Object value: {obj}

Available DataProperties for this subject class:
{props_text}

Return JSON:
{
  "is_data_property": <true/false>,
  "property_uri": "<URI if is_data_property, else empty string>",
  "property_name": "<name if is_data_property, else empty string>"
}
```

`predicate_match_select_prompt(subject, subject_class, predicate, obj, obj_class, properties)`
```text
You are an ontology expert. Find the ObjectProperty that best describes
the relationship between the subject entity and the object entity.

Subject: {subject} (Class: {subject_class})
Predicate (natural language): {predicate}
Object: {obj} (Class: {obj_class})

Available ObjectProperties from ontology search:
{props_text}

Return JSON:
{
  "property_uri": "<URI from the list above, or empty string if none fit>",
  "property_name": "<name>",
  "confidence": <0.0-1.0>
}
```

Callsites in Node 6 flow:
- `self.llm.chat_json(type_match_select_prompt(self.source_filename, "CTI threat intelligence report document", candidates))` during report root selection.
- `self.llm.chat_json(type_match_select_prompt(entity, context, candidates))` for subject/object class matching.
- `self.llm.chat_json(type_match_select_prompt(entity, context, subs))` when drilling down subclasses.
- `self.llm.chat_json(data_property_check_prompt(subject_class_uri, predicate, obj, data_props))` for literal/edge decision.
- `self.llm.chat_json(predicate_match_select_prompt(subject, subject_class_name, predicate, obj, obj_class_name, obj_props))` for object-property selection.

### Node 7: Internal Entity Resolution (`node_internal_entity_resolution`)

`entity_resolution_prompt(entity_a, entity_b, class_name)`
```text
You are a CTI knowledge graph expert. Determine if the following two entities
refer to the same real-world entity.

Entity A: {entity_a}
Entity B: {entity_b}
Entity Type: {class_name}

Return JSON:
{
  "is_same": <true/false>,
  "canonical_name": "<more representative name if same, else empty string>",
  "reason": "<brief explanation>"
}
```

Callsite:
- `self.llm.chat_json(entity_resolution_prompt(a, b, class_uri))`

### Node 8: Existing Entity Resolution (`node_existing_entity_resolution`)

`entity_resolution_prompt(entity_a, entity_b, class_name)`
```text
You are a CTI knowledge graph expert. Determine if the following two entities
refer to the same real-world entity.

Entity A: {entity_a}
Entity B: {entity_b}
Entity Type: {class_name}

Return JSON:
{
  "is_same": <true/false>,
  "canonical_name": "<more representative name if same, else empty string>",
  "reason": "<brief explanation>"
}
```

Callsite:
- `self.llm.chat_json(entity_resolution_prompt(name, existing_name, class_uri))`

### Node 9: Data Insert (`node_data_insert`, `_insert_node`, `_insert_edge`)

`node_cypher_prompt(name, class_name, class_uri, filename, properties)`
```text
Generate a Cypher query to MERGE a node in Neo4j with the following details.
Use MERGE to avoid duplicates. Set all properties on creation and update.

Node:
- name: {name}
- entity_type: {class_uri}
- source_document: {filename}
- additional_properties: {json.dumps(properties, ensure_ascii=False)}

Return JSON:
{
  "cypher": "<Cypher query string>"
}
```

`edge_cypher_prompt(subject, subject_class, predicate, predicate_uri, obj, obj_class, sentence, chunk, is_literal)`

If `is_literal == true`:
```text
Generate a Cypher query to add a property on subject node (not a separate node).
Subject:
- name: {subject}
- entity_type: {subject_class}
- property_uri: {predicate_uri}
- object: {obj}
- source_sentence: {sentence}
- source_chunk: {chunk}

Return JSON:
{
  "cypher": "<Cypher query string>"
}
```

If `is_literal == false`:
```text
Generate a Cypher query to create an object-property relationship.

Subject: {subject} ({subject_class})
Predicate: {predicate} ({predicate_uri})
Object: {obj} ({obj_class})
Source sentence: {sentence}
Source chunk: {chunk}

Return JSON:
{
  "cypher": "<Cypher query string>"
}
```

Callsites:
- `self.llm.chat_json(node_cypher_prompt(name, class_name, class_uri, self.source_filename, {}))`
- `self.llm.chat_json(edge_cypher_prompt(..., is_literal=False))`
- `self.llm.chat_json(edge_cypher_prompt(..., is_literal=True))`
