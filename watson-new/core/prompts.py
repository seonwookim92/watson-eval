"""Prompt builder functions for each pipeline step.

All functions return plain strings (prompts) to be sent to the LLM.
"""
from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List


def chunk_prompt(block: str) -> str:
    return textwrap.dedent(
        f"""
        You are a semantic text splitter. Your goal is to find the best point to end a
        text chunk so that context is preserved (e.g., at the end of a paragraph or a
        complete thought).

        I will provide a block of text. Identify the most appropriate sentence to END
        the chunk within this block.
        - Return one complete sentence only.
        - The sentence must appear verbatim in the text block.
        - The sentence must end with sentence-final punctuation.
        - Return JSON only in the following format:
          {{
            "last_sentence": "<exact sentence>"
          }}

        TEXT BLOCK:
        {block}
        """
    ).strip()


def paraphrase_stage1_prompt(chunk: str) -> str:
    return textwrap.dedent(
        f"""
        You are editing CTI text for knowledge-graph extraction.
        ONLY PRINT THE REWRITTEN TEXT.

        Rewrite the text below with these rules:
        1. Remove only flowery language, rhetorical phrasing, quotation-like narration, and clearly non-technical filler.
        2. Keep the technical meaning unchanged.
        3. Preserve the original order of information.
        4. Do not resolve pronouns yet unless required to keep grammar valid.
        5. Do not add new facts.
        6. Preserve relation structure. Do not compress multi-step relations into fewer relations.
        7. Preserve aggregate, category, and intermediate concepts when they are explicitly stated.
        8. Preserve quantity, grouping, causality, aliasing, disclosure/time, inclusion, and example markers.
        9. Preserve category/example structures such as "payloads including X" and "ransomware such as Y".
        10. Do not merge a category and its example into one span unless the original text does so.

        Text:
        {chunk}
        """
    ).strip()


def paraphrase_stage1_verify_prompt(original: str, rewritten: str) -> str:
    return textwrap.dedent(
        f"""
        You are a strict quality checker for CTI text processing.
        Compare the rewritten text with the original text and answer in JSON.

        {{
          "has_decorative_language": <true/false>,
          "technical_content_preserved": <true/false>,
          "relation_count_preserved": <true/false>,
          "causal_structure_preserved": <true/false>,
          "category_instance_structure_preserved": <true/false>,
          "alias_and_time_markers_preserved": <true/false>,
          "issues": "<brief description or empty string>"
        }}

        Original:
        {original}

        Rewritten:
        {rewritten}
        """
    ).strip()


def paraphrase_stage1_retry_prompt(original: str, previous: str, issues: str) -> str:
    return textwrap.dedent(
        f"""
        You are an expert CTI analyst. A previous stage-1 rewrite failed quality checks.
        Improve the rewritten text based on the failure reasons.

        Original text:
        {original}

        Previous rewritten text:
        {previous}

        Failure reasons:
        {issues}

        Rewrite again with these rules:
        1. Remove only decorative language, quotation-like narration, and clearly non-technical filler.
        2. Keep the technical meaning unchanged.
        3. Preserve the original order of information.
        4. Do not add new facts.
        5. Preserve relation count and multi-step relation structure.
        6. Preserve aggregate, category, intermediate, and example entities.
        7. Preserve alias, disclosure, timing, causality, inclusion, and quantity markers.
        ONLY PRINT THE REWRITTEN TEXT.
        """
    ).strip()


def paraphrase_stage2_prompt(original: str, annotated: str, pronoun_guide: str) -> str:
    return textwrap.dedent(
        f"""
        You are resolving pronouns in CTI text.
        The text contains annotated pronouns such as <It:1>.
        For each numbered pronoun, identify the most explicit proper noun or named noun phrase
        from the original text that can replace that exact pronoun occurrence.

        Return JSON only:
        {{
          "replacements": [
            {{"id": 1, "pronoun": "It", "replacement": "Cosmic Duke"}},
            {{"id": 2, "pronoun": "They", "replacement": "APT28 operators"}}
          ]
        }}

        Original text:
        {original}

        Annotated text:
        {annotated}

        Pronoun inventory:
        {pronoun_guide}
        """
    ).strip()


def paraphrase_stage2_retry_prompt(
    original: str,
    annotated: str,
    pronoun_guide: str,
    previous_replacements: str,
    issues: str,
) -> str:
    return textwrap.dedent(
        f"""
        You are resolving pronouns in CTI text. The previous answer was not usable.
        Try again and fix the issues.

        Return JSON only:
        {{
          "replacements": [
            {{"id": 1, "pronoun": "It", "replacement": "Cosmic Duke"}}
          ]
        }}

        Original text:
        {original}

        Annotated text:
        {annotated}

        Pronoun inventory:
        {pronoun_guide}

        Previous replacements:
        {previous_replacements}

        Issues:
        {issues}
        """
    ).strip()


def paraphrase_stage2_verify_prompt(original: str, replaced: str) -> str:
    return textwrap.dedent(
        f"""
        You are a strict CTI equivalence checker.
        Compare the original paragraph and the changed paragraph.
        Decide whether both paragraphs are technically the same.

        Return JSON only:
        {{
          "equivalent": <true/false>,
          "issues": "<brief description or empty string>"
        }}

        Original paragraph:
        {original}

        Changed paragraph:
        {replaced}
        """
    ).strip()


def paraphrase_stage3_prompt(chunk: str) -> str:
    return textwrap.dedent(
        f"""
        You are rewriting CTI text for knowledge-graph extraction.
        ONLY PRINT THE REWRITTEN TEXT.

        Replace every remaining pronoun and every vague or underspecified reference with the
        explicit noun phrase it refers to. This includes:
        - personal pronouns
        - demonstrative pronouns used without a following noun
        - "the former", "the latter", "the first", "the second"
        - "both campaigns", "all three groups", "the two organizations"
        - "such tactics", "these operations", "the above techniques", "the following"

        Rules:
        1. Use the exact named entities or noun phrases from the text when possible.
        2. Preserve technical meaning and factual content.
        3. Preserve the original order of information.
        4. Do not add new facts.

        Text:
        {chunk}
        """
    ).strip()


def paraphrase_stage3_retry_prompt(original: str, previous: str, issues: str) -> str:
    return textwrap.dedent(
        f"""
        You are rewriting CTI text for knowledge-graph extraction.
        The previous rewrite still had underspecified references or changed technical meaning.
        Fix the issues and ONLY PRINT THE REWRITTEN TEXT.

        Original text:
        {original}

        Previous rewrite:
        {previous}

        Issues:
        {issues}
        """
    ).strip()


def paraphrase_stage3_verify_prompt(original: str, rewritten: str) -> str:
    return textwrap.dedent(
        f"""
        You are a strict CTI quality checker.
        Compare the rewritten text with the original text.

        Return JSON only:
        {{
          "has_underspecified_references": <true/false>,
          "technical_content_preserved": <true/false>,
          "issues": "<brief description or empty string>"
        }}

        Original:
        {original}

        Rewritten:
        {rewritten}
        """
    ).strip()


def paraphrase_stage4_prompt(chunk: str) -> str:
    return textwrap.dedent(
        f"""
        You are rewriting CTI text into relation-friendly sentences for knowledge-graph extraction.
        ONLY PRINT THE REWRITTEN TEXT.

        Rules:
        1. Reconstruct the text into simple sentences with a clear Subject, Predicate, and Object structure.
        2. Each sentence must express only one relation.
        3. Preserve technical meaning and factual content.
        4. Preserve the original order of information.
        5. Keep explicit noun phrases. Do not reintroduce pronouns.

        Text:
        {chunk}
        """
    ).strip()


def paraphrase_stage4_retry_prompt(original: str, previous: str, issues: str) -> str:
    return textwrap.dedent(
        f"""
        You are rewriting CTI text into simple SPO sentences.
        The previous rewrite was not acceptable. Fix the issues and ONLY PRINT THE REWRITTEN TEXT.

        Original text:
        {original}

        Previous rewrite:
        {previous}

        Issues:
        {issues}
        """
    ).strip()


def paraphrase_stage4_verify_prompt(original: str, rewritten: str) -> str:
    return textwrap.dedent(
        f"""
        You are a strict CTI quality checker.
        Compare the rewritten text with the original text.

        Return JSON only:
        {{
          "clear_spo": <true/false>,
          "one_relation_per_sentence": <true/false>,
          "technical_content_preserved": <true/false>,
          "issues": "<brief description or empty string>"
        }}

        Original:
        {original}

        Rewritten:
        {rewritten}
        """
    ).strip()


def entity_extraction_prompt(chunk_name: str, document: str) -> str:
    return textwrap.dedent(
        f"""
        You are a Cyber Threat Intelligence analyst preparing inputs for knowledge-graph extraction.

        Read the document chunk below and extract the entities that are important for Cyber Threat Intelligence.
        Focus on entities such as threat actors, malware, tools, campaigns, vulnerabilities, targets,
        organizations, infrastructure, files, registries, and other concrete CTI-relevant entities.

        For each entity, provide:
        - source_chunk: the exact chunk identifier "{chunk_name}"
        - name: the entity name exactly as it appears in the document when possible
        - description: one sentence explaining why the entity matters in this chunk

        Rules:
        - Extract only entities that are important for CTI understanding.
        - Do not extract pronouns, vague references, or generic terms with no concrete referent.
        - Keep descriptions factual, concise, and grounded in the document.
        - If no relevant entities exist, return an empty list.
        - Return JSON only in the following format:
        {{
          "entities": [
            {{
              "source_chunk": "{chunk_name}",
              "name": "Entity Name",
              "description": "One-sentence description."
            }}
          ]
        }}

        Document chunk:
        {document}
        """
    ).strip()


def entity_inventory_from_triplets_prompt(
    chunk_name: str,
    document: str,
    triplets: List[Dict[str, str]],
) -> str:
    triplets_text = "\n".join(
        f'- {item["subject"]} | {item["predicate"]} | {item["object"]}'
        for item in triplets
    ) if triplets else "(no extracted triplets)"
    return textwrap.dedent(
        f"""
        You are preparing entity nodes for ontology type matching in a CTI knowledge graph.

        Given the document chunk and the extracted triplets from that chunk, build an entity
        inventory for meaningful subjects and objects appearing in the triplets.

        For each entity, provide:
        - source_chunk: the exact chunk identifier "{chunk_name}"
        - name: the entity name, preserving the wording from the document when possible
        - description: one factual sentence that explains what the entity is in this chunk,
          in a way that helps ontology type matching

        Rules:
        - Include every meaningful subject or object from the triplets unless it is a pronoun,
          malformed fragment, or obvious extraction noise.
        - Do not omit a term just because it looks generic if it functions as a graph node in the triplets.
        - Merge only obvious duplicates or spelling variants that clearly refer to the same entity.
        - Prefer the more explicit span when multiple names refer to the same entity.
        - Keep descriptions concise, factual, and grounded in the document.
        - The description should say what the entity is in context, not just why it matters.
        - If no valid entities can be formed, return an empty list.

        Return JSON only in the following format:
        {{
          "entities": [
            {{
              "source_chunk": "{chunk_name}",
              "name": "Entity Name",
              "description": "One-sentence description."
            }}
          ]
        }}

        Document chunk:
        {document}

        Extracted triplets:
        {triplets_text}
        """
    ).strip()


def triplet_prompt(
    chunk: str,
    sentence: str,
    chunk_entities: List[Dict[str, str]],
    sentence_entities: List[Dict[str, str]],
) -> str:
    chunk_entities_text = "\n".join(
        f'- {item["name"]}: {item["description"]}' for item in chunk_entities
    ) if chunk_entities else "(no chunk-level entity inventory)"
    sentence_entities_text = "\n".join(
        f'- {item["name"]}: {item["description"]}' for item in sentence_entities
    ) if sentence_entities else "(no entity matched directly from the sentence)"
    return textwrap.dedent(
        f"""
        You are a Knowledge Graph construction expert specializing in CTI (Cyber Threat Intelligence).

        Given the context paragraph and the target sentence, extract all Subject-Predicate-Object
        (SPO) triplets that represent factual relationships relevant to cybersecurity.

        Rules:
        - Extract ONLY from the Target Sentence - use Context paragraph solely to resolve ambiguous references (e.g., pronouns, 'the group', 'the malware', 'the campaign')
        - First identify the atomic factual statements explicitly present in the target sentence, then convert them into triplets.
        - Subject and Object must be named entities or specific technical terms (not pronouns).
        - Extract all explicit relations in the sentence, including aggregate, category, member, and example relations when they are directly supported.
        - Preserve the semantic structure of the sentence. Do not compress a multi-step relation into a shorter paraphrased relation.
        - Keep intermediate concepts if they are explicitly stated in the sentence or are required to preserve the original relation structure.
        - Use the entity inventory below as optional grounding context, not as a hard constraint.
        - If the sentence contains a valid CTI subject or object that is not in the inventory, extract it anyway.
        - If a subject or object is only a partial span of a listed entity, expand it to the full entity name when that preserves the original meaning.
        - Use subject and object spans from the sentence as faithfully as possible. Prefer the most specific valid span that still preserves the sentence meaning.
        - Predicate must be a concise verb phrase describing the relationship.
        - Extract only factual, objective relationships. Omit opinions or speculation.
        - Prefer the original wording of the relation when possible. Do not replace a relation with a looser summary if the sentence states a more specific one.
        - When a sentence expresses a category followed by enumerated members or identifiers, extract the category-to-member relations explicitly.
        - When a sentence contains patterns such as "X including Y" or "X such as Y", preserve that structure. Prefer the instance span Y for the instance-level relation, and do not keep "X such as Y" as one merged entity span unless it is a fixed name.
        - Do NOT extract meta-statements about the report itself (e.g., 'report describes', 'section discusses', 'authors note').
        - Do NOT extract from figure descriptions, image descriptions, captions, legends, or visual-layout explanations.
        - Do NOT extract from appendix entries, sample tables, JSON-like rows, metadata blocks, or list-like inventory records.
        - Do NOT extract from corrupted OCR fragments, truncated strings, malformed values, or obviously broken sentences.
        - Do NOT extract if the sentence contains unresolved context-dependent references such as "this server", "this image", "the above", "the following", "it", or "they" and the exact referent cannot be resolved with high confidence from the context paragraph.
        - If the target sentence is too short or contains no extractable CTI relationships, return {{"triplets": []}}.
        - Return JSON in the following format:
        {{
          "triplets": [
            {{"subject": "...", "predicate": "...", "object": "..."}},
            ...
          ]
        }}

        Context paragraph:
        {chunk}

        Chunk-level entity inventory:
        {chunk_entities_text}

        Entities matched in the target sentence:
        {sentence_entities_text}

        Target sentence:
        {sentence}
        """
    ).strip()


def ioc_prompt(candidate: str, context: str, detected_type: str = "") -> str:
    return textwrap.dedent(
        f"""
        You are a CTI IoC analyst. Determine if the following candidate string is a genuine,
        complete IoC (Indicator of Compromise) such as an IP address, domain, URL, file hash,
        CVE ID, email address, or registry key.

        Validation rules — apply ALL before deciding:
        1. The candidate must be a COMPLETE, STANDALONE IoC value, not a fragment accidentally
           extracted from a larger malformed or non-IoC string.
           Example: if the full context contains "188.116.1116.32.16.32.164" (invalid — one octet
           is 1116), any IP-like sub-string extracted from it is NOT a valid IoC.
        2. IP addresses: every octet must be in the range 0–255. If any octet is out of range
           in the surrounding context, set is_ioc to false even if the candidate itself looks valid.
        3. Domains, URLs, hashes, CVEs: must appear correctly and completely in the context,
           not as a partial or corrupted match.
        4. If you are not confident the candidate is a real, standalone IoC, set is_ioc to false.

        If it is a genuine IoC, also provide the rearmed (defanged-reversed) version
        (e.g., "hxxp" -> "http", "192[.]168[.]1[.]1" -> "192.168.1.1").

        Return JSON:
        {{
          "is_ioc": <true/false>,
          "rearmed_value": "<rearmed string or original if not IoC>",
          "ioc_type": "<confirmed IoC type such as ip4, fqdn, url, md5, sha1, sha256, email, cve, ttp, arn, or empty string>"
        }}

        Detected IoC type from iocsearcher: {detected_type or "(unknown)"}
        Candidate string: {candidate}
        Full context: {context}
        """
    ).strip()


def entity_pre_classification_prompt(entity: str, context: str) -> str:
    """Returns a short ontology-friendly concept label for the entity.

    The label is used as a search query against the ontology class index,
    so it should be a generic 2-4 word concept rather than the entity name.
    """
    return textwrap.dedent(
        f"""
        You are a CTI (Cyber Threat Intelligence) entity classifier.
        Given the entity name and its context, return a short concept label (2-4 words)
        that best describes the TYPE of thing this entity is.

        This label will be used as a semantic search query against an ontology class index.
        Prefer generic, ontology-friendly terms — NOT the entity name itself.

        Examples (entity → concept label):
        - "Predator spyware"                      → "malware spyware"
        - "Apple Inc."                            → "organization company"
        - "CVE-2023-41991"                        → "vulnerability CVE identifier"
        - "iOS 16.7"                              → "operating system software"
        - "192.168.1.1"                           → "IP address network indicator"
        - "Google's Threat Analysis Group (TAG)"  → "organization security team"
        - "exploit chain"                         → "attack pattern exploit"
        - "phishing email"                        → "attack pattern phishing message"
        - "registry key"                          → "file system registry artifact"
        - "SHA256 hash"                           → "file hash indicator"
        - "Cobalt Strike"                         → "malware tool"
        - "Microsoft Exchange"                    → "software application"
        - "threat actor group"                    → "threat actor group"

        Return JSON only:
        {{
          "concept": "<short concept label, 2-4 words>",
          "reason": "<one sentence explanation>"
        }}

        Entity: {entity}
        Context: {context}
        """
    ).strip()


def type_match_select_prompt(entity: str, context: str, candidates: List[Dict[str, str]]) -> str:
    candidates_text = "\n".join(
        f"- {c['name']} ({c['uri']})" for c in candidates
    ) if candidates else "(no candidates found)"
    return textwrap.dedent(
        f"""
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
        {{
          "class_uri": "<URI from the list above, or empty string if none fit>",
          "class_name": "<name>",
          "confidence": <0.0-1.0>
        }}
        """
    ).strip()


def data_property_check_prompt(
    subject_class_uri: str,
    predicate: str,
    obj: str,
    data_properties: List[Dict[str, str]],
) -> str:
    props_text = "\n".join(
        f"- {p['name']} ({p['uri']})" for p in data_properties
    ) if data_properties else "(none found)"
    return textwrap.dedent(
        f"""
        You are an ontology expert. Determine whether the following predicate should be
        modeled as a DataProperty (literal value on subject node) or an ObjectProperty
        (relationship to another entity node).

        Subject class: {subject_class_uri}
        Predicate (natural language): {predicate}
        Object value: {obj}

        Available DataProperties for this subject class:
        {props_text}

        Return JSON:
        {{
          "is_data_property": <true/false>,
          "property_uri": "<URI if is_data_property, else empty string>",
          "property_name": "<name if is_data_property, else empty string>"
        }}
        """
    ).strip()


def predicate_match_select_prompt(
    subject: str,
    subject_class: str,
    predicate: str,
    obj: str,
    obj_class: str,
    properties: List[Dict[str, str]],
) -> str:
    props_text = "\n".join(
        f"- {p['name']} ({p['uri']})" for p in properties
    ) if properties else "(none found)"
    return textwrap.dedent(
        f"""
        You are an ontology expert. Find the ObjectProperty that best describes
        the relationship between the subject entity and the object entity.

        Subject: {subject} (Class: {subject_class})
        Predicate (natural language): {predicate}
        Object: {obj} (Class: {obj_class})

        Available ObjectProperties from ontology search:
        {props_text}

        Return JSON:
        {{
          "property_uri": "<URI from the list above, or empty string if none fit>",
          "property_name": "<name>",
          "confidence": <0.0-1.0>
        }}
        """
    ).strip()


def predicate_query_expansion_prompt(
    subject: str,
    predicate: str,
    obj: str,
    subject_class_uri: str = "",
    object_class_uri: str = "",
) -> str:
    return textwrap.dedent(
        f"""
        You are helping with ontology property retrieval.
        Generate up to 5 short English query variants for the predicate below so they are easier
        to match against ontology property labels.

        Rules:
        - Preserve the original relationship meaning.
        - Do NOT reverse direction.
        - Prefer short ontology-friendly verb phrases.
        - Each query should express one concept only.
        - Do not add explanations.

        Return JSON:
        {{
          "queries": ["<query1>", "<query2>", "<query3>"]
        }}

        Subject: {subject}
        Subject class URI: {subject_class_uri or "unknown"}
        Predicate: {predicate}
        Object: {obj}
        Object class URI: {object_class_uri or "unknown"}
        """
    ).strip()


def entity_resolution_prompt(entity_a: str, entity_b: str, class_name: str) -> str:
    return textwrap.dedent(
        f"""
        You are a CTI knowledge graph expert. Determine if the following two entities
        refer to the same real-world entity.

        Entity A: {entity_a}
        Entity B: {entity_b}
        Entity Type: {class_name}

        Return JSON:
        {{
          "is_same": <true/false>,
          "canonical_name": "<more representative name if same, else empty string>",
          "reason": "<brief explanation>"
        }}
        """
    ).strip()


def class_resolution_agent_prompt(
    entity: str,
    context: str,
    transcript: str,
    remaining_calls: int,
    force_finish: bool = False,
) -> str:
    finish_rule = (
        "You MUST finish now. Do not request another tool call."
        if force_finish else
        "You may either call one MCP tool or finish with the final answer."
    )
    return textwrap.dedent(
        f"""
        You are an ontology expert navigating an ontology via MCP tools.

        Requirements:
        - You have full control over ontology exploration.
        - Do NOT assume the first search result is correct.
        - You may explore multiple branches.
        - If a candidate class seems plausible, inspect subclasses before finalizing.
        - If subclasses do not fit, you may return to a broader parent or explore a different branch.
        - Prefer the most specific accurate class, but if no specific subclass fits, return the best broader class.
        - Only use these tools: search_classes, list_root_classes, list_subclasses, get_class_hierarchy, get_class_details, list_available_facets, drill_into_classes.
        - {finish_rule}

        Tool argument schema:
        - search_classes: {{"query": "<text query>"}}
        - list_root_classes: {{}}
        - list_subclasses: {{"class_uri": "<class URI>"}}
        - get_class_hierarchy: {{"class_uri": "<class URI>"}}
        - get_class_details: {{"class_uri": "<class URI>"}}
        - list_available_facets: {{"class_uri": "<class URI>"}}
        - drill_into_classes: {{"query": "<concept label>", "class_uri": "<URI to expand, or omit to start from roots>"}}

        When to use drill_into_classes:
        - Use it when search_classes scores are all below 0.5 or the top result doesn't clearly match.
        - Start with only `query` (no class_uri) to see root classes and their descriptions.
        - Then call again with the chosen class_uri to go one level deeper.
        - It shows raw class names and descriptions so YOU decide which branch to follow.

        Never invent argument names. In particular, NEVER use:
        - parent_class
        - parent_uri
        - uri
        - parent_class_uri

        Tool-call self-check before every tool call:
        1. The tool name is allowed.
        2. The arguments exactly match the schema above.
        3. Every required argument is present.
        4. Do not repeat a previously failed call with the same tool and arguments.

        Exploration policy:
        - Do not finish immediately after the first plausible result if it is broad, generic, or low-confidence.
        - Before finishing, do the following when applicable:
          1. search for the entity name directly
          2. inspect the strongest plausible candidate
          3. inspect subclasses of that candidate if any may exist
          4. compare at least one alternative branch if the first candidate is generic or uncertain
        - If search_classes scores are uniformly low (< 0.5), switch to drill_into_classes to navigate hierarchically.
        - If a previous tool call failed with a validation error, treat that as evidence that the argument schema was wrong and correct it before continuing.
        - Repeating the same invalid call is a reasoning failure.

        Tool transcript so far:
        {transcript}

        Return JSON in one of these forms:

        Tool call:
        {{
          "action": "call_tool",
          "tool": "<tool name>",
          "arguments": {{ ... }},
          "reason": "<why this tool call is needed>"
        }}

        Final answer:
        {{
          "action": "finish",
          "class_uri": "<URI or empty string>",
          "class_name": "<name or empty string>",
          "confidence": <0.0-1.0>,
          "reason": "<why this is the best stopping point, including why the strongest alternative candidate was rejected>"
        }}

        Your task is to find the best class for the entity below with these policies & requirements.

        Entity: {entity}
        Context: {context}
        Remaining MCP tool calls: {remaining_calls}

        """
    ).strip()


def data_property_resolution_agent_prompt(
    subject: str,
    subject_class_uri: str,
    subject_entity_uri: str,
    predicate: str,
    obj: str,
    transcript: str,
    remaining_calls: int,
    force_finish: bool = False,
) -> str:
    finish_rule = (
        "You MUST finish now. Do not request another tool call."
        if force_finish else
        "You may either call one MCP tool or finish with the final answer."
    )
    return textwrap.dedent(
        f"""
        You are an ontology expert navigating ontology properties via MCP tools.

        Requirements:
        - You have full control over ontology exploration.
        - Only choose a DataProperty if it clearly matches the subject class and object value.
        - If the object should remain an entity relationship, finish with no property URI.
        - A temporary typed entity has already been created in MCP for this subject. Prefer `recommend_attribute` first.
        - Only use these tools: recommend_attribute, search_properties, get_class_details, list_available_facets.
        - {finish_rule}

        Tool argument schema:
        - recommend_attribute: {{"entity_uri": "<temporary subject entity URI>", "query": "<text query>", "value": "<literal value>", "context": "<optional context>"}}
        - search_properties: {{"query": "<text query>"}}
        - get_class_details: {{"class_uri": "<class URI>"}}
        - list_available_facets: {{"class_uri": "<class URI>"}}

        Never invent argument names. In particular, NEVER use:
        - subject_class
        - keyword
        - predicate_keyword
        - uri

        Tool-call self-check before every tool call:
        1. The tool name is allowed.
        2. The arguments exactly match the schema above.
        3. Every required argument is present.
        4. Do not repeat a previously failed call with the same tool and arguments.

        Exploration policy:
        - Start with `recommend_attribute` unless the temporary entity URI is missing or unusable.
        - `recommend_attribute` is context-aware and domain/range-aware. Use it as the primary tool.
        - `search_properties` only accepts a text query. It does NOT support domain/range filters.
        - Use `search_properties` only as a fallback or to inspect alternatives after `recommend_attribute`.
        - Do not finish immediately after one failed search unless you have tried a reasonable alternative wording or checked class details/facets when relevant.
        - If a previous tool call failed with a validation error, correct the argument schema before continuing.
        - Repeating the same invalid call is a reasoning failure.

        Modeling policy:
        - Choose a DataProperty only if the object is best modeled as a literal value, not as a reusable CTI entity or concept.
        - If the object is a malware family, vulnerability, campaign, tool, attacker, behavior, capability, payload, or other reusable CTI concept, it should remain an entity relationship rather than a DataProperty.

        Tool transcript so far:
        {transcript}

        Return JSON in one of these forms:

        Tool call:
        {{
          "action": "call_tool",
          "tool": "<tool name>",
          "arguments": {{ ... }},
          "reason": "<why this tool call is needed>"
        }}

        Final answer:
        {{
          "action": "finish",
          "is_data_property": <true/false>,
          "property_uri": "<URI or empty string>",
          "property_name": "<name or empty string>",
          "confidence": <0.0-1.0>,
          "reason": "<why this is the best stopping point, including why the strongest alternative interpretation was rejected>"
        }}

        Decide whether the object should be modeled as a literal DataProperty on the subject.

        Subject: {subject}
        Subject class URI: {subject_class_uri}
        Temporary subject entity URI: {subject_entity_uri}
        Predicate (natural language): {predicate}
        Object value: {obj}
        Remaining MCP tool calls: {remaining_calls}
        """
    ).strip()


def object_property_resolution_agent_prompt(
    subject: str,
    subject_class_uri: str,
    subject_entity_uri: str,
    predicate: str,
    obj: str,
    obj_class_uri: str,
    object_entity_uri: str,
    predicate_queries: List[str],
    transcript: str,
    remaining_calls: int,
    force_finish: bool = False,
) -> str:
    finish_rule = (
        "You MUST finish now. Do not request another tool call."
        if force_finish else
        "You may either call one MCP tool or finish with the final answer."
    )
    return textwrap.dedent(
        f"""
        You are an ontology expert navigating ontology properties via MCP tools.

        Requirements:
        - You have full control over ontology exploration.
        - Do NOT assume the first property search result is correct.
        - Verify domain/range fit as much as possible with the available tools.
        - If no ontology property clearly fits, return an empty property URI.
        - Temporary typed entities have already been created in MCP for the subject and object. Prefer `recommend_relation` first.
        - Only use these tools: recommend_relation, search_properties, get_class_details, list_available_facets.
        - {finish_rule}

        Tool argument schema:
        - recommend_relation: {{"subject_uri": "<temporary subject entity URI>", "object_uri": "<temporary object entity URI>", "query": "<text query>", "context": "<optional context>"}}
        - search_properties: {{"query": "<text query>"}}
        - get_class_details: {{"class_uri": "<class URI>"}}
        - list_available_facets: {{"class_uri": "<class URI>"}}

        Never invent argument names. In particular, NEVER use:
        - subject_class
        - keyword
        - predicate_keyword
        - uri

        Tool-call self-check before every tool call:
        1. The tool name is allowed.
        2. The arguments exactly match the schema above.
        3. Every required argument is present.
        4. Do not repeat a previously failed call with the same tool and arguments.

        Exploration policy:
        - Start with `recommend_relation` unless a temporary entity URI is missing or unusable.
        - `recommend_relation` is domain/range-aware and should be the primary tool.
        - `search_properties` only accepts a text query. It does NOT support domain/range filters.
        - Use `search_properties` as a fallback or to inspect alternatives after `recommend_relation`.
        - Use the provided normalized predicate queries before inventing new wording.
        - Before finishing, try direct property search with a reasonable query and inspect class details/facets when the recommendation is weak or empty.
        - Do not finish immediately after one failed search if a simple alternative wording is still available.
        - If a previous tool call failed with a validation error, correct the argument schema before continuing.
        - Repeating the same invalid call is a reasoning failure.
        - If no exact lexical match exists but a schema-valid candidate is semantically close, choose that candidate instead of returning empty.
        - Never choose a property outside the schema-valid candidates surfaced by the tools.

        Tool transcript so far:
        {transcript}

        Normalized predicate queries to try in order:
        {json.dumps(predicate_queries, ensure_ascii=False)}

        Return JSON in one of these forms:

        Tool call:
        {{
          "action": "call_tool",
          "tool": "<tool name>",
          "arguments": {{ ... }},
          "reason": "<why this tool call is needed>"
        }}

        Final answer:
        {{
          "action": "finish",
          "property_uri": "<URI or empty string>",
          "property_name": "<name or empty string>",
          "confidence": <0.0-1.0>,
          "reason": "<why this is the best stopping point, including why the strongest alternative property candidate was rejected>"
        }}

        Find the best ObjectProperty for the relationship below.

        Subject: {subject}
        Subject class URI: {subject_class_uri}
        Temporary subject entity URI: {subject_entity_uri}
        Predicate (natural language): {predicate}
        Object: {obj}
        Object class URI: {obj_class_uri}
        Temporary object entity URI: {object_entity_uri}
        Remaining MCP tool calls: {remaining_calls}        
        """
    ).strip()


def node_cypher_prompt(
    name: str,
    class_name: str,
    class_uri: str,
    filename: str,
    properties: Dict[str, Any],
) -> str:
    return textwrap.dedent(
        f"""
        Generate a Cypher query to MERGE a node in Neo4j with the following details.
        Use MERGE to avoid duplicates. Set all properties on creation and update.

        Node:
        - name: {name}
        - entity_type: {class_uri}
        - source_document: {filename}
        - additional_properties: {json.dumps(properties, ensure_ascii=False)}

        Return JSON:
        {{
          "cypher": "<Cypher query string>"
        }}
        """
    ).strip()


def edge_cypher_prompt(
    subject: str,
    subject_class: str,
    predicate: str,
    predicate_uri: str,
    obj: str,
    obj_class: str,
    sentence: str,
    chunk: str,
    is_literal: bool,
) -> str:
    if is_literal:
        return textwrap.dedent(
            f"""
            Generate a Cypher query to add a property on subject node (not a separate node).
            Subject:
            - name: {subject}
            - entity_type: {subject_class}
            - property_uri: {predicate_uri}
            - object: {obj}
            - source_sentence: {sentence}
            - source_chunk: {chunk}

            Return JSON:
            {{
              "cypher": "<Cypher query string>"
            }}
            """
        ).strip()
    return textwrap.dedent(
        f"""
        Generate a Cypher query to create an object-property relationship.

        Subject: {subject} ({subject_class})
        Predicate: {predicate} ({predicate_uri})
        Object: {obj} ({obj_class})
        Source sentence: {sentence}
        Source chunk: {chunk}

        Return JSON:
        {{
          "cypher": "<Cypher query string>"
        }}
        """
    ).strip()
