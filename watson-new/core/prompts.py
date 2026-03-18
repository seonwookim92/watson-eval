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
        You are splitting CTI text into extraction-ready sentences for knowledge-graph construction.
        ONLY PRINT THE REWRITTEN TEXT.

        Allowed edits:
        1. Split a sentence that expresses multiple relations into separate sentences, one relation each.
        2. Make the grammatical subject explicit in each sentence — do not use pronouns or omit it.
        3. Separate embedded clauses or prepositional phrases that carry an independent relation into
           their own sentence, using only wording already present in the source.

        Strict prohibitions:
        - Do NOT convert passive voice to active voice when doing so would invert subject and object roles.
        - Do NOT compress support or purposive verb constructions.
          Keep "is used to install", "is designed to", "is known to", "is used for" exactly as written.
        - Do NOT introduce verbs, predicates, or noun phrases not present in the original text.
        - Do NOT merge or reorder relations across sentences.
        - Do NOT omit any relation, entity, or qualifier present in the original.
        - If a sentence already expresses exactly one clear relation, copy it unchanged.

        Text:
        {chunk}
        """
    ).strip()


def paraphrase_stage4_retry_prompt(original: str, previous: str, issues: str) -> str:
    return textwrap.dedent(
        f"""
        You are splitting CTI text into extraction-ready sentences for knowledge-graph construction.
        The previous rewrite failed quality checks. Fix the issues and ONLY PRINT THE REWRITTEN TEXT.

        Allowed edits:
        1. Split multi-relation sentences into separate single-relation sentences.
        2. Make the grammatical subject explicit — no pronouns.
        3. Separate embedded clauses carrying an independent relation using only original wording.

        Strict prohibitions:
        - Do NOT convert passive to active when doing so inverts subject and object roles.
        - Do NOT compress support verbs. Keep "is used to install", "is designed to" exactly as written.
        - Do NOT introduce verbs, predicates, or noun phrases not in the original.
        - Do NOT omit any relation, entity, or qualifier.
        - If a sentence already has one relation, copy it unchanged.

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
        You are a strict CTI quality checker for knowledge-graph extraction.
        Compare the rewritten text with the original text and answer every field carefully.

        Return JSON only:
        {{
          "clear_spo": <true/false>,
          "one_relation_per_sentence": <true/false>,
          "technical_content_preserved": <true/false>,
          "argument_roles_preserved": <true/false — no subject/object inversion occurred>,
          "predicate_semantics_preserved": <true/false — support verbs like "is used to install" were NOT compressed to direct action verbs like "installs"; predicate meaning is unchanged>,
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


def span_normalization_prompt(
    sentence: str,
    subject: str,
    predicate: str,
    obj: str,
    role: str,
    span: str,
) -> str:
    return textwrap.dedent(
        f"""
        You are normalizing entity spans in CTI knowledge-graph triplets.
        Your job is NOT to invent new facts. Your job is only to decide whether the target span
        should remain as one entity node or be reduced to a cleaner head entity with attached qualifiers.

        Target triplet:
        - subject: {subject}
        - predicate: {predicate}
        - object: {obj}

        Target role: {role}
        Target span: {span}
        Source sentence: {sentence}

        Allowed decisions:
        1. "keep": keep the span exactly as-is
        2. "rewrite_head": keep one entity node but rewrite it to a cleaner head span
        3. "split": produce one cleaner canonical entity span plus zero or more attached qualifiers

        Important rules:
        - If the span is a fixed name, official name, malware family, organization/team name, IoC, CVE, or product/version string, prefer "keep".
        - If the span contains relation-bearing tails such as "in Y", "for Y", "by Y", "against Y", "from Y", "with Y", "such as Y", or "including Y",
          you may choose "rewrite_head" or "split".
        - If you choose "split", relation_hint must describe the relation FROM the canonical span TO the qualifier value.
        - Use only words grounded in the source sentence.
        - If unsure, choose "keep".
        - Do not output more than 3 qualifiers.

        Return JSON only:
        {{
          "decision": "keep" | "rewrite_head" | "split",
          "canonical_span": "<string>",
          "qualifiers": [
            {{
              "relation_hint": "<short relation from canonical span to qualifier>",
              "value": "<qualifier span from the sentence>"
            }}
          ],
          "confidence": <0.0-1.0>,
          "reason": "<brief explanation>"
        }}
        """
    ).strip()


def qualifier_node_worthiness_prompt(
    sentence: str,
    subject: str,
    predicate: str,
    obj: str,
    canonical_span: str,
    role: str,
    relation_hint: str,
    qualifier_value: str,
    entity_memory: str = "",
) -> str:
    memory_block = entity_memory or "(none)"
    return textwrap.dedent(
        f"""
        You are judging whether a qualifier extracted from a CTI sentence should be promoted to an independent graph node.
        Decide whether the qualifier is node-worthy for a cybersecurity knowledge graph.

        Return JSON only:
        {{
          "is_node_worthy": <true/false>,
          "category": "entity|attribute|status|role_label|noise",
          "suggested_handling": "promote_node|keep_as_qualifier|attach_as_attribute|drop",
          "confidence": <0.0-1.0>,
          "reason": "<brief explanation>"
        }}

        Decision guidelines:
        - "entity": independent real-world or technical thing that can stand as a graph node
        - "attribute": descriptive property/value that should usually stay attached to another node
        - "status": state/condition/boolean-like qualifier, not an independent node
        - "role_label": generic role/category phrase, not a named entity by itself
        - "noise": malformed fragment or overly generic term with low graph value
        - If unsure, do NOT promote to a node.
        - Use only the given sentence and triplet context. Do not invent facts.

        In-context examples:
        Example 1:
        - Canonical span: 0-day exploit chain
        - Relation hint: targets
        - Qualifier value: iPhones
        Output:
        {{"is_node_worthy": true, "category": "entity", "suggested_handling": "promote_node", "confidence": 0.95, "reason": "iPhones is an independent technical target entity."}}

        Example 2:
        - Canonical span: 0-day exploit chain
        - Relation hint: status
        - Qualifier value: in-the-wild
        Output:
        {{"is_node_worthy": false, "category": "status", "suggested_handling": "keep_as_qualifier", "confidence": 0.97, "reason": "in-the-wild is a status, not an independent entity node."}}

        Example 3:
        - Canonical span: Intellexa
        - Relation hint: is_a
        - Qualifier value: commercial surveillance vendor
        Output:
        {{"is_node_worthy": false, "category": "role_label", "suggested_handling": "keep_as_qualifier", "confidence": 0.92, "reason": "commercial surveillance vendor is a generic role label, not a named entity."}}

        Example 4:
        - Canonical span: Predator spyware
        - Relation hint: installs_on
        - Qualifier value: device
        Output:
        {{"is_node_worthy": false, "category": "noise", "suggested_handling": "drop", "confidence": 0.90, "reason": "device is too generic to add as a useful node here."}}

        Source sentence: {sentence}
        Original triplet:
        - subject: {subject}
        - predicate: {predicate}
        - object: {obj}

        Canonical span: {canonical_span}
        Role: {role}
        Qualifier relation hint: {relation_hint}
        Qualifier value: {qualifier_value}
        Relevant entity memory:
        {memory_block}
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

        Evidence priority:
        1. Direct definitional statements such as "X is a Y"
        2. Apposition such as "X, a Y,"
        3. Strong name-pattern clues such as CVE IDs, version strings, or malware-family terms
        4. Relation-role clues from the surrounding SPO context
        If these conflict, trust the stronger evidence first.

        Examples (entity → concept label):
        - "Predator spyware"                      → "malicious tool malware"
        - "Apple Inc."                            → "organization company"
        - "CVE-2023-41991"                        → "vulnerability CVE identifier"
        - "iOS 16.7"                              → "operating system software"
        - "192.168.1.1"                           → "IP address network indicator"
        - "Google's Threat Analysis Group (TAG)"  → "organization security team"
        - "exploit chain"                         → "attack pattern exploit"
        - "phishing email"                        → "attack pattern phishing message"
        - "phishing tactics"                      → "attack pattern technique"
        - "registry key"                          → "file system registry artifact"
        - "SHA256 hash"                           → "file hash indicator"
        - "Cobalt Strike"                         → "malicious tool"
        - "3AM ransomware"                        → "malicious tool ransomware"
        - "Cuba ransomware"                       → "malicious tool ransomware"
        - "LockBit"                               → "malicious tool ransomware"
        - "Microsoft Exchange"                    → "software application"
        - "threat actor group"                    → "threat actor group"
        - "North Korea"                           → "geographic location country"
        - "Israel"                                → "geographic location country"
        - "healthcare sector"                     → "organization industry"
        - "finance industry"                      → "organization industry"
        - "Qualcomm"                              → "organization company"
        - "personally identifiable information"   → "content data sensitive"

        Additional guidance for common mistakes:
        - Country or region names (North Korea, Iran, Russia, etc.) → "geographic location country"
          Do NOT classify countries as Facets or Identity attributes.
        - Industry sector names (healthcare, finance, manufacturing, education) → "organization industry"
          These are victim sector labels, NOT targeting-behavior concepts.
        - Malware/ransomware family names, even unfamiliar ones → "malicious tool malware"
          If the context uses words like ransomware, backdoor, RAT, spyware, implant → "malicious tool"
        - Attack methods/techniques (phishing, call-back, spear-phishing) → "attack pattern technique"
          These are behaviors, not victim-targeting concepts.

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
    def _fmt(c: Dict[str, str]) -> str:
        desc = c.get("description", "").strip()
        return f"- {c['name']} ({c['uri']})" + (f"\n    Description: {desc}" if desc else "")
    candidates_text = "\n".join(_fmt(c) for c in candidates) if candidates else "(no candidates found)"
    return textwrap.dedent(
        f"""
        You are an ontology expert. Given the entity below, find the most accurate
        class in the ontology that best describes it.

        Specificity rule: Prefer specific classes over generic parents, BUT only when
        the specific class accurately describes the entity's fundamental nature.
        Do NOT pick a subclass merely because it mentions related concepts.
          WRONG: Qualcomm → WearableDevice  (Qualcomm is the company, not its product)
          RIGHT: Qualcomm → Organization
          WRONG: North Korea → CountryOfResidenceFacet  (that is a personal-identity attribute)
          RIGHT: North Korea → Location or Place

        When evidence conflicts, use this priority order:
        1. Explicit definitional statements
        2. Appositive descriptions
        3. Strong name-pattern signals
        4. Relation-role clues from the local SPO
        Do not label an organization as a threat actor unless the context explicitly says
        the entity is the attacker rather than merely a vendor, reporter, or research team.

        FORBIDDEN — never select these for entity typing:
        - UCO Facet classes (names ending in "Facet", e.g. CountryOfResidenceFacet,
          IdentifierFacet, LanguagesFacet). Facets are attribute-bundles attached to
          Identity nodes, not entity types.
        - VictimTargeting — this describes adversarial targeting behaviour, not the
          entity being targeted. Industry sectors (healthcare, finance) and attack
          techniques (phishing, call-back) must NOT use this class.
        - Structural/meta-classes such as AttributedName, Grouping — these are
          ontology-internal constructs, not entity types.

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


def predicate_candidate_judge_prompt(
    chunk_text: str,
    sentence: str,
    subject: str,
    subject_class_uri: str,
    predicate: str,
    obj: str,
    object_class_uri: str,
    candidates: List[Dict[str, Any]],
) -> str:
    candidate_lines = "\n".join(
        [
            (
                f"- property_uri: {item.get('property_uri', '')}\n"
                f"  property_name: {item.get('property_name', '')}\n"
                f"  is_inverse: {str(bool(item.get('is_inverse', False))).lower()}\n"
                f"  is_data_property: {str(bool(item.get('is_data_property', False))).lower()}\n"
                f"  confidence: {item.get('confidence', 0.0)}\n"
                f"  source: {item.get('source', '')}\n"
                f"  note: {item.get('note', '')}"
            )
            for item in candidates
        ]
    ) if candidates else "(no candidates)"
    return textwrap.dedent(
        f"""
        You are judging ontology predicate candidates for one CTI triplet.

        Your task:
        - Use the full chunk and source sentence to decide whether one of the provided ontology
          property candidates correctly represents the SPO relation.
        - Choose exactly one candidate from the list, or choose no match.
        - Never invent a property URI that is not in the candidate list.

        Modeling rules:
        - `is_data_property=true` means the object should be treated as a literal value attached to the subject.
        - `is_inverse=true` means the ontology direction is reversed relative to the extracted SPO.
        - Be strict. If the candidate list does not clearly fit the chunk evidence, return no match.
        - If a candidate would change the meaning of the SPO rather than normalize it, reject it.

        Full chunk:
        {chunk_text}

        Source sentence:
        {sentence}

        Extracted SPO:
        - subject: {subject}
        - subject_class_uri: {subject_class_uri}
        - predicate: {predicate}
        - object: {obj}
        - object_class_uri: {object_class_uri}

        Candidate properties:
        {candidate_lines}

        Return JSON:
        {{
          "property_uri": "<candidate URI or empty string>",
          "property_name": "<candidate name or empty string>",
          "is_inverse": <true/false>,
          "is_data_property": <true/false>,
          "confidence": <0.0-1.0>,
          "reason": "<brief explanation>"
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
          IMPORTANT: "most specific" means most accurately describing the entity's nature — not just
          any subclass that mentions a related concept. If a parent class is accurate and the subclass
          would require over-interpretation, choose the parent.
          WRONG: Qualcomm → WearableDevice  (Qualcomm is the company, not its chip-powered product)
          RIGHT: Qualcomm → Organization
        - Only use these tools: search_classes, list_root_classes, list_subclasses, get_class_hierarchy, get_class_details, list_available_facets, drill_into_classes.
        - {finish_rule}

        FORBIDDEN — never select these for entity typing:
        - UCO Facet classes (names ending in "Facet", e.g. CountryOfResidenceFacet,
          IdentifierFacet, LanguagesFacet). Facets are attribute-bundles attached to
          Identity nodes, not entity types. For country/region names, search for
          Location or Place instead.
        - VictimTargeting — describes adversarial targeting behaviour, not the entity
          being targeted. Industry sectors (healthcare, finance, manufacturing) and
          attack technique names (phishing, call-back phishing) must NOT use this class.
        - Structural/meta-classes such as AttributedName, Grouping — ontology-internal
          constructs that must not be used as entity types.

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

        Weak recommend_relation handling (IMPORTANT):
        - Inspect the [Score: X.XXXX] values in `recommend_relation` results.
        - If ALL returned scores are below 0.45, or if `recommend_relation` returned "No valid ObjectProperties found",
          this signals that the entity types may be incorrect or the domain/range does not cover this relation.
          In that case, you MUST call `search_properties` with the predicate phrase before finishing.
        - `search_properties` is type-agnostic and can find the right property even when entity typing is imperfect.
        - If `search_properties` returns a candidate that is a better semantic match for the predicate phrase
          (even if not domain/range validated), prefer it over a weak `recommend_relation` result with score < 0.45.

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
          "is_inverse": <true/false>,
          "confidence": <0.0-1.0>,
          "reason": "<why this is the best stopping point, including why the strongest alternative property candidate was rejected>"
        }}

        Inverse-direction rule:
        - If the selected property is only valid in the reverse ontology direction
          (the recommend_relation result marks it as an INVERSE relation), set
          "is_inverse": true.
        - If the selected property is valid in the current subject -> object direction,
          set "is_inverse": false.
        - Do not set "is_inverse": true unless the chosen property is explicitly an
          inverse-direction match in the tool results.

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
