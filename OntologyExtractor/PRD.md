# PRD: OntologyExtractor

## 1. 개요

**프로젝트명**: OntologyExtractor
**목적**: CTI(Cyber Threat Intelligence) 리포트 문서에서, 사용자가 제공한 Ontology Schema(.ttl, .owl 등)에 부합하는 Knowledge Graph를 자동 추출하여 Neo4j에 저장하는 LangGraph 기반 Python 스크립트

---

## 2. 실행 방식

```bash
python OntologyExtractor.py <file_or_url> <ontology_schema_file>
```

- `<file_or_url>`: 분석할 CTI 리포트 (로컬 파일 경로 또는 URL)
- `<ontology_schema_file>`: 사용할 Ontology Schema 단일 파일 경로 (.ttl, .owl, .rdf 등)

---

## 3. 의존 프로젝트 및 인터페이스

### 3.1 서브모듈
| 이름 | 경로 | 연동 방식 |
|------|------|-----------|
| universal-ontology-mcp | `OntologyExtractor/universal-ontology-mcp/` | subprocess + MCP JSON-RPC over stdio |
| TextItDown | `TextItDown/` | subprocess 실행 (`python TextItDown/textitdown.py <input> <output>`) |

### 3.2 외부 라이브러리
| 라이브러리 | 용도 |
|-----------|------|
| LangGraph | 파이프라인 그래프 오케스트레이션 |
| iocsearcher | IoC 후보 추출 |
| chromadb | 로컬 벡터 DB (Python 라이브러리) |
| NLTK | 문장 단위 분리 |
| neo4j (Python driver) | Neo4j 연결 및 데이터 삽입 |

### 3.3 LLM 인터페이스
- **Endpoint**: `POST http://192.168.100.2:8081/v1/chat/completions`
- **Model**: `Qwen3/Qwen3-Coder-Next`
- **Response Format**: `{"type": "json_object"}` (structured output) — JSON 파싱이 필요한 모든 Step에 적용
- **모든 프롬프트는 영어로 작성**

### 3.4 Embedding 인터페이스
- **Endpoint**: `POST http://192.168.100.2:8082/v1/embeddings`
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **필수 옵션**: `"truncate_prompt_tokens": 256`

### 3.5 Neo4j
- **Bolt URI**: `bolt://192.168.100.2:7687`
- **ID**: `neo4j`
- **Password**: `password`

---

## 4. 설정 파일 (config.json)

스크립트 실행 디렉토리에 `config.json`을 두며, 하드코딩 없이 모든 설정값을 여기서 관리한다.

```json
{
  "llm": {
    "base_url": "http://192.168.100.2:8081/v1",
    "model": "Qwen3/Qwen3-Coder-Next"
  },
  "embedding": {
    "base_url": "http://192.168.100.2:8082/v1",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "truncate_prompt_tokens": 256
  },
  "neo4j": {
    "uri": "bolt://192.168.100.2:7687",
    "user": "neo4j",
    "password": "password"
  },
  "chunking": {
    "method": "semantic",
    "max_chunk_bytes": 5000
  },
  "mcp": {
    "server_script": "./universal-ontology-mcp/main.py",
    "max_tool_calls_type_matching": 150,
    "max_tool_calls_property_matching": 30
  },
  "retry": {
    "semantic_chunking": 3,
    "paraphrasing": 4,
    "triplet_extraction": 3,
    "data_insert": 3
  },
  "entity_resolution": {
    "top_k": 10
  }
}
```

---

## 5. 산출물 디렉토리 구조

```
run_{YYYYMMDD}_{HHMMSS}/
├── run.log
└── intermediate/
    ├── preprocess/
    │   └── <원본 파일명>.txt
    ├── chunking/
    │   ├── chunk_0000.txt
    │   ├── chunk_0001.txt
    │   └── ...
    ├── paraphrase/
    │   ├── chunk_0000.txt          (성공)
    │   ├── chunk_0000_tried.txt    (실패, 최종 tried 결과 저장)
    │   └── ...
    ├── triplets.json
    └── typedTriplets.json
```

---

## 6. LangGraph 노드 구성

```
Pre-processing
    └── Chunking
            └── Paraphrasing
                    └── Triplet Extraction
                            └── IoC Detection & Rearm
                                    └── Triplet Type Matching
                                            └── Internal Entity Resolution
                                                    └── Existing Entity Resolution
                                                            └── Data Insert
```

모든 노드는 순차 처리(Sequential). 재실행(Resumability) 기능 없음.

---

## 7. 노드 상세 명세

### Node 1: Pre-processing

**목적**: 입력 파일/URL을 plain text로 변환

**처리 절차**:
1. 현재 디렉토리에 `run_{YYYYMMDD}_{HHMMSS}/intermediate/preprocess/` 폴더 생성
2. TextItDown 실행:
   ```bash
   python TextItDown/textitdown.py <input> <output_path>
   ```
   - `<output_path>`: `preprocess/` 폴더 경로 지정
3. 변환된 plain text 파일 경로를 다음 노드로 전달

**로그**: 변환 성공/실패 여부 기록

---

### Node 2: Chunking

**목적**: plain text를 의미 단위로 분할

**기본 방식**: Semantic Chunking
**폴백 방식**: Paragraph Chunking (Semantic 실패 시)

#### 2-1. Semantic Chunking
- `MAX_CHUNK_BYTES = 5000` (config에서 읽음)
- 텍스트를 `MAX_CHUNK_BYTES` 단위로 슬라이싱한 뒤 LLM에 전송:

```
You are a semantic text splitter. Your goal is to find the best point to end a
text chunk so that context is preserved (e.g., at the end of a paragraph or a
complete thought).

I will provide a block of text. Identify the most appropriate sentence to END
the chunk within this block.
Return ONLY the exact text of that last sentence. DO NOT PARAPHRASE.

TEXT BLOCK:
{text_block}
```

- LLM이 반환한 문장을 기준으로 해당 위치까지를 하나의 Chunk로 확정
- 잘린 지점 이후부터 다시 `MAX_CHUNK_BYTES`만큼 슬라이싱하여 반복
- **Retry**: Chunk당 최대 3회. 3회 실패 시 해당 지점부터 Paragraph Chunking으로 전환

#### 2-2. Paragraph Chunking (Fallback)
- `\n` 기준으로 split
- 공백/빈 줄은 결과에서 제외

**산출물**: `intermediate/chunking/chunk_{NNNN}.txt` (4자리 zero-padding)

---

### Node 3: Paraphrasing

**목적**: Chunk 내용을 Subject-Predicate-Object 중심의 명확한 영어 서술로 변환

**처리 절차**:

#### Step A: Paraphrase 요청 프롬프트
```
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

#### Step B: 검증 요청 프롬프트
```
You are a strict quality checker for CTI text processing.
Review the following paraphrased text against the original and answer in JSON:

{
  "has_figurative_language": <true/false>,
  "has_pronouns_or_nominal_phrases": <true/false>,
  "missing_technical_details": <true/false>,
  "issues": "<brief description or empty string>"
}

Original:
{original_chunk}

Paraphrased:
{paraphrased_chunk}
```

#### Step C: 재시도 로직
- 검증 결과 모든 항목이 `false`면 성공 → 다음 단계 진행
- 하나라도 `true`면 재시도 프롬프트 전송:

```
You are an expert CTI analyst. A previous paraphrasing attempt failed quality checks.
Improve the paraphrased text based on the failure reasons.

Original text:
{original_chunk}

Previous paraphrase attempt:
{previous_paraphrase}

Failure reasons:
{issues}

Apply the same rules:
1. Remove all figurative language and filler phrases.
2. Use Subject-Predicate-Object sentence structure throughout.
3. Replace all pronouns and nominal references with explicit proper nouns or named entities.
4. Preserve every technical detail from the original.
```

- **최대 시도 횟수**: 4회 (초회 포함)
- **4회 후에도 실패 시**: 원본 Chunk 그대로 사용 (로그에 실패 기록)

**산출물**:
- `intermediate/paraphrase/chunk_{NNNN}.txt`: 성공한 결과 (또는 4회 실패 후 원본)
- `intermediate/paraphrase/chunk_{NNNN}_tried.txt`: 실패한 Chunk의 마지막 tried 결과

---

### Node 4: Triplet Extraction

**목적**: Paraphrase된 텍스트에서 (Subject, Predicate, Object) 형태의 Triplet 추출

**처리 절차**:
1. 각 Chunk를 NLTK로 문장 단위 분리
2. 문장 단위로 LLM에 Triplet 추출 요청:

```
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
{paraphrased_chunk}

Target sentence:
{sentence}
```

- **Retry**: 최대 3회. 3회 후에도 유효한 JSON이 나오지 않으면 해당 문장 Skip (로그에 기록)

**산출물**: `intermediate/triplets.json`

```json
[
  {
    "id": 0,
    "source_sentence": "...",
    "source_chunk": "chunk_0000",
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "isSubjectIoC": false,
    "isObjectIoC": false
  },
  ...
]
```

---

### Node 5: IoC Detection & Rearm

**목적**: Triplet의 Subject/Object에서 IoC를 탐지하고, Defanged 표현을 원래 값으로 복원(Rearm)

**처리 절차**:
1. 각 Triplet의 Subject와 Object에 대해 `iocsearcher` 라이브러리로 IoC 후보 추출
2. IoC 후보가 있을 경우 LLM에 판별 및 Rearm 요청:

```
You are a CTI IoC analyst. Determine if the following string is a genuine IoC
(Indicator of Compromise) such as an IP address, domain, URL, file hash, CVE ID,
email address, or registry key.

If it is an IoC, also provide the rearmed (defanged reversed) version
(e.g., "hxxp" -> "http", "192[.]168[.]1[.]1" -> "192.168.1.1").

Return JSON:
{
  "is_ioc": <true/false>,
  "rearmed_value": "<rearmed string or original if not IoC>"
}

String to evaluate: {candidate}
Context: {subject_or_object}
```

3. `is_ioc: true`이면 해당 Triplet의 `isSubjectIoC` 또는 `isObjectIoC`를 `true`로 설정
4. `rearmed_value`로 Subject/Object 값 덮어쓰기

**업데이트**: `intermediate/triplets.json` 갱신 (in-place 덮어쓰기)

---

### Node 6: Triplet Type Matching

**목적**: 각 SPO에 대해 Ontology Schema 상의 Class/Property를 매핑

**universal-ontology-mcp 연동**:
- `universal-ontology-mcp/main.py`를 subprocess로 실행
- MCP 환경변수: `ONTOLOGY_DIR = <schema 파일의 부모 디렉토리>`
- 통신: MCP JSON-RPC over stdio
- **사용 허용 Tool**: `get_ontology_summary`, `list_root_classes`, `list_subclasses`, `get_class_hierarchy`, `search_classes`, `search_properties`, `get_class_details`, `list_available_facets`
- **사용 금지 Tool**: `create_entity`, `set_property`, `attach_component`, `remove_entity`, `reset_graph`, `export_graph`, `visualize_graph`, `export_graph`, `validate_entity` 등 Graph 조작 Tool 전체

#### 6-0. Report Document Root Node 처리
- **처음 단 한 번**: 입력 문서 자체를 Root Node로 등록
- Node name: 입력 파일명 (확장자 포함)
- Entity Type: MCP를 통해 CTI 보고서/문서에 해당하는 적합한 Class 탐색 (MCP Tool 호출 최대 150회)
- 이후 모든 추출 Entity는 이 Root Node와 연결 대상으로 고려됨 (Data Insert 단계에서 처리)

#### 6-1. Subject Type Matching
- Subject는 항상 Entity로 처리
- MCP Tool을 사용해 LLM이 적합한 Class URI 탐색
- 상위 Class 발견 시, 하위 Class(Subclass)도 반드시 확인하여 더 구체적인 Class가 있다면 하위 Class를 선택하도록 프롬프트에 명시:

```
You are an ontology expert. Given the entity below, find the most specific
class in the ontology that best describes it.

IMPORTANT: If you find a matching class, always check its subclasses using
list_subclasses() before finalizing. Choose the most specific (leaf-level)
class that still accurately describes the entity. Do NOT select a broader
parent class if a more specific subclass is a better fit.

Entity: {subject}
Context: {source_sentence}

Use the available MCP tools to search and navigate the ontology, then return:
{
  "class_uri": "<URI>",
  "class_name": "<name>",
  "confidence": <0.0-1.0>
}
```

- **MCP Tool 호출 최대**: 150회 (Triplet 하나당)

#### 6-2. Predicate/Object Type Matching

**Object Entity/Literal 판단 순서**:

1. `isObjectIoC == true` → 무조건 Entity
2. `isObjectIoC == false` → Subject의 Class에서 사용 가능한 DataProperty를 MCP로 확인
   → DataProperty가 Subject-Object 관계를 명확히 설명하면: Object는 **Literal**, 해당 DataProperty가 Predicate
   → 그렇지 않으면: Object는 **Entity**

**Object가 Literal인 경우**:
- 판단 과정에서 결정된 DataProperty URI가 Predicate으로 사용
- Object는 별도 Class/Type 없이 Literal 값으로 저장

**Object가 Entity인 경우**:
- Subject Type Matching과 동일한 방식으로 Object의 Class 탐색 (MCP Tool 호출 최대 150회, Triplet 하나당)
- Predicate에 해당하는 ObjectProperty 탐색 (MCP Tool 호출 최대 30회, Triplet 하나당):

```
You are an ontology expert. Find the ObjectProperty that best describes
the relationship between the subject entity and the object entity.

Subject: {subject} (Class: {subject_class})
Predicate (natural language): {predicate}
Object: {object} (Class: {object_class})

Use search_properties() and get_class_details() to find the best matching
ObjectProperty URI, then return:
{
  "property_uri": "<URI>",
  "property_name": "<name>",
  "confidence": <0.0-1.0>
}
```

**DataProperty 탐색 MCP Tool 호출 최대**: 30회 (Triplet 하나당)

**산출물**: `intermediate/typedTriplets.json`

```json
[
  {
    "id": 0,
    "source_sentence": "...",
    "source_chunk": "chunk_0000",
    "subject": "...",
    "subject_class_uri": "...",
    "subject_class_name": "...",
    "predicate": "...",
    "predicate_uri": "...",
    "object": "...",
    "object_is_literal": false,
    "object_class_uri": "...",
    "object_class_name": "...",
    "isSubjectIoC": false,
    "isObjectIoC": false
  },
  ...
]
```

---

### Node 7: Internal Entity Resolution

**목적**: 동일한 Entity Type 내에서 중복/유사 Entity를 통합

**처리 절차**:
1. `typedTriplets.json`에서 Entity를 `class_uri` 기준으로 그룹화
2. 같은 그룹 내 Entity 쌍에 대해 LLM으로 동일 여부 판별:

```
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

3. 동일 판정 시 `canonical_name`으로 통합, `typedTriplets.json` 내 모든 참조 업데이트
4. **별도 파일 저장 없음** — 로그에만 통합 내역 기록:
   ```
   [Entity Resolution] Merged "{entity_a}" + "{entity_b}" -> "{canonical_name}" (Type: {class_name})
   ```

---

### Node 8: Existing Entity Resolution

**목적**: Neo4j의 기존 Entity와 현재 추출 Entity 중복 해소

**처리 절차**:
1. 추출된 각 Entity에 대해 Embedding 생성 (Embedding API 호출)
2. ChromaDB에서 동일 Entity Type의 Entity 중 Embedding 유사도 Top-10 조회:
3. 조회된 후보에 대해 Node 7과 동일한 LLM 판별 로직 적용
4. 통합 시 Neo4j의 기존 Entity name을 canonical_name으로 사용
5. **별도 파일 저장 없음** — 로그에만 통합 내역 기록:
   ```
   [Existing Entity Resolution] Merged "{new_entity}" -> existing "{neo4j_entity}" (Type: {class_name})
   ```

---

### Node 9: Data Insert

**목적**: 최종 Knowledge Graph를 Neo4j에 삽입하고 Embedding 저장

#### 9-1. Report Root Node 삽입
- 가장 먼저 Report 문서를 Root Node로 Neo4j에 삽입
- Properties: `name` (파일명), `source_document` (파일명), `entity_type` (Ontology Class URI)

#### 9-2. Node 삽입
- Entity Node 삽입 (Literal Object는 Node로 삽입하지 않고 Edge 단계에서 Property로 처리)
- Node Properties: Ontology DataProperty로 확인된 값들 + `source_document` (원본 파일명)
- LLM을 이용해 Cypher 생성 및 실행:

```
Generate a Cypher query to MERGE a node in Neo4j with the following details.
Use MERGE to avoid duplicates. Set all properties on creation and update.

Node:
- name: {name}
- entity_type: {class_name}
- source_document: {filename}
- additional_properties: {data_properties_dict}

Return JSON:
{
  "cypher": "<Cypher query string>"
}
```

- **Retry**: 최대 3회. 실패 시 로그에 오류 기록 후 해당 Node Skip

#### 9-3. Edge 삽입
- ObjectProperty Predicate: MERGE Edge (Node to Node)
- DataProperty Predicate: 해당 Subject Node에 Property로 SET
- LLM을 이용해 Cypher 생성 및 실행
- **Retry**: 최대 3회. 실패 시 로그에 오류 기록 후 해당 Edge Skip

#### 9-4. Embedding 저장
- 삽입 성공한 모든 Entity에 대해 ChromaDB에 Embedding 저장:
  ```python
  # ChromaDB Collection Schema: name(str), entity_type(str), embedding(vector)
  collection.upsert(Doc(
      id=f"{entity_type}::{name}",
      fields={"name": name, "entity_type": entity_type, "embedding": embedding_vector}
  ))
  ```

#### 9-5. 성공률 로그
```
[Data Insert] Nodes: {success}/{total} inserted
[Data Insert] Edges: {success}/{total} inserted
[Data Insert] Embeddings: {success}/{total} saved to ChromaDB
```

---

## 8. 로깅

- **로그 파일**: `run_{YYYYMMDD}_{HHMMSS}/run.log`
- **로그 레벨**: INFO (기본), ERROR (오류 시)
- **기록 항목**:
  - 각 Node 시작/완료 시각
  - Chunking: 총 Chunk 수, Paragraph 폴백 발생 위치
  - Paraphrasing: 각 Chunk 성공/실패 여부, 원본 사용 여부
  - Triplet Extraction: Skip된 문장 목록
  - IoC Detection: 탐지된 IoC 목록
  - Type Matching: MCP Tool 호출 횟수 누계
  - Entity Resolution: 통합 내역 (Internal + Existing)
  - Data Insert: 성공률, 실패한 Node/Edge 및 오류 메시지

---

## 9. 파일 구조 (프로젝트)

```
OntologyExtractor/
├── OntologyExtractor.py         # 메인 스크립트 (LangGraph 파이프라인)
├── config.json                  # 설정 파일
├── universal-ontology-mcp/      # 서브모듈
└── run_{YYYYMMDD}_{HHMMSS}/     # 실행별 산출물 (런타임 생성)
    ├── run.log
    └── intermediate/
        ├── preprocess/
        ├── chunking/
        ├── paraphrase/
        ├── triplets.json
        └── typedTriplets.json
```

---

## 10. 제약 및 정책 요약

| 항목 | 값 |
|------|----|
| Chunking 방식 | Semantic (기본), Paragraph (폴백) |
| MAX_CHUNK_BYTES | 5000 |
| Semantic Chunking Retry | 3회 |
| Paraphrasing Loop | 최대 4회, 실패 시 원본 Chunk 사용 |
| Paraphrasing 출력 언어 | 영어 |
| Triplet Extraction Retry | 3회, 실패 시 Skip |
| Subject/Object Type Matching MCP Tool 호출 | 최대 150회 (Triplet 하나당) |
| Property Matching MCP Tool 호출 | 최대 30회 (Triplet 하나당) |
| Entity Resolution Top-K | 10 |
| Data Insert Retry | 3회 |
| 병렬 처리 | 없음 (순차 처리) |
| 재실행(Resumability) | 없음 |
| LLM 응답 형식 | structured output (`json_object`) |
| 모든 프롬프트 언어 | 영어 |
| MCP Graph 조작 Tool 사용 | 금지 |
