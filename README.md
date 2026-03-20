# CTI Ontology Extraction — Evaluation Framework

사이버 위협 인텔리전스(CTI) 보고서에서 **엔티티(Entity)**와 **트리플(Triple/Relation)**을 추출하고,
이를 표준 온톨로지 스키마(UCO / STIX / MalOnt)에 매핑하는 시스템들을 비교 평가하는 프레임워크입니다.

---

## 디렉터리 구조

```
eval/
├── run.py                   # 전체 실행 wrapper (모든 모델 × 스키마)
├── evaluate_entity.py       # 엔티티 추출 평가 (Embedding + LLM-as-a-Judge)
├── evaluate_triple.py       # 트리플 추출 평가 (soft/full, implicit 포함)
├── setup.sh                 # 환경 설치 스크립트
├── .gitignore
├── evaluation.md            # 평가 방법론 문서
├── baselines_analysis.md    # 베이스라인 분석 노트
│
├── watson/                # 우리 모델 (MCP 기반 온톨로지 분석기)
│   ├── eval.py              # 추출 실행
│   ├── evaluate.py          # 평가 (P/R/F1 계산)
│   ├── main.py              # 단일 파일 분석 CLI
│   ├── core/                # 파이프라인 핵심 로직
│   │   ├── pipeline/        # LangGraph 파이프라인 (노드, 상태)
│   │   ├── eval/            # 평가용 loader, matcher, metrics
│   │   └── mcp/             # MCP 클라이언트
│   ├── mcp/                 # MCP 서버 (universal-ontology-mcp)
│   ├── .env.sample          # 환경변수 템플릿
│   ├── requirements.txt
│   └── USAGE.md             # watson 상세 사용법
│
├── baselines/
│   ├── ctinexus/            # CTINexus (LLM 기반, 스키마 인지)
│   ├── ttpdrill/            # TTPDrill (NLP 기반, BM25 매칭)
│   ├── gtikg/               # GTIKG/CTIKG (LLM Open IE, 스키마 없음)
│   ├── ladder/              # LADDER (placeholder)
│   └── threat-intel-kg/     # Threat-Intelligence-KG (placeholder)
│
├── ontology/                # 공유 온톨로지 스키마 파일
│   ├── uco/                 # Unified Cyber Ontology
│   ├── stix/                # STIX 2.1 (OASIS)
│   └── malont/              # MalOnt (Malware Ontology)
│
├── datasets/                # 벤치마크 데이터셋
│   ├── ctinexus/annotation/ # CTINexus 어노테이션 (151개 JSON)
│   └── ctikg/               # CTIKG 벤치마크 CSV
│
└── outputs/                 # 모든 모델의 추출 결과
    ├── {model}_{schema}_results.json
    └── ...
```

---

## 데이터셋

### CTINexus Annotation Dataset
- **위치**: `datasets/ctinexus/annotation/`
- **크기**: 151개 JSON 파일 (APT 보고서, 랜섬웨어 분석, 취약점 보고서 등)
- **포맷**:
  ```json
  {
    "text": "CTI 보고서 원문 텍스트",
    "explicit_triplets": [
      {"subject": "Volt Typhoon", "relation": "targets", "object": "US critical infrastructure"}
    ],
    "entities": [
      {"entity_id": 0, "entity_name": "Volt Typhoon", "entity_type": "Attacker"}
    ]
  }
  ```

### CTIKG Benchmark
- **위치**: `datasets/ctikg/triple extraction benchmark and ctikg results.csv`
- **포맷**: `Sentence`, `Ground_Truth`, `sampled_tactic` 컬럼

---

## 모델 & 베이스라인

### 1. `watson` — 우리 모델 (MCP 기반 온톨로지 분석기)

**핵심 차별점**: MCP(Model Context Protocol) 서버를 통해 실시간으로 온톨로지를 조회하며 엔티티를 매핑합니다. LangGraph 파이프라인으로 청킹 → 패러프레이징 → 온톨로지 매핑을 순차 처리합니다.

| 항목 | 내용 |
|------|------|
| 방식 | LLM + MCP 온톨로지 서버 |
| 스키마 지원 | UCO, STIX, MalOnt |
| 엔티티 타이핑 | ✓ (MCP로 URI 해석) |
| 릴레이션 타이핑 | ✓ (UCO ObjectProperty 매핑) |
| venv 경로 | `watson/.venv` |

### 2. `ctinexus` — CTINexus (LLM Prompt Engineering)

**특징**: 온톨로지 스키마 전체를 시스템 프롬프트에 주입한 뒤 LLM이 직접 클래스를 할당합니다. Few-shot 예시 3개를 포함한 2단계(IE → ET) 파이프라인.

| 항목 | 내용 |
|------|------|
| 방식 | LLM 프롬프트 엔지니어링 |
| 스키마 지원 | baseline, UCO, STIX, MalOnt |
| 엔티티 타이핑 | ✓ |
| 릴레이션 타이핑 | ✓ |
| venv 경로 | `baselines/ctinexus/.venv` |

### 3. `ttpdrill` — TTPDrill (NLP + BM25)

**특징**: LLM 없이 spaCy 의존구문 분석(SRL)으로 S-V-O를 추출한 뒤 BM25 유사도로 온톨로지 클래스와 매칭합니다. 가장 빠르고 API 비용이 없습니다.

| 항목 | 내용 |
|------|------|
| 방식 | NLP (spaCy SRL + BM25 매칭) |
| 스키마 지원 | baseline, UCO, STIX, MalOnt |
| 엔티티 타이핑 | △ (BM25 근사 매칭) |
| 릴레이션 타이핑 | △ (BM25 근사 매칭) |
| venv 경로 | `baselines/ttpdrill/.venv_ttpdrill` |

### 4. `gtikg` — GTIKG/CTIKG (LLM Open IE)

**특징**: 온톨로지 스키마 없이 LLM이 자유 형식으로 트리플을 추출합니다. 엔티티 타입은 항상 `null`이며, 스키마 매핑 능력이 없다는 것을 명시적으로 보여주는 베이스라인입니다.

| 항목 | 내용 |
|------|------|
| 방식 | LLM Open Information Extraction |
| 스키마 지원 | baseline만 (스키마 무관) |
| 엔티티 타이핑 | ✗ (항상 null) |
| 릴레이션 타이핑 | ✗ (자유 텍스트) |
| venv 경로 | `baselines/gtikg/.venv_gtikg` |

---

## 환경 설치

### 전체 설치

```bash
# 모든 모델의 venv 생성 및 패키지 설치
bash setup.sh

# 특정 모델만 설치
bash setup.sh watson
bash setup.sh ctinexus ttpdrill
```

### 설치 후 API 키 설정

```bash
# watson/.env 파일에 API 키 입력
cp watson/.env.sample watson/.env
# OPENAI_API_KEY, OPENAI_MODEL 등 설정
```

> **참고**: ctinexus와 gtikg는 `baselines/ctinexus/.env`의 `OPENAI_API_KEY`를 공유합니다. `setup.sh`가 자동으로 심링크를 생성합니다.

---

## 추출 실행 — `run.py`

모든 모델과 스키마를 하나의 wrapper로 실행합니다.

### 기본 사용법

```bash
# 전체 job 목록 확인 (✓ = 결과파일 존재)
python run.py --list

# 커맨드 확인만 (실행 안 함)
python run.py --model all --schema all --dry-run
```

### 스모크 테스트 (샘플 3개)

```bash
# 모든 모델 × UCO, 3개 샘플
python run.py --schema uco --limit 3

# 우리 모델만, 모든 스키마, 5개 샘플
python run.py --model watson --schema all --limit 5
```

### 전체 실행

```bash
# 전체 데이터셋, 특정 모델 × 스키마
python run.py --model ctinexus --schema uco
python run.py --model ttpdrill --schema uco stix malont
python run.py --model watson --schema uco stix malont

# 모든 모델 × 모든 스키마 (전체 12개 job)
python run.py --model all --schema all

# 이미 있는 파일은 스킵
python run.py --model all --schema all --skip-existing
```

### 옵션 레퍼런스

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `all` | `watson` \| `ctinexus` \| `ttpdrill` \| `gtikg` \| `all` (복수 가능) |
| `--schema` | `all` | `uco` \| `stix` \| `malont` \| `baseline` \| `all` (복수 가능) |
| `--limit` | 전체 | 모델당 처리할 최대 샘플 수 |
| `--skip-existing` | off | 결과 파일이 이미 있으면 건너뜀 |
| `--dry-run` | off | 커맨드 출력만, 실행 안 함 |
| `--list` | off | job 목록 확인 후 종료 |

### 스키마 지원 매트릭스

| | baseline | uco | stix | malont |
|--|:--:|:--:|:--:|:--:|
| `watson` | — | ✓ | ✓ | ✓ |
| `ctinexus` | ✓ | ✓ | ✓ | ✓ |
| `ttpdrill` | ✓ | ✓ | ✓ | ✓ |
| `gtikg` | ✓ | — | — | — |

---

## 출력 포맷

모든 모델의 결과는 `outputs/{model}_{schema}_results.json`에 저장되며 동일한 포맷을 따릅니다.

```json
[
  {
    "file": "volt-typhoon-targets-us-critical-infrastructure.json",
    "text": "원문 CTI 보고서 텍스트...",
    "ontology": "uco",
    "extracted_entities": [
      { "name": "Volt Typhoon", "class": "Organization" },
      { "name": "LOLBins",      "class": "MaliciousTool" }
    ],
    "extracted_triplets": [
      {
        "subject":        "Volt Typhoon",
        "relation":       "targets",
        "relation_class": "object",
        "object":         "US critical infrastructure"
      }
    ]
  }
]
```

> **참고**: `gtikg`는 스키마 매핑이 없으므로 `class`가 항상 `null`이며, `relation_class`가 없습니다.

---

## 평가 실행 — `evaluate_entity.py` / `evaluate_triple.py`

`outputs/` 파일을 ground truth와 비교하여 Precision / Recall / F1을 계산합니다.
두 스크립트 모두 **Embedding 유사도**와 **LLM-as-a-Judge** 두 가지 방법론으로 동시에 평가합니다.

### 평가 단계 (Step) 개요

| Step | 스크립트 | 내용 |
|------|----------|------|
| Step 1 | `evaluate_entity.py` | 엔티티 추출 평가 (이름 매칭) |
| Step 2 | `evaluate_triple.py --mode soft` | 트리플 평가 — Subject + Object 매칭 |
| Step 2 | `evaluate_triple.py --mode full` | 트리플 평가 — Subject + Relation + Object 전체 매칭 |
| Step 3 | `evaluate_triple.py --include-implicit` | Implicit 트리플까지 gold 포함 |

---

### Step 1 — 엔티티 평가

```bash
# 기본 실행 (Embedding + LLM 동시 평가)
python evaluate_entity.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/

# 샘플 수 제한 (테스트용)
python evaluate_entity.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --limit 5

# 결과 JSON 저장
python evaluate_entity.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --output outputs/watson_uco_entity_metrics.json
```

#### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--results` | 필수 | 모델 추출 결과 JSON 파일 |
| `--ground-truth` | 필수 | GT 어노테이션 디렉터리 |
| `--threshold` | `0.75` | Embedding 매칭 임계값 |
| `--embedding-mode` | `.env` | Embedding backend (`local` \| `remote`) |
| `--embedding-model` | `.env` | Embedding 모델명 |
| `--embedding-base-url` | `.env` | Remote embedding base URL (`.../v1` 또는 `.../v1/embeddings`) |
| `--embedding-api-key` | `.env` | Remote embedding API key |
| `--llm-provider` | `.env` | LLM 판단에 사용할 provider (`openai` \| `anthropic` \| `gemini` \| `ollama`) |
| `--llm-model` | `.env` | LLM 모델명 |
| `--llm-base-url` | — | Ollama 등 로컬 LLM base URL |
| `--hitl` | off | Human-in-the-loop 모드 활성화 |
| `--limit` | 전체 | 처리할 최대 샘플 수 |
| `--output` | — | 결과 JSON 저장 경로 (상세 매칭 로그 포함) |

---

### 상세 평가 결과 분석 (JSON Output)

`--output` 옵션을 사용하여 결과를 저장하면, 각 샘플별로 **어떤 항목이 정답(Gold)이었고, 모델이 추출한 것 중 무엇이 매칭에 성공/실패했는지** 상세 내역을 확인할 수 있습니다. 이는 모델의 오답 원인(False Positive)을 분석하는 데 매우 유용합니다.

#### 결과 JSON 구조 예시

```json
{
  "samples": [
    {
      "id": "volt-typhoon-targets-us",
      "gold_list": ["Volt Typhoon", "US critical infrastructure"],
      "match_log": [
        {
          "prediction": "Volt Typhoon",
          "emb_match": { "is_correct": true, "gold": "Volt Typhoon", "score": 1.0 },
          "llm_judge": { "is_correct": true, "gold": "Volt Typhoon" }
        },
        {
          "prediction": "Unknown Hacker Group",
          "emb_match": { "is_correct": false, "gold": null, "score": 0.45 },
          "llm_judge": { "is_correct": false, "gold": null }
        }
      ]
    }
  ]
}
```

- **`gold_list`**: 해당 샘플에서 찾아야 했던 실제 정답 리스트 전체.
- **`match_log`**: 모델이 추출한 모든 항목(`prediction`)에 대한 개별 판정 결과.
    - `is_correct`: 정답으로 인정되었는지 여부.
    - `gold`: 매칭된 정답 항목 명칭.
    - `score`: (Embedding 한정) 유사도 점수.
- **오답 분석 팁**: `is_correct`가 `false`인 항목들을 모아보면 모델이 지어낸 정보(Hallucination)인지, 혹은 온톨로지 범위 밖의 정보를 뽑은 것인지 파악할 수 있습니다.

---

### Step 2 — 트리플 평가 (Soft / Full)

```bash
# Soft 매칭 (Subject + Object만 비교)
python evaluate_triple.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --mode soft

# Full 매칭 (Subject + Relation + Object 전체 비교)
python evaluate_triple.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --mode full

# 결과 저장
python evaluate_triple.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --mode soft \
  --output outputs/watson_uco_triple_soft_metrics.json
```

---

### Step 3 — Implicit 트리플 포함 평가

```bash
# Explicit + Implicit 트리플 모두 gold로 사용
python evaluate_triple.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --mode soft \
  --include-implicit

# Full 매칭 + Implicit 포함
python evaluate_triple.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --mode full \
  --include-implicit
```

#### 추가 옵션 (`evaluate_triple.py`)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `soft` | `soft` (S+O 매칭) \| `full` (S+R+O 매칭) |
| `--include-implicit` | off | Implicit 트리플도 gold pool에 추가 |
| `--embedding-mode` | `.env` | Embedding backend (`local` \| `remote`) |
| `--embedding-model` | `.env` | Embedding 모델명 |
| `--embedding-base-url` | `.env` | Remote embedding base URL (`.../v1` 또는 `.../v1/embeddings`) |
| `--embedding-api-key` | `.env` | Remote embedding API key |
| (나머지) | — | `evaluate_entity.py`와 동일 |

---

### HITL (Human-in-the-Loop) 모드

`--hitl` 플래그를 추가하면 각 매칭 후보를 하나씩 검토할 수 있습니다.
Embedding 점수와 LLM 판단 결과를 동시에 보여주며 최종 결정은 사람이 내립니다.

```bash
python evaluate_entity.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --hitl --limit 10

python evaluate_triple.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --mode soft --hitl --limit 5
```

#### HITL 인터랙션 커맨드

| 입력 | 동작 |
|------|------|
| `y` 또는 Enter | 현재 제안 매칭 수락 |
| `n` | 매칭 거부 (no match) |
| `<숫자>` | 다른 gold 항목 선택 (번호로 지정) |
| `s` | 이 항목은 LLM 판단에 위임 |
| `a` | 남은 모든 항목을 자동 처리 (LLM 판단) |
| `q` | 현재 샘플 중단, 다음 샘플로 이동 |

---

### 전체 비교 평가 (bash 루프)

```bash
# 모든 결과 파일에 대해 entity + triple 평가 일괄 실행
for f in outputs/*_results.json; do
  base=$(basename "$f" _results.json)

  python evaluate_entity.py \
    --results "$f" \
    --ground-truth datasets/ctinexus/annotation/ \
    --output "outputs/${base}_entity_metrics.json"

  python evaluate_triple.py \
    --results "$f" \
    --ground-truth datasets/ctinexus/annotation/ \
    --mode soft \
    --output "outputs/${base}_triple_soft_metrics.json"

  python evaluate_triple.py \
    --results "$f" \
    --ground-truth datasets/ctinexus/annotation/ \
    --mode soft --include-implicit \
    --output "outputs/${base}_triple_implicit_metrics.json"
done
```

---

## 권장 워크플로우

```bash
# 1. 환경 설치
bash setup.sh

# 2. API 키 설정
cp .env.sample .env
nano .env   # LLM_PROVIDER, OPENAI_API_KEY 등 입력

# 3. 스모크 테스트 (3개 샘플)
python run.py --model all --schema uco --limit 3

# 4. 전체 추출 실행
python run.py --model all --schema all --skip-existing

# 5. Step 1 — 엔티티 평가
python evaluate_entity.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --output outputs/watson_uco_entity_metrics.json

# 6. Step 2 — 트리플 평가 (Soft)
python evaluate_triple.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --mode soft \
  --output outputs/watson_uco_triple_soft_metrics.json

# 7. Step 3 — Implicit 포함 트리플 평가
python evaluate_triple.py \
  --results outputs/watson_uco_results.json \
  --ground-truth datasets/ctinexus/annotation/ \
  --mode soft --include-implicit \
  --output outputs/watson_uco_triple_implicit_metrics.json
```

---

## 환경변수 (.env)

루트 `.env`에 설정:

```env
# LLM Provider (legacy watson 추출에 사용)
LLM_PROVIDER=openai          # openai | ollama
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

# Ollama (로컬 LLM 사용 시)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# 평가용 LLM judge (evaluate_entity.py / evaluate_triple.py)
EVAL_LLM_PROVIDER=ollama
EVAL_LLM_MODEL=llama3.1:8b
EVAL_LLM_BASE_URL=http://localhost:11434

# 평가용 Embedding backend
EVAL_EMBEDDING_MODE=local    # local | remote
EVAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# EVAL_EMBEDDING_BASE_URL=http://192.168.100.2:8082/v1
# EVAL_EMBEDDING_API_KEY=

# watson-new 추출용 LLM / Embedding
WATSON_NEW_LLM_BASE_URL=http://192.168.100.2:8081/v1
WATSON_NEW_LLM_MODEL=qwen3.5-35b
WATSON_NEW_EMBEDDING_MODE=remote
WATSON_NEW_EMBEDDING_BASE_URL=http://192.168.100.2:8082/v1
WATSON_NEW_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# WATSON_NEW_EMBEDDING_API_KEY=
```

메모:

- `watson-new/eval.py`, `evaluate_entity.py`, `evaluate_triple.py` 는 모두 루트 `.env`를 읽습니다.
- `EVAL_EMBEDDING_MODE=remote` 로 두면 평가 스크립트의 embedding matcher도 remote endpoint를 사용합니다.
- `EVAL_LLM_BASE_URL` 또는 `OPENAI_BASE_URL`을 설정하면 평가용 LLM judge가 해당 OpenAI-compatible endpoint를 사용합니다.
- `WATSON_NEW_*` 값은 `watson-new` 추출 파이프라인용입니다. `run.py --model watson-new ...` 실행 시에도 같은 값을 사용합니다.
