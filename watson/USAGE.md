# Usage Guide

## 전제 조건

```bash
cd cyber-ontology-analyzer
pip install -r requirements.txt
cp .env.sample .env  # API 키 등 환경변수 설정
```

---

## 스크립트 개요

| 스크립트 | 역할 |
|----------|------|
| `eval.py` | CTI 텍스트 → 엔티티·트리플 추출 → `outputs/` 포맷 저장 |
| `evaluate.py` | `outputs/` 파일 + ground truth → P/R/F1 계산 |

---

## 1. eval.py — 추출

### 동작 방식

1. CTINexus annotation JSON 파일들을 로드
2. 각 텍스트를 청킹 (기본 4000 토큰, 400 오버랩)
3. MCP 서버를 통해 지정한 온톨로지 스키마로 엔티티·트리플 추출
4. `outputs/` baseline 파일과 동일한 list 포맷으로 저장

### 스키마 모드

#### `--schema uco` (기본값)
UCO(Unified Cyber Ontology) 스키마 사용.
엔티티를 `MaliciousTool`, `Organization`, `IPv4Address` 등 UCO 클래스로 매핑.
`relation_class`는 UCO ObjectProperty 단축명 (`performer`, `object`, `communicatesWith` 등).

```bash
python eval.py \
  --dataset ctinexus \
  --data-path ../../datasets/ctinexus/annotation/ \
  --output ../../outputs/ctinexus_uco_our_results.json \
  --schema uco
```

#### `--schema stix`
STIX 2.1 (OASIS) 스키마 사용.
엔티티를 `ThreatActor`, `Malware`, `Artifact` 등 STIX 타입으로 매핑.
`relation_class`는 STIX relationship type (`uses`, `targets`, `attributed-to` 등).

```bash
python eval.py \
  --dataset ctinexus \
  --data-path ../../datasets/ctinexus/annotation/ \
  --output ../../outputs/ctinexus_stix_our_results.json \
  --schema stix
```

#### `--schema malont`
MalOnt(Malware Ontology) 스키마 사용.
악성코드 중심 온톨로지로 매핑.

```bash
python eval.py \
  --dataset ctinexus \
  --data-path ../../datasets/ctinexus/annotation/ \
  --output ../../outputs/ctinexus_malont_our_results.json \
  --schema malont
```

### 샘플 테스트 (--limit)

전체 실행 전 소수 샘플로 동작 확인:

```bash
# 3개만, 상세 출력 포함
python eval.py \
  --dataset ctinexus \
  --data-path ../../datasets/ctinexus/annotation/ \
  --output /tmp/test_uco.json \
  --schema uco \
  --limit 3 --verbose

# STIX 5개
python eval.py \
  --dataset ctinexus \
  --data-path ../../datasets/ctinexus/annotation/ \
  --output /tmp/test_stix.json \
  --schema stix \
  --limit 5
```

### 출력 포맷 예시

```json
[
  {
    "file": "volt-typhoon-targets-us-critical-infrastructure.json",
    "text": "Volt Typhoon targets US critical infrastructure...",
    "ontology": "uco",
    "extracted_entities": [
      { "name": "Volt Typhoon", "class": "Organization" },
      { "name": "US critical infrastructure", "class": "Action" }
    ],
    "extracted_triplets": [
      {
        "subject": "Volt Typhoon",
        "relation": "targets",
        "relation_class": "object",
        "object": "US critical infrastructure"
      },
      {
        "subject": "Volt Typhoon",
        "relation": "uses living-off-the-land techniques",
        "relation_class": "performer",
        "object": "LOLBins"
      }
    ]
  }
]
```

### 옵션 레퍼런스

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | 필수 | `ctinexus` \| `ctikg` |
| `--data-path` | 필수 | annotation 디렉터리 경로 |
| `--output` | 필수 | 결과 JSON 저장 경로 |
| `--schema` | `uco` | `uco` \| `stix` \| `malont` |
| `--limit` | 전체 | 처리할 최대 샘플 수 |
| `--verbose` | off | LLM 입출력·그래프 상세 출력 |

---

## 2. evaluate.py — 평가

### 동작 방식

1. 추출 결과 파일(`outputs/*.json`) 로드
2. ground truth annotation 디렉터리 로드
3. `file` 필드로 예측-정답 매칭
4. 선택한 matching 전략으로 트리플·엔티티 비교
5. Precision / Recall / F1 출력 (Macro + Micro)

### 매칭 전략 모드

#### `--match-mode jaccard` (기본값)
토큰 단위 Jaccard 유사도로 매칭. 별도 모델 불필요, 가장 빠름.
기본 threshold: **0.5** (subject·relation·object 각각 비교).

```bash
python evaluate.py \
  --results ../../outputs/ctinexus_uco_our_results.json \
  --ground-truth ../../datasets/ctinexus/annotation/ \
  --match-mode jaccard

# threshold 조정 (더 엄격하게)
python evaluate.py \
  --results ../../outputs/ctinexus_uco_our_results.json \
  --ground-truth ../../datasets/ctinexus/annotation/ \
  --match-mode jaccard \
  --eval-threshold 0.7
```

#### `--match-mode embedding`
문장 임베딩 코사인 유사도로 매칭. 의미적으로 유사한 표현 인식.
기본 threshold: **0.75**. `sentence-transformers` 필요.

```bash
python evaluate.py \
  --results ../../outputs/ctinexus_uco_our_results.json \
  --ground-truth ../../datasets/ctinexus/annotation/ \
  --match-mode embedding

# threshold 조정
python evaluate.py \
  --results ../../outputs/ctinexus_uco_our_results.json \
  --ground-truth ../../datasets/ctinexus/annotation/ \
  --match-mode embedding \
  --eval-threshold 0.8
```

#### `--match-mode llm`
LLM이 예측-정답 쌍을 직접 판단. 가장 정확하나 시간·비용 소요.
OpenAI 또는 로컬 Ollama 사용 가능.

```bash
# Ollama (로컬)
python evaluate.py \
  --results ../../outputs/ctinexus_uco_our_results.json \
  --ground-truth ../../datasets/ctinexus/annotation/ \
  --match-mode llm \
  --eval-provider ollama \
  --eval-model llama3.1:8b \
  --eval-base-url http://localhost:11434

# OpenAI
python evaluate.py \
  --results ../../outputs/ctinexus_uco_our_results.json \
  --ground-truth ../../datasets/ctinexus/annotation/ \
  --match-mode llm \
  --eval-provider openai \
  --eval-model gpt-4o
```

### 메트릭 저장 (--output)

`--output`을 지정하면 per-sample 상세 메트릭도 함께 저장됩니다.

```bash
python evaluate.py \
  --results ../../outputs/ctinexus_uco_our_results.json \
  --ground-truth ../../datasets/ctinexus/annotation/ \
  --match-mode jaccard \
  --output ../../outputs/ctinexus_uco_our_metrics.json
```

저장 포맷:
```json
{
  "results_file": "ctinexus_uco_our_results.json",
  "matcher": "JaccardMatcher",
  "num_samples": 151,
  "triple_metrics": {
    "macro_precision": 0.62, "macro_recall": 0.58, "macro_f1": 0.59,
    "micro_precision": 0.65, "micro_recall": 0.60, "micro_f1": 0.62
  },
  "entity_metrics": { "..." },
  "samples": [
    {
      "id": "volt-typhoon-targets-us-critical-infrastructure",
      "predicted_triples": [...],
      "gold_triples": [...],
      "triple_metrics": { "precision": 0.8, "recall": 0.7, "f1": 0.75, "tp": 7, "predicted": 9, "gold": 10 },
      "predicted_entities": [...],
      "gold_entities": [...],
      "entity_metrics": { "..." }
    }
  ]
}
```

### 기존 baseline 일괄 평가 (비교용)

```bash
for f in ctinexus_baseline ctinexus_uco ctinexus_stix ctinexus_malont; do
  python evaluate.py \
    --results ../../outputs/${f}_results.json \
    --ground-truth ../../datasets/ctinexus/annotation/ \
    --match-mode jaccard \
    --output ../../outputs/${f}_metrics.json
done
```

### 옵션 레퍼런스

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--results` | 필수 | 평가할 결과 JSON 파일 경로 |
| `--ground-truth` | 필수 | CTINexus annotation 디렉터리 |
| `--match-mode` | `jaccard` | `jaccard` \| `embedding` \| `llm` |
| `--eval-threshold` | 자동 | jaccard=0.5, embedding=0.75 |
| `--eval-provider` | env | `openai` \| `ollama` (llm 모드 전용) |
| `--eval-model` | env | LLM 모델명 (llm 모드 전용) |
| `--eval-base-url` | env | Ollama 엔드포인트 (llm 모드 전용) |
| `--output` | 없음 | per-sample 상세 메트릭 저장 경로 |

---

## 권장 워크플로우

```bash
# 1) 스모크 테스트 (3개 샘플)
python eval.py \
  --dataset ctinexus \
  --data-path ../../datasets/ctinexus/annotation/ \
  --output /tmp/smoke_uco.json \
  --schema uco --limit 3 --verbose

python evaluate.py \
  --results /tmp/smoke_uco.json \
  --ground-truth ../../datasets/ctinexus/annotation/ \
  --match-mode jaccard

# 2) 전체 추출 (스키마별)
python eval.py --schema uco   --dataset ctinexus --data-path ../../datasets/ctinexus/annotation/ --output ../../outputs/ctinexus_uco_our_results.json
python eval.py --schema stix  --dataset ctinexus --data-path ../../datasets/ctinexus/annotation/ --output ../../outputs/ctinexus_stix_our_results.json
python eval.py --schema malont --dataset ctinexus --data-path ../../datasets/ctinexus/annotation/ --output ../../outputs/ctinexus_malont_our_results.json

# 3) 전체 평가 (우리 결과 + 기존 baseline 비교)
for f in ctinexus_uco_our ctinexus_stix_our ctinexus_malont_our \
          ctinexus_baseline ctinexus_uco ctinexus_stix ctinexus_malont; do
  python evaluate.py \
    --results ../../outputs/${f}_results.json \
    --ground-truth ../../datasets/ctinexus/annotation/ \
    --match-mode jaccard \
    --output ../../outputs/${f}_metrics.json
done
```
