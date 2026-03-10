# Evaluation Methodology

## 개요

본 평가는 두 개의 독립된 레이어로 구성된다.

- **Layer 1 — Extraction Quality**: 모든 모델(watson, ctinexus, ttpdrill, gtikg)을 CTINexus 벤치마크 기준으로 공정하게 비교한다.
- **Layer 2 — Schema Grounding Quality**: watson 고유의 온톨로지 매핑 품질을 별도 방법으로 평가한다.

---

## 데이터셋

**CTINexus** (`datasets/ctinexus/annotation/`, 151개 JSON)

각 파일의 구조:
```json
{
  "text": "...",
  "entities": [{"entity_name": "WannaCry", "entity_type": "Malware"}],
  "explicit_triplets": [{"subject": "WannaCry", "relation": "encrypts", "object": "files"}]
}
```

Ground truth는 CTINexus 자체 taxonomy(예: `"Malware"`, `"Threat Actor"`)를 사용하며, UCO/STIX 클래스 레이블은 포함되어 있지 않다. 이 점이 Layer 1과 Layer 2의 평가 설계를 분리하는 근본 이유다.

---

## Layer 1: Extraction Quality

### 대상

전체 모델: watson, ctinexus, ttpdrill, gtikg

### 1-1. Triple Extraction

예측 트리플 집합과 GT 트리플 집합 사이의 Precision / Recall / F1을 측정한다.

#### Matching 전략

두 가지를 모두 보고한다.

**Jaccard Matching** (threshold = 0.5)

트리플을 `"subject relation object"` 문자열로 직렬화한 뒤 토큰 집합의 Jaccard 유사도를 계산한다.

```
Jaccard(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|
```

- 장점: 빠르고 재현 가능
- 단점: 어휘 변형(`"uses"` vs `"employs"`)에 취약

**Embedding Matching** (threshold = 0.75)

트리플 문자열을 sentence-transformer로 인코딩한 뒤 코사인 유사도 행렬을 계산하고 greedy 매칭을 적용한다.

```
sim(pred_i, gold_j) = cosine(embed(pred_i), embed(gold_j))
```

동일 GT 트리플은 최대 한 번만 TP로 카운트된다.

- 장점: 의미적으로 같지만 다르게 표현된 트리플 포착 가능
- 단점: 임계값 선택에 따라 수치가 달라짐

#### 집계 방식

- **Micro F1**: 전체 샘플의 TP/FP/FN 합산 후 계산 (빈도 반영)
- **Macro F1**: 샘플별 F1을 계산한 뒤 평균 (샘플 균등 반영)

두 가지 모두 보고한다.

### 1-2. Entity Extraction

entity_name 텍스트만 비교한다 (class/type은 무시). Jaccard + Embedding 동일 방식 적용.

- **Entity Precision**: 예측된 entity 중 GT에 매칭되는 비율
- **Entity Recall**: GT entity 중 예측된 것의 비율
- **Entity F1**: 조화평균

---

## Layer 2: Schema Grounding Quality (watson 전용)

### 배경

CTINexus GT에는 UCO/STIX/MalOnt 클래스 레이블이 없다. 따라서 "watson이 `MaliciousCode`로 분류했는데 맞는가"를 GT와 직접 비교하는 것은 불가능하다. 이 레이어는 GT 없이 watson의 온톨로지 매핑 품질을 내재적(intrinsic)으로 측정한다.

### 2-1. Ontology Validity Rate

예측된 entity class 및 relation class가 실제 UCO/STIX/MalOnt 스펙에 정의된 term인지 확인한다.

```
Validity Rate = # 유효한 ontology term / 전체 예측 class 수
```

온톨로지 파일(`ontology/uco/`, `ontology/stix/`, `ontology/malont/`)에서 유효 term 목록을 추출하여 검증한다. 베이스라인은 formal ontology로 grounding하지 않으므로 이 지표가 해당되지 않는다.

### 2-2. LLM-as-Judge (Schema Correctness)

GPT-4를 judge로 사용하여, 추출된 entity/triple에 할당된 온톨로지 클래스가 원문 맥락에서 의미적으로 적합한지 평가한다.

**평가 프롬프트 구조:**
```
Context: [원문 CTI 텍스트]
Entity: "[entity_name]"
Predicted class: "[UCO/STIX class]" from [schema] ontology

Q: Is this ontology class semantically appropriate for the entity
   given the cybersecurity context? Rate 0–3:
   0 = Incorrect  1 = Partially correct  2 = Mostly correct  3 = Correct
   Provide a brief justification.
```

샘플링: 전체 샘플 중 30개를 무작위 선택하여 entity 및 relation class 각각 평가.

**보고 지표:**
- 평균 Judge Score (0–3)
- 클래스별 오류 패턴 분석 (정성)

### 2-3. Schema Coverage & Diversity

샘플당 사용된 distinct ontology class 수를 측정한다.

```
Coverage = # distinct classes used per sample (평균)
```

watson이 텍스트의 의미 구조를 얼마나 세밀하게 온톨로지로 표현하는지를 나타낸다.

---

## 베이스라인과 비교 범위

| 지표 | Watson | CTINexus | TTPDrill | GTIKG |
|---|:---:|:---:|:---:|:---:|
| Triple F1 (Jaccard) | ✓ | ✓ | ✓ | ✓ |
| Triple F1 (Embedding) | ✓ | ✓ | ✓ | ✓ |
| Entity F1 | ✓ | ✓ | ✓ | ✓ |
| Ontology Validity Rate | ✓ | — | — | — |
| LLM-judge Schema Score | ✓ | — | — | — |
| Schema Coverage | ✓ | — | — | — |

베이스라인 모델들은 formal ontology로 grounding된 class를 출력하지 않으므로 Layer 2 지표는 적용되지 않는다. 이는 watson의 핵심 기여를 드러내는 구조적 차이점이다.

---

## 논문에서의 Claim 구조

### Claim A — "의미론적으로 더 잘 뽑는다"

- 근거: Layer 1 Triple/Entity F1 (Jaccard + Embedding)
- watson ≥ baselines on CTINexus benchmark

### Claim B — "스키마를 정확하게 매핑한다"

- 근거: Layer 2 Validity Rate + LLM-judge Score
- 베이스라인은 이 평가 자체가 불가능 → formal ontology grounding의 유무 자체가 contribution

### Claim C — "기존 모델은 구조화된 KG를 생성하지 못한다"

- 근거: 베이스라인 출력 포맷 자체가 비정형 (relation_class 없음, 온톨로지 term 없음)
- watson은 UCO/STIX/MalOnt 준거 KG를 직접 생성 → downstream 활용 가능

---

## 한계 및 향후 작업

- **UCO/STIX 전용 annotated eval set 부재**: 소규모라도 도메인 전문가가 레이블링한 UCO/STIX gold set을 구축하면 Layer 2를 더 엄밀하게 평가할 수 있다.
- **LLM-judge 재현성**: GPT-4 버전, temperature 등에 따라 점수가 변동될 수 있다. Judge 프롬프트와 모델 버전을 논문에 명시해야 한다.
- **CTINexus 데이터 분포 편향**: 특정 위협 유형(랜섬웨어, APT)이 과대 대표될 수 있다. 세부 카테고리별 성능도 보고하는 것이 바람직하다.
