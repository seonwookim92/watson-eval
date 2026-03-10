# GTIKG (CTIKG) Evaluation Usage Guide

이 문서는 LLM 기반의 Open Information Extraction(개방형 정보 추출) 모델인 GTIKG를 사용하여 비정형 사이버 위협 인텔리전스(CTI)에서 개체(Entity)와 트리플(Triplet)을 추출하는 방법을 설명합니다.

## 1. 기반 로직 상세 설명 (How It Works)

### 1) GTIKG 오리지널 로직
GTIKG(CTIKG)는 프롬프트 엔지니어링과 거대 언어 모델(LLM) 에이전트를 기반으로 동작하는 파이프라인입니다.
- **Schema-less OpenIE**: 텍스트를 입력받아 동사와 명사를 자유롭게 판별하여 `[Subject, Relation, Object]` 3가지 원소로 이루어진 자유 텍스트 형태의 트리플을 무한히 추출합니다.
- 특정 사이버 보안 온톨로지 스키마(예: UCO, STIX, MALOnt)에 얽매이지 않고, LLM이 문맥을 판단해 임의로 생성한 단어들을 기반으로 지식 그래프(Knowledge Graph)의 원천 데이터를 생성합니다.

### 2) 우리 실험 셋팅으로의 변환 (What We Changed)
우리의 평가 방법론(4단계: 1. NER, 2. Triple Extraction, 3. Entity Mapping, 4. Relation Mapping)에 맞춰 GTIKG를 벤치마킹 베이스라인으로 적용하기 위해 다음과 같은 조정을 거쳤습니다.
- **1, 2단계 추출 기능 통합 (`eval_gtikg.py`)**: 복잡한 Jupyer Notebook(`Knowledge Graph Construction.ipynb`) 내부의 코어 프롬프트 모델만을 파이썬 스크립트로 덜어내어 자동화된 평가 스크립트로 구축했습니다.
- **표준 JSON 포맷 출력 맞춤**: LLM이 문자열 텍스트로 내뱉는 `[SUBJECT:..., RELATION:..., OBJECT:...]` 정규표현식 데이터를 파싱하여 CTINexus 및 TTPDrill과 **정확히 동일한 구조(`extracted_entities`, `extracted_triplets`)로 렌더링**하도록 변환했습니다.
- **한계점 반영 (Ontology Mapping 불가)**: GTIKG는 "스키마 없는 자율 추출" 방식이므로 추출한 대상이 UCO나 STIX의 어떤 클래스인지(Entity Typing) 판별하는 로직이 전무합니다. 따라서 이 한계를 명확히 보여주기 위해 **모든 엔티티의 추출 결과의 `class` 값을 `null`로 강제 할당**하도록 파이프라인을 설계했습니다.


## 2. 사전 준비 (Setup)
다른 환경과 의존성이 충돌하지 않도록 GTIKG 전용 가상환경을 사용합니다.
```bash
# GTIKG 디렉터리로 이동
cd /Users/seonwookim/Documents/Programming/Security/cyber-ontology/eval/baselines/GTIKGResearch

# 가상환경 활성화
source .venv_gtikg/bin/activate
# 필요한 경우: pip install openai tqdm python-dotenv
```

루트의 상위 디렉터리(또는 `ctinexus/.env`)에 있는 `.env` 파일의 `OPENAI_API_KEY` 값을 읽어서 구동되므로, OpenAI API 키가 정상적으로 선언되어 있는지 확인하십시오.

## 3. 실행 방법 (Usage)
`eval_gtikg.py` 스크립트를 사용하여 평가를 진행합니다. 해당 스크립트는 149개의 CTI 보고서를 돌면서 GPT-4 기반으로 엔티티와 관계를 뽑습니다.

### 명령어 구조
```bash
python eval_gtikg.py [데이터_개수]
```

- **[데이터_개수]**: 처리할 리포트의 개수 (0이면 전체 데이터셋 149개 처리)

### 예시 (Examples)

#### 1) 소량 데이터 테스트
데이터 3개만 추출해보고 싶을 때:
```bash
python eval_gtikg.py 3
```

#### 2) 전체 데이터셋 평가
149개 전체 데이테셋에 대해 LLM 추출을 돌릴 때:
```bash
python eval_gtikg.py 0
```
*(주의: LLM API 호출에 비용과 시간이 소요될 수 있습니다.)*

## 4. 결과 파일 (Outputs)
실행 결과는 상위의 `eval/outputs/` 디렉토리에 JSON 형식으로 저장됩니다.
- `gtikg_baseline_results.json`

결과 파일을 확인해보면 `extracted_triplets`의 S-R-O는 자유롭게 잘 뽑히지만, 모델 자체가 스키마 매핑 능력이 없기 때문에 `extracted_entities`의 모든 `class` 항목은 명시적으로 `null` 값으로 비워져 출력됩니다. 이를 통해 3, 4단계(매핑) 영역에서의 타 베이스라인(CTINexus)과의 성능 차별성을 명확히 입증할 수 있습니다. 
