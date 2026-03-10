# TTPDrill-0.5 Evaluation Usage Guide

이 문서는 전통적인 NLP(Natural Language Processing) 기반 매칭 파이프라인인 TTPDrill을 사용하여 다양한 사이버 보안 온톨로지(UCO, STIX, MALOnt)에서 엔티티 및 관계 명칭을 추출하고 맵핑하는 방법을 설명합니다.

## 1. 기반 로직 상세 설명 (How It Works)

### 1) TTPDrill 오리지널 로직
TTPDrill은 원래 Stanford CoreNLP 구조 구문 분석기(Dependency Parser)와 AllenNLP의 의미역 결정(SRL, Semantic Role Labeling) 모델을 사용해 CTI 텍스트로부터 행동 정보를 추출합니다.
- **주어(Subject), 동사(What), 목적어(Where)** 등의 시맨틱 역할을 추출하여 텍스트 상의 개체 및 행위들을 선별합니다.
- 추출된 문자열들을 사전에 정의된 (MITRE ATT&CK 또는 Ghaith 온톨로지 테이블) **CSV 데이터베이스의 토큰들과 BM25(TF-IDF 변형) 유사도 알고리즘으로 비교**하여 가장 점수가 높은 클래스와 매핑합니다.

### 2) 우리 실험 셋팅으로의 변환 (What We Changed)
LLM(CTINexus 등)의 접근 방식과 비교하기 위한 **NLP 베이스라인**으로 사용하기 위해 TTPDrill의 코어 로직을 우리 데이터 구조에 맞게 대폭 수정했습니다.
- **NLP 파이프라인 현대화**: 원래 사용하던 무거운 Java 기반 Stanford CoreNLP 환경 의존성을 끊어내고, 더 가볍고 범용적인 Python 기반 **Spacy 라이브러리로 전면 교체(Migration)** 했습니다 (`relation_miner.py`). 이로 인해 환경 구성과 실행 속도가 크게 개선되었고, ROOT 동사를 정확히 포착하도록 파서를 재작성했습니다.
- **온톨로지 CSV 자동 변환**: UCO, STIX, MALOnt의 JSON 스키마를 TTPDrill의 BM25 모델이 이해할 수 있도록 구조화된 `ontology_details.csv` 데이터베이스 형태로 동적 생성하는 파이프라인을 구축했습니다 (`ontology_to_csv.py`).
- **BM25 매칭 치명적 버그 수정**: 원래 로직에 존재하던 추출 단어가 알파벳 문자 단위로 쪼개져 토큰화되는 치명적인 버그(`val.split() 누락`)를 수정하여 온톨로지 클래스 간 실제 명칭 매칭(Word-level matching)이 정상 동작하도록 해결했습니다.
- **통합 평가 환경 구축**: LLM 기반 파이프라인의 결과 파일 구조(`extracted_entities`, `extracted_triplets`)와 정확히 동일한 형식의 JSON을 출력하도록 `eval_ttpdrill.py`를 신규 작성하여 Fair Comparison(공정한 성능 비교)이 가능하도록 변경했습니다.


## 2. 사전 준비 (Setup)
다른 베이스라인(CTINexus)과 충돌하지 않도록 TTPDrill 전용 가상환경을 사용합니다. `baselines/TTPDrill-0.5` 디렉토리 안에 `.venv_ttpdrill` 이 이미 준비되어 있습니다.
```bash
# TTPDrill 디렉터리로 이동
cd /Users/seonwookim/Documents/Programming/Security/cyber-ontology/eval/baselines/TTPDrill-0.5

# 가상환경 활성화
source .venv_ttpdrill/bin/activate
# 필요한 경우: pip install -r requirements.txt (spacy, rank-bm25 등 포함)
```

## 3. 실행 방법 (Usage)
`eval_ttpdrill.py` 스크립트를 사용하여 NLP 기반 평가를 진행합니다. 해당 스크립트는 내부적으로 `configuration.py`의 파일 대상을 교체하고 TTPDrill 파이프라인을 호출합니다.

### 명령어 구조
```bash
python eval_ttpdrill.py [데이터_개수] [온톨로지_모드]
```

- **[데이터_개수]**: 처리할 리포트의 개수 (0이면 전체 데이터셋 149개 처리)
- **[온톨로지_모드]**: `baseline`(기본), `uco`, `stix`, `malont` 중 하나 선택

### 예시 (Examples)

#### 1) 소량 데이터 테스트
STIX 온톨로지에 대해 데이터 3개만 추출해보고 싶을 때:
```bash
python eval_ttpdrill.py 3 stix
```

#### 2) 전체 데이터셋 평가
UCO 스키마와 기반 로직을 매칭하여 149개 전체 데이터의 엔티티를 추출할 때:
```bash
python eval_ttpdrill.py 0 uco
```

#### 3) 특정 모드로 실행 및 결과 확인
새로 추가된 MALOnt 온톨로지로 실행할 때:
```bash
python eval_ttpdrill.py 3 malont
```

## 4. 결과 파일 (Outputs)
실행 결과는 상위의 `eval/outputs/` 디렉토리에 JSON 형식으로 저장됩니다. 구조는 CTINexus와 동일한 통합 형태를 띕니다.
- `ttpdrill_uco_results.json`
- `ttpdrill_stix_results.json`
- `ttpdrill_malont_results.json`
