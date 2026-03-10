# CTINexus Evaluation Usage Guide

이 문서는 CTINexus를 사용하여 다양한 사이버 보안 온톨로지(Baseline, UCO, STIX, MALOnt)에서 엔티티 및 관계를 추출하는 방법을 설명합니다.

## 1. 기반 로직 상세 설명 (How It Works)

### 1) CTINexus 오리지널 로직
CTINexus는 LLM(대형 언어 모델)을 프롬프트 엔지니어링 기반으로 활용하여 비정형 위협 인텔리전스(CTI) 텍스트에서 지식을 추출하는 시스템입니다.
- **Information Extraction (IE)**: 미리 정의된 사이버 보안 코어 카테고리를 프롬프트로 주어, 주어(Subject) - 관계(Relation) - 목적어(Object) 형태의 트리플을 추출합니다.
- **Entity Typing (ET)**: 추출된 텍스트 엔티티를 특정 온톨로지 개념이나 MITRE TTPs 등에 맵핑하는 작업을 수행합니다.

### 2) 우리 실험 셋팅으로의 변환 (What We Changed)
LLM의 추론 능력을 최대한 활용하여 UCO, STIX, MALOnt 같은 거대한 온톨로지를 다루도록 파이프라인을 재설계했습니다.
- **대규모 스키마 프롬프트 주입**: 기존 CTINexus처럼 분절된 사전을 쓰는 대신, UCO, STIX, MALOnt의 전체 Class / Property 계층 구조를 요약하여 시스템 프롬프트에 통째로 주입했습니다.
- **End-to-End 매핑**: 별도의 외부 텍스트 매칭 알고리즘을 타지 않고, LLM이 문맥을 평가하여 해당 엔티티가 "UCO의 어떤 Class인지", 관계가 "어떤 Property인지" 프롬프트 내부에서 직접 추론하고 할당(Assign)하도록 강제했습니다.
- **Few-Shot Examples 맞춤 교체**: 각 온톨로지(UCO, STIX, MALOnt)에 최적화된 3개의 데모 샷을 작성하여 모델이 온톨로지 특이적인 Entity와 Triplet을 어떻게 뽑아야 하는지 가이드를 제시했습니다.


## 2. 사전 준비 (Setup)
먼저 가상 환경을 활성화하고 필요한 라이브러리를 설치해야 합니다.
```bash
source .venv/bin/activate
# 필요한 경우: pip install -r requirements.txt
```

`.env` 파일에 `OPENAI_API_KEY`와 `OPENAI_MODEL` (기본값: gpt-4o-mini)이 설정되어 있는지 확인하세요.

## 3. 실행 방법 (Usage)
`eval_ctinexus.py` 스크립트를 사용하여 평가를 진행합니다.

### 명령어 구조
```bash
python eval_ctinexus.py [데이터_개수] [온톨로지_모드]
```

- **[데이터_개수]**: 처리할 리포트의 개수 (0이면 전체 데이터셋 처리)
- **[온톨로지_모드]**: `baseline`, `uco`, `stix`, `malont` 중 하나 선택

### 예시 (Examples)

#### 1) 소량 샘플 테스트 (개수 제한)
UCO 온톨로지로 상위 3개 데이터만 테스트해보고 싶을 때:
```bash
python eval_ctinexus.py 3 uco
```

#### 2) 전체 데이터셋 평가
STIX 온톨로지로 모든 데이터를 평가할 때:
```bash
python eval_ctinexus.py 0 stix
```

#### 3) 특정 모드로 실행 및 결과 확인
새로 추가된 MALOnt 온톨로지로 실행할 때:
```bash
python eval_ctinexus.py 3 malont
```

## 4. 결과 파일 (Outputs)
실행 결과는 `eval/outputs/` 디렉토리에 JSON 형식으로 저장되며, 표준화된 형식(`extracted_entities`, `extracted_triplets`)을 따릅니다.
- `ctinexus_uco_results.json`
- `ctinexus_stix_results.json`
- `ctinexus_malont_results.json`

## 5. 프롬프트 재생성 (개발용)
온톨로지 스키마가 변경되어 프롬프트를 다시 만들고 싶을 때는 다음 스크립트를 실행하세요.
```bash
python /tmp/generate_all_prompts.py
```
