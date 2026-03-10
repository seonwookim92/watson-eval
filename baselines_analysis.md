# 사이버 보안 온톨로지 베이스라인 분석 결과

사이버 보안 온톨로지 관련 실험을 위해 `@baselines` 폴더 내 5개 프로젝트를 분석한 결과입니다. 각 프로젝트가 4가지 평가 항목(NER, Triple 추출, Entity 클래스 맵핑, Relation 맵핑)을 어떻게 지원하는지 정리했습니다.

## 요약 테이블

| Baseline 프로젝트 | 1. NER | 2. Triple (S-R-O) | 3. Entity Mapping | 4. Relation Mapping | 주요 온톨로지/스키마 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **GTIKGResearch (CTIKG)** | ✅ | ✅ | ⚠️ 부분적 | ❌ | 자체 정규화 (LLM 기반) |
| **LADDER** | ✅ | ✅ | ✅ (ID 맵핑) | ⚠️ 부분적 | Mitre ATT&CK |
| **TTPDrill-0.5** | ✅ | ✅ | ✅ (ID 맵핑) | ✅ | Mitre ATT&CK |
| **TiKG** | ✅ | ✅ | ⚠️ 부분적 | ❌ | DNRTI/CyNER (고정 타입) |
| **ctinexus** | ✅ | ✅ | ✅ (16개 클래스) | ⚠️ 부분적 | 자체 16개 클래스 스키마 |

---

## 상세 분석 내용

### 1. GTIKGResearch (CTIKG)
- **추출 방식**: LLM(GPT-4, Yi 등)을 활용한 Multi-agent 시스템.
- **NER/Triple**: LLM 프롬프트를 통해 [Subject, Relation, Object] 형태의 트리플을 자유 형식으로 추출합니다.
- **Entity Mapping**: `generate_prompt_postprocess` 프롬프트에서 대명사를 특정 이름(예: CVE-xxx, XLoader)으로 치환하거나 접미사를 제거하는 등 '정규화(Normalization)'를 수행하지만, UCO와 같은 외부 정규 온톨로지 클래스로의 엄격한 맵핑은 수행하지 않습니다.
- **Relation Mapping**: 관계(Relation)에 대해서는 맵핑을 수행하지 않고 자연어 형태 그대로 유지합니다.

### 2. LADDER
- **추출 방식**: BiLSTM-CRF 기반 NER 및 지도 학습 기반 관계 추출.
- **NER/Triple**: 문장에서 엔티티를 뽑고 그들 사이의 관계를 추출하여 Triple을 생성합니다.
- **Entity Mapping**: 추출된 엔티티를 **Mitre ATT&CK Technique ID**와 맵핑하는 로직이 포함되어 있습니다 ([enterprise-techniques.csv](file:///Users/seonwookim/Documents/Programming/Security/cyber-ontology/eval/baselines/LADDER/attack_pattern/enterprise-techniques.csv) 활용).
- **Relation Mapping**: 관계 유형을 분류하지만, 온톨로지의 `ObjectProperty`와 맵핑하기보다는 공격 단계(Attack Step) 식별에 집중합니다.

### 3. TTPDrill-0.5
- **추출 방식**: Stanford CoreNLP/AllenNLP를 이용한 시맨틱 의존성 파싱(Semantic Dependency Parsing).
- **NER/Triple**: 문장 구조 분석을 통해 행동(Action)과 대상(Object)을 추출합니다.
- **Entity Mapping**: BM25 유사도 알고리즘을 사용하여 추출된 구성 요소를 **Mitre ATT&CK ID (T-code)**로 맵핑합니다 ([ontology/](file:///Users/seonwookim/Documents/Programming/Security/cyber-ontology/eval/baselines/TTPDrill-0.5/ontology_reader.py#139-148) 폴더 내 다수의 맵핑 파일 존재).
- **Relation Mapping**: 추출된 동사(Verb)를 Mitre의 Tactic/Technique 관련 행동으로 맵핑하는 과정이 포함되어 있어 4개 항목 중 가장 높은 수준의 온톨로지 준수율을 보입니다.

### 4. Threat-Intelligence-Knowledge-Graphs (TiKG)
- **추출 방식**: SecureBERT + BiLSTM + CRF 파이프라인.
- **NER/Triple**: DNRTI, CyNER 등의 데이터셋을 학습하여 엔티티와 관계를 추출합니다.
- **Entity Mapping**: 학습 데이터셋에 정의된 고정된 타입(예: Tool, Malware, Actor)으로 분류(Typing)를 수행하지만, 계층적 온톨로지 클래스 맵핑은 제한적입니다.
- **Relation Mapping**: 데이터셋 내 정의된 관계 유형으로 분류하며, 외부 온톨로지 스키마와의 직접적인 맵핑 로직은 코드상에서 확인되지 않습니다.

### 5. ctinexus
- **추출 방식**: LLM(OpenAI, Gemini 등)의 In-Context Learning (ICL) 활용.
- **NER/Triple**: 프롬프트를 통해 16가지 주요 보안 클래스를 준수하며 Triple을 추출합니다.
- **Entity Mapping**: [et.jinja](file:///Users/seonwookim/Documents/Programming/Security/cyber-ontology/eval/baselines/ctinexus/ctinexus/prompts/et.jinja) 프롬프트에 정의된 **16개 엔티티 클래스**(Malware, Campaign, Threat Actor, Vulnerability 등)로 모든 엔티티를 강제 맵핑합니다. 이는 사용자가 요구하는 '스키마 내 클래스 맵핑'에 가장 근접한 LLM 기반 방식입니다.
- **Relation Mapping**: 관계를 추출하지만, 관계명 자체를 정해진 `ObjectProperty` 목록과 맵핑하기보다는 두 엔티티 클래스 간의 관계를 서술하는 방식입니다.

---

## 실험 활용 제언
- **정밀한 ID 기반 맵핑**이 중요하다면: **TTPDrill-0.5**가 가장 적합합니다. 코드 내에 유사도 기반 맵핑 로직(`BM25Okapi`)이 잘 구현되어 있습니다.
- **LLM을 활용한 유연한 클래스 맵핑**을 원한다면: **ctinexus**를 추천합니다. 프롬프트([prompts/ie.jinja](file:///Users/seonwookim/Documents/Programming/Security/cyber-ontology/eval/baselines/ctinexus/ctinexus/prompts/ie.jinja), [prompts/et.jinja](file:///Users/seonwookim/Documents/Programming/Security/cyber-ontology/eval/baselines/ctinexus/ctinexus/prompts/et.jinja))만 수정하면 사용자께서 원하는 **UCO 스키마 클래스**로 즉시 변경하여 실험이 가능합니다.
- **추출 성능(F1 Score)** 자체가 중요하다면: 최신 전용 모델을 사용하는 **TiKG**나 **LADDER**의 NER 성능이 우수할 것으로 판단됩니다.
