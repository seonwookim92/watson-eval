# Ladder NER & RE → Watson-Eval Integration

## Changes Made

### Modified Files

| File | Change |
|------|--------|
| [setup.sh](../../setup.sh) | Added `setup_ladder_ner()` and `setup_ladder_re()` functions + updated `all` targets |
| [run.py](../../run.py) | Added `ladder_ner` and `ladder_re` to `MODELS` registry (schema-agnostic) |

### New Files

| File | Purpose |
|------|---------|
| [eval_ladder_ner.py](./ner/eval_ladder_ner.py) | CyNER NER wrapper — processes ctinexus dataset, outputs `{file, text, ontology, extracted_entities, extracted_triplets}` |
| [eval_ladder_re.py](./relation_extraction/eval_ladder_re.py) | RE inference wrapper — loads `150_best_1.pt` checkpoint, classifies 11 relation types, outputs same format |

## Output Format

Both scripts produce JSON matching `eval_ctinexus.py`:

```json
[{
  "file": "report_name.json",
  "text": "truncated text…",
  "ontology": "none",
  "extracted_entities": [{"name": "Entity1", "class": "Malware"}],
  "extracted_triplets": [{"subject": "E1", "relation": "targets", "relation_class": "targets", "object": "E2"}]
}]
```

## Verification

```
> python run.py --model all --list
Model        Schema     Output file
--------------------------------------------------
  watson     uco/stix/malont  →  3 jobs
  ctinexus   uco/stix/malont  →  3 jobs
  ttpdrill   uco/stix/malont  →  3 jobs
  gtikg      none             →  1 job
  ladder_ner none             →  1 job    ← NEW
  ladder_re  none             →  1 job    ← NEW
Total: 12 jobs

> python run.py --model ladder_ner ladder_re --dry-run --limit 3
Done  : 2/2 succeeded
```

## Execution Results

1. **`ladder_ner` (NER)**:
   - `python run.py --model ladder_ner` 실행 완료
   - CTINexus 데이터셋 73개 파일에 대해 **CyNER 기반 엔티티 추출**을 모두 정상 수행했습니다.
   - `outputs/ladder_ner_none_results.json` 생성 완료

2. **`ladder_re` (TRIPLE)**:
   - `python run.py --model ladder_re` 실행 완료 (12s)
   - Ladder 자체 테스트 데이터셋(`150` dataset)에 대해 관계 추출(inference) 수행
   - 출력 예: `targets`, `indicates`, `discoveredIn` 등 11개 관계 타입 중 알맞은 트리플 추출 성공
   - `<e1>` 태그 기반의 엔티티 추출용 정규식 버그를 수정하여 정상 동작 확인

## Evaluation Scripts (evaluate_*.py) 동작 분석

1. **`evaluate_entity.py` (for NER)**:
   - `ladder_ner` 결과에 대해 실행이 가능하지만, 스크립트 내부에 **LLM 심사관(LLM-as-a-Judge)** 로직이 포함되어 있습니다.
   - 현재 설정이 로컬 `ollama`로 잡혀 있어서, Ollama 서버가 켜져 있지 않으면 실행 도중 Connection Error가 발생합니다. (Ollama 구동 후 정상 평가 가능)

2. **`evaluate_triple.py` (for RE)**:
   - `ladder_re` 결과는 이 스크립트로 평가할 수 **없습니다**.
   - 이유: `evaluate_triple.py`는 `datasets/ctinexus/annotation/` 디렉토리 안의 파일 이름과 매칭하여 Ground Truth를 비교합니다. 하지만 `ladder_re`는 ctinexus 대신 독자적인 `150` 데이터셋(문장 단위 학습셋)을 사용하므로, 비교할 Ground Truth 파일이 매칭되지 않아 전체 스킵(Skipped) 처리됩니다.

## Usage

```bash
# Setup (bash / Linux / WSL)
bash setup.sh ladder_ner ladder_re

# Run pipeline
python run.py --model ladder_ner ladder_re
```
