⏺ Embedding 평가 방식 상세설명

  1. 트리플을 하나의 문자열로 변환                                                                                                                                         
   
  예측값과 정답을 각각 단일 문자열로 직렬화합니다.                                                                                                                         
                                                                                                                                                                         
  예측: {subject: "Google's Threat Analysis Group", relation: "discovered", object: "0-day exploit chain for iPhones"}
  → "Google's Threat Analysis Group discovered 0-day exploit chain for iPhones"

  정답: {subject: "Google's Threat Analysis Group (TAG)", relation: "discovered", object: "an in-the-wild 0-day exploit chain"}
  → "Google's Threat Analysis Group (TAG) discovered an in-the-wild 0-day exploit chain"

  2. 전체를 한 번에 인코딩 (배치)

  all_strings = [pred_1, pred_2, ..., pred_N, gold_1, gold_2, ..., gold_M]
  embeddings = sentence_transformer.encode(all_strings)
  # 각 문자열 → 384차원 벡터

  3. 코사인 유사도 행렬 계산

  예측 N개 × 정답 M개의 모든 쌍에 대해 코사인 유사도를 계산합니다.

  sim(pred_i, gold_j) = (pred_i · gold_j) / (|pred_i| × |gold_j|)

  예시 행렬 (threshold=0.75):

                gold_1(TAG+chain)  gold_2(Intellexa)  gold_3(Predator)
  pred_1(TAG)       0.91 ✓             0.21               0.15
  pred_2(Intellexa) 0.18               0.89 ✓             0.22
  pred_3(Predator)  0.12               0.19               0.95 ✓

  4. Greedy 매칭 (정답은 한 번만 사용)

  유사도 높은 순으로 정렬 후 탐욕적으로 매칭합니다. 정답 하나는 최대 한 번만 TP로 카운트됩니다.

  Step 1: (pred_3, gold_3) sim=0.95 → TP, gold_3 사용 완료
  Step 2: (pred_1, gold_1) sim=0.91 → TP, gold_1 사용 완료
  Step 3: (pred_2, gold_2) sim=0.89 → TP, gold_2 사용 완료

  5. P/R/F1 계산

  TP        = 3 (threshold 0.75 이상인 매칭 수)
  Predicted = 4 (예측 트리플 총 수)
  Gold      = 6 (정답 트리플 총 수)

  Precision = 3/4 = 0.75
  Recall    = 3/6 = 0.50
  F1        = 2×(0.75×0.50)/(0.75+0.50) = 0.60

  ---
  Jaccard vs Embedding 차이

  같은 쌍에서:

  ┌───────────────────────────────────────────────────────────────────────────┬─────────────┬─────────────┐
  │                                   비교                                    │   Jaccard   │  Embedding  │
  ├───────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
  │ "0-day exploit chain for iPhones" vs "an in-the-wild 0-day exploit chain" │ 0.33 (miss) │ 0.91 (hit)  │
  ├───────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
  │ "SynAck abuses NTFS" vs "SynAck abuses NTFS transactions"                 │ 0.75 (hit)  │ 0.97 (hit)  │
  ├───────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
  │ "CVE-2023-41991" vs "bugs in iOS 16.7"                                    │ 0.0 (miss)  │ 0.31 (miss) │
  └───────────────────────────────────────────────────────────────────────────┴─────────────┴─────────────┘

  Jaccard는 단어가 얼마나 겹치느냐만 보기 때문에 paraphrase를 전혀 잡지 못합니다. Embedding은 의미 공간에서의 거리를 측정하므로 다르게 표현된 같은 사실을 인식합니다.

  ---
  코드 흐름 (실제 코드 기준)

  core/eval/matchers.py:EmbeddingMatcher._batch_match():

  def _batch_match(self, pred_strs, gold_strs) -> int:
      # 1. 한 번에 인코딩
      all_embs = self.model.encode(pred_strs + gold_strs)
      pred_embs = all_embs[:len(pred_strs)]
      gold_embs = all_embs[len(pred_strs):]

      # 2. 유사도 행렬 (모든 pred × gold 쌍)
      scores = {
          (pi, gj): cosine(pred_embs[pi], gold_embs[gj])
          for pi in range(len(pred_embs))
          for gj in range(len(gold_embs))
      }

      # 3. 높은 순으로 greedy 매칭
      return _greedy_match(scores, threshold=self.threshold)

  Entity 평가도 동일 방식인데, 트리플 문자열 대신 entity_name 문자열만 사용합니다.