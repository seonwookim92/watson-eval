# TextItDown

TextItDown is a Python CLI that converts files into Markdown first, then turns Markdown into final plain text.

## 실행 방법

```bash
cd /mnt/d/ontology
python TextItDown/textitdown.py <input> [output.txt]
```

- `input`: 이미지/ PDF/ URL/ Office/ 텍스트 파일 경로
- `output.txt` 생략 시 `<input>.txt`로 저장

## 기본 설정

`TextItDown/config.json` 예시:

```json
{
  "ocr": {
    "model": "zai-org/GLM-OCR",
    "base_url": "http://192.168.100.2:8080",
    "request": {
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 4096,
      "stream": false
    },
    "image": {
      "detail": "high"
    },
    "prompt": {
      "system": "You are a document OCR assistant. Extract text from the image as accurately as possible and return only markdown.",
      "user": "Convert all readable content in this image to markdown.\nPreserve structure for tables, formulas, and code blocks when possible."
    }
  },
  "image_captioning": {
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "base_url": "http://192.168.100.2:8079/v1"
  },
  "llm": {
    "model": "openai/gpt-oss-120b",
    "base_url": "http://192.168.100.2:8081/v1"
  }
}
```

### OCR 엔드포인트(중요)

- GLM OCR HTTP fallback은 `POST /chat/completions`를 기본으로 시도하고, 환경에 따라 `/v1/chat/completions`도 함께 시도합니다.
- 404/405는 보통 GET 호출이나 경로 미스매치에서 발생합니다.
- `image` 블록에서 `detail: high`는 이미지 해상도 단위의 인식 품질 보정에 유리합니다.
- `request` 블록에서 `temperature=0.0`, `max_tokens`를 조절해 결과 재현성과 용량을 제어합니다.

## 현재 처리 파이프라인

1. 입력 형식 감지
2. Stage 1: Markdown 변환
   - 이미지: GLM-OCR
   - PDF: 페이지별 분기(텍스트 20자 미만 -> 이미지 OCR, 아니면 MarkItDown)
   - URL: HTML 다운로드 후 MarkItDown
   - Office/텍스트: MarkItDown
3. Stage 2-1: 이미지 캡션 (`![...](...)` 패턴)
4. Stage 2-2: Markdown 표 블록 텍스트화
5. Stage 2-3: 마크다운 서식 제거 후 Plain Text 저장

중간 결과는 `TextItDown/intermediate/{YYYYMMDD_HHMMSS}/{input_name}.md`에 자동 저장됩니다.

## 필수/권장 의존성

- `markitdown`
- `pymupdf`
- `requests`
- `openai` (선택: 설치 시 우선 사용)

`requirements.txt`:

```text
markitdown
pymupdf
requests
openai
```
