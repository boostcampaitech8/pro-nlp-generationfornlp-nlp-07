# Generation for NLP - 수능 문제 풀이 프로젝트

한국어 수능 시험(국어, 사회) 문제를 풀기 위한 AI 모델 개발 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 작은 규모의 모델로 수능 시험을 풀어보는 도전을 목표로 합니다. 한국어의 특성과 수능 시험의 특징을 바탕으로 수능에 특화된 AI 모델을 만들어 GPT, Claude, Gemini 같은 대형 모델들을 뛰어넘는 것을 목표로 합니다.

## 평가 방법

- **평가 지표**: Macro F1-score
- 각 선택지 클래스(1, 2, 3, 4, 5)별로 F1-score를 계산한 후 평균
- 불균형 데이터를 고려한 공정한 평가

## 프로젝트 구조

```
pro-nlp-generationfornlp-nlp-07/
├── .github/
│   ├── ISSUE_TEMPLATE/          # GitHub Issue 템플릿
│   └── workflows/               # CI/CD 워크플로우
├── data/                        # 데이터 파일 (.gitignore)
│   ├── train/
│   └── test/
├── notebooks/                   # 실험용 노트북 (선택사항)
│   └── experiments/            # 각 팀원의 실험 노트북
├── src/                         # 소스 코드
│   ├── config/                  # 설정 파일
│   ├── data/                    # 데이터 처리
│   ├── models/                  # 모델 로딩 및 설정
│   ├── training/                 # 훈련 관련
│   ├── inference/               # 추론 관련
│   └── utils/                   # 유틸리티 함수
├── scripts/                     # 실행 스크립트
│   ├── train.py                 # 훈련 스크립트
│   ├── inference.py             # 추론 스크립트
│   └── experiments/             # 각 팀원의 실험 스크립트
│       └── memberA/             # 팀원A의 실험 예시
├── experiments/                 # 실험 결과 및 로그
├── outputs/                     # 훈련 출력 (.gitignore)
├── submissions/                 # 제출 파일
├── requirements.txt             # 패키지 의존성
├── .env.example                 # 환경 변수 템플릿
└── README.md                    # 이 파일
```

## 설치 및 설정

### 1. 저장소 클론

이 저장소는 Private 저장소이므로 인증이 필요합니다. Personal Access Token을 사용하여 클론하세요.

```bash
git clone https://<username>:<personal-access-token>@github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-07.git
cd pro-nlp-generationfornlp-nlp-07
```

### 2. Git 사용자 정보 설정

클론 후 로컬 저장소에 사용자 정보를 설정하세요.

```bash
git config --local user.name "your-username"
git config --local user.email "your-email@example.com"
```

> **참고**: `--local` 옵션은 현재 저장소에만 적용됩니다. 모든 저장소에 적용하려면 `--global` 옵션을 사용하세요.

### 3. 가상환경 생성 및 활성화

```bash
python3.10 -m venv --system-site-packages .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

### 4. 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. 환경 변수 설정

`.env.example` 파일을 참고하여 `.env` 파일을 생성하고 Hugging Face 토큰을 설정하세요:

```bash
cp .env.example .env
# .env 파일을 편집하여 HF_TOKEN을 설정
```

`.env` 파일 내용:
```
HF_TOKEN=your_huggingface_token_here
HF_ORG=NLP-07-ODQA
```

## 사용 방법

### 기본 훈련

```bash
python scripts/train.py
```

### 추론

```bash
python scripts/inference.py --checkpoint <checkpoint_path> --test-data <test_data_path> --output <output_path>
```

또는 최신 체크포인트 사용:
```bash
python scripts/inference.py --output submissions/output.csv
```

## 팀원별 실험 가이드

### 팀원A가 다른 모델로 실험하는 경우

#### 1. 실험 스크립트 위치

```
scripts/experiments/memberA/
├── train_bert_model.py      # BERT 모델로 훈련
├── train_roberta_model.py   # RoBERTa 모델로 훈련
└── inference_bert_model.py  # BERT 모델로 추론
```

#### 2. 예시: `scripts/experiments/memberA/train_bert_model.py`

이 파일은 다른 모델로 실험하는 방법을 보여주는 예시입니다. 각 팀원은 이 파일을 참고하여 자신만의 실험 스크립트를 만들 수 있습니다.

주요 변경 사항:
- `MODEL_NAME` 변수만 변경하여 다른 모델 사용
- `get_lora_config()`의 `target_modules`를 모델에 맞게 조정
- 하이퍼파라미터를 스크립트 내에서 직접 설정

#### 3. 모델별 커스터마이징

모델마다 chat template이나 설정이 다르면 `src/models/model_loader.py`에 모델별 분기 처리를 추가할 수 있습니다.

#### 4. 권장 워크플로우

1. **실험 스크립트 생성**: `scripts/experiments/memberA/train_xxx_model.py`
2. **공통 모듈 재사용**: `src/`의 기존 함수 활용
3. **모델명만 변경**: `MODEL_NAME` 변수 수정
4. **하이퍼파라미터 조정**: 스크립트 내에서 직접 설정
5. **결과 저장**: `outputs/memberA_xxx_model/` 또는 `experiments/exp_xxx/`
6. **Hugging Face 업로드**: `NLP-07-ODQA/memberA-xxx-model-v1` 형식

#### 5. 요약

- **파일 위치**: `scripts/experiments/memberA/train_xxx_model.py`
- **공통 모듈 재사용**: `src/`의 기존 함수들
- **모델 변경**: `MODEL_NAME` 변수만 수정
- **추가 파일 필요성**: 모델별 특수 처리가 필요한 경우에만

## Hugging Face 연동

### 팀 조직 정보

- **조직명**: `NLP-07-ODQA`
- **URL**: https://huggingface.co/NLP-07-ODQA

### 모델 업로드 규칙

- 모델명: `NLP-07-ODQA/{model-type}-v{version}` 또는 `NLP-07-ODQA/{member-name}-{model-type}`
- 예시: `NLP-07-ODQA/gemma-ko-2b-lora-v1`, `NLP-07-ODQA/member1-bert-base`
- README에 실험 정보, 하이퍼파라미터, 성능 기록 (Macro F1-score)
- 태그: `korean`, `csat`, `nlp-competition`, `generation-for-nlp`

### 모델 업로드 예시

```python
from src.utils.hf_utils import upload_model_to_hf

upload_model_to_hf(
    checkpoint_path="outputs/gemma_ko_2b",
    model_name="NLP-07-ODQA/gemma-ko-2b-lora-v1",
    experiment_name="gemma-ko-2b-lora-v1"
)
```

## Git 워크플로우

### 브랜치 전략

- `main`: 안정화된 코드만 유지
- `feature/{작업자명}-{작업내용}`: 각 실험/기능별 브랜치

### 커밋 메시지

- 형식: `[타입] 간단한 설명`
- 타입: `feat`, `fix`, `experiment`, `docs`, `refactor`
- 예: `[experiment] Add BERT-based model with 0.85 F1-score`

### GitHub Issue 및 Project

- 실험 추적: GitHub Issue와 Project 보드 사용
- Issue 라벨: `experiment`, `model`, `preprocessing`, `evaluation`, `bug`, `enhancement`

## 주의사항

- **데이터 파일**: 모든 데이터 파일(`data/`, `*.csv`)은 Git에 포함되지 않습니다. 팀원 간 별도로 공유하거나 공유 스토리지를 사용하세요.
- **baseline_code.ipynb**: 원본 baseline 노트북은 Git에 포함하지 않습니다. 로컬에서만 참고용으로 사용하세요.
- **환경 변수**: `.env` 파일은 Git에 포함되지 않습니다. `.env.example`을 참고하여 로컬에서 생성하세요.

## 라이선스

[라이선스 정보를 추가하세요]

## 기여자

[팀원 목록을 추가하세요]

