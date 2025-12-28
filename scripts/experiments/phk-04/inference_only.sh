#!/bin/bash
export TZ='Asia/Seoul'

BASE_OUTPUT_DIR="../../../outputs/T8091"

cleanup_cache() {
    echo ""
    echo "=========================================="
    echo "   🧹 캐시 정리 중..."
    echo "=========================================="

    CACHE_DIR="/data/ephemeral/home/.cache/huggingface/hub"

    if [ -d "$CACHE_DIR" ]; then
        echo "정리 전: $(du -sh $CACHE_DIR 2>/dev/null | cut -f1)"
        rm -rf ${CACHE_DIR}/*
        echo "정리 후: $(du -sh $CACHE_DIR 2>/dev/null | cut -f1)"
        echo "✅ 캐시 정리 완료"
    else
        echo "⚠️ 캐시 디렉토리 없음"
    fi
    echo ""
}

# ============================================================
# 🎯 실험 설정: 여기만 수정하세요!
# ============================================================
#
# 형식: "실험명:체크포인트:추가인자"
#
# 실험명: 학습 시 사용한 폴더명 (예: qwen3-32b-qlora-v5)
# 체크포인트: checkpoint-4682, best_model 등
# 추가인자: --temperature 0.7 같은 추가 옵션 (없으면 비워두기)
#
# 예시:
# "qwen3-32b-qlora-v5:checkpoint-4682:"              <- Epoch 1
# "qwen3-32b-qlora-v5:best_model:"                   <- Best model
# "qwen3-32b-qlora-v4.1:best_model:"                 <- 다른 실험
# "qwen3-32b-qlora-v5:best_model:--temperature 0.7"  <- 추가 인자
# ============================================================

EXPERIMENTS=(
    # === qwen3-32b-qlora-v5 실험 ===
    "qwen3-32b-qlora-v5:checkpoint-4682:"

    # === qwen3-32b-qlora-v4.1 실험 (비교용) ===
    # "qwen3-32b-qlora-v4.1:best_model:"          # v4.1 Best

    # === qwen2.5-32b-it-qlora-v4.2 실험 ===
    # "qwen2.5-32b-it-qlora-v4.2:best_model:"     # qwen2.5 v4.2
    # "qwen2.5-32b-it-qlora-v4.2:checkpoint-2000:" # qwen2.5 v4.2 ckpt

    # === Temperature 실험 (동일 모델, 다른 파라미터) ===
    # "qwen3-32b-qlora-v5:best_model:--temperature 0.7"
    # "qwen3-32b-qlora-v5:best_model:--temperature 1.5"
)

# ============================================================
# 여기 아래는 수정 불필요
# ============================================================

total=${#EXPERIMENTS[@]}
current=0

echo "=========================================="
echo "   🔍 추론 배치 실행"
echo "   총 ${total}개 추론 예정"
echo "=========================================="
echo "시작 시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 추론 목록 미리보기
echo "📋 추론 예정 목록:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name ckpt_name _ <<< "$exp"
    echo "  - ${exp_name} / ${ckpt_name}"
done

echo ""
echo "🧹 시작 전 기존 캐시 정리..."
cleanup_cache

for exp_config in "${EXPERIMENTS[@]}"; do
    current=$((current + 1))

    # : 기준으로 분리
    IFS=':' read -r exp_name ckpt_name additional_args <<< "$exp_config"

    # 자동으로 경로 생성
    model_path="${BASE_OUTPUT_DIR}/${exp_name}/${ckpt_name}"

    # 출력명 생성 (체크포인트명을 포함)
    if [ "$ckpt_name" = "best_model" ]; then
        output_name="${exp_name}"
    else
        # checkpoint-4682 -> ckpt-4682
        ckpt_suffix=$(echo "$ckpt_name" | sed 's/checkpoint-/ckpt-/')
        output_name="${exp_name}_${ckpt_suffix}"
    fi

    # 추가 인자가 있으면 출력명에 반영
    if [ -n "$additional_args" ]; then
        # --temperature 0.7 -> temp0.7
        args_suffix=$(echo "$additional_args" | sed 's/--//g' | sed 's/ //g' | sed 's/temperature/temp/')
        output_name="${output_name}_${args_suffix}"
    fi

    echo ""
    echo "=========================================="
    echo "   📊 추론 [$current/$total]: $output_name"
    echo "=========================================="
    echo "실험명: $exp_name"
    echo "체크포인트: $ckpt_name"
    echo "모델 경로: $model_path"
    echo "출력명: $output_name"
    echo "추가 인자: ${additional_args:-없음}"
    echo "추론 시작: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 모델 경로 존재 확인
    if [ ! -d "$model_path" ]; then
        echo "❌ 모델 경로가 존재하지 않습니다: $model_path"
        echo ""
        echo "💡 사용 가능한 경로 확인:"
        echo "   ls -la ${BASE_OUTPUT_DIR}/${exp_name}/"
        echo ""
        if [ -d "${BASE_OUTPUT_DIR}/${exp_name}" ]; then
            echo "실제 경로 목록:"
            ls -d ${BASE_OUTPUT_DIR}/${exp_name}/*/ 2>/dev/null || echo "  (체크포인트 없음)"
        else
            echo "  ⚠️ 실험 디렉토리 자체가 없음: ${BASE_OUTPUT_DIR}/${exp_name}"
        fi
        exit 1
    fi

    # 추론 실행
    python inference.py \
        --exp_name "$output_name" \
        --model_name "$model_path" \
        $additional_args

    inference_exit_code=$?

    if [ $inference_exit_code -ne 0 ]; then
        echo "❌ 추론 실패: $output_name (exit code: $inference_exit_code)"
        echo "   시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
        exit $inference_exit_code
    fi

    echo ""
    echo "✅ 추론 완료: $output_name"
    echo "   종료 시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"

    # 결과 파일 경로 출력
    output_file="${BASE_OUTPUT_DIR}/${output_name}/output.csv"
    if [ -f "$output_file" ]; then
        echo "   📁 결과 저장: $output_file"
        echo "   📊 예측 개수: $(tail -n +2 $output_file | wc -l)"
    fi

    # 다음 추론 준비
    if [ $current -lt $total ]; then
        echo ""
        echo "♻️  다음 추론 준비 중..."
        cleanup_cache
        echo "⏳ 5초 대기..."
        sleep 5
    fi
done

echo ""
echo "=========================================="
echo "   🎉 모든 추론 완료! ($current/$total)"
echo "   종료 시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "📊 결과 요약:"
for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name ckpt_name additional_args <<< "$exp_config"

    # 출력명 재생성 (동일 로직)
    if [ "$ckpt_name" = "best_model" ]; then
        output_name="${exp_name}"
    else
        ckpt_suffix=$(echo "$ckpt_name" | sed 's/checkpoint-/ckpt-/')
        output_name="${exp_name}_${ckpt_suffix}"
    fi

    if [ -n "$additional_args" ]; then
        args_suffix=$(echo "$additional_args" | sed 's/--//g' | sed 's/ //g' | sed 's/temperature/temp/')
        output_name="${output_name}_${args_suffix}"
    fi

    output_file="${BASE_OUTPUT_DIR}/${output_name}/output.csv"
    if [ -f "$output_file" ]; then
        count=$(tail -n +2 $output_file | wc -l)
        echo "  ✓ $output_name: $count 예측"
    else
        echo "  ✗ $output_name: 결과 파일 없음"
    fi
done
echo ""
