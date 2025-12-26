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

# 실험 목록 (exp_name:model_name:additional_args)
experiments=(
    "qwen3-32b-qlora-v4.1:unsloth/Qwen3-32B-bnb-4bit:--lora_r 16 --lora_alpha 32"
    "qwen3-32b-qlora-v4.2:unsloth/Qwen3-32B-bnb-4bit:--lora_r 32 --lora_alpha 64"
    "qwen3-32b-qlora-v4.3:unsloth/Qwen3-32B-bnb-4bit:--lora_r 64 --lora_alpha 128"
)

total=${#experiments[@]}
current=0
previous_model=""

echo "=========================================="
echo "   🚀 실험 시작 (학습 + 추론)"
echo "   총 ${total}개 실험 예정"
echo "=========================================="

echo "🧹 시작 전 기존 캐시 정리..."
cleanup_cache

for exp in "${experiments[@]}"; do
    current=$((current + 1))
    
    # : 기준으로 exp_name과 model_name 분리
    IFS=':' read -r exp_name model_name additional_args <<< "$exp"
    
    echo ""
    echo "=========================================="
    echo "   📊 실험 [$current/$total]: $exp_name"
    echo "=========================================="
    echo "모델: $model_name"
    echo "추가 인자: $additional_args"
    echo ""
    
    # ========== 학습 단계 ==========
    echo "🔥 [1/2] 학습 시작"
    echo "시작 시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
    
    python train.py \
        --exp_name "$exp_name" \
        --model_name "$model_name" \
        $additional_args
    
    train_exit_code=$?
    
    if [ $train_exit_code -ne 0 ]; then
        echo "❌ 학습 실패: $exp_name (exit code: $train_exit_code)"
        exit $train_exit_code
    fi
    
    echo "✅ 학습 완료: $exp_name"
    echo "종료 시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
    
    # 학습 후 메모리 정리를 위한 대기
    echo ""
    echo "⏳ 메모리 정리를 위해 10초 대기..."
    sleep 10
    cleanup_cache
    
    # ========== 추론 단계 ==========
    echo ""
    echo "🔍 [2/2] 추론 시작"
    echo "시작 시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
    
    # 학습된 모델 경로
    trained_model_path="${BASE_OUTPUT_DIR}/${exp_name}/best_model"
    
    # 모델 경로 존재 확인
    if [ ! -d "$trained_model_path" ]; then
        echo "❌ 모델 경로가 존재하지 않습니다: $trained_model_path"
        exit 1
    fi
    
    echo "학습된 모델 경로: $trained_model_path"
    
    python inference.py \
        --exp_name "$exp_name" \
        --model_name "$trained_model_path"
    
    inference_exit_code=$?
    
    if [ $inference_exit_code -ne 0 ]; then
        echo "❌ 추론 실패: $exp_name (exit code: $inference_exit_code)"
        exit $inference_exit_code
    fi
    
    echo "✅ 추론 완료: $exp_name"
    echo "종료 시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
    
    # ========== 다음 실험 준비 ==========
    if [ $current -lt $total ]; then
        next_exp="${experiments[$current]}"
        IFS=':' read -r next_exp_name next_model_name <<< "$next_exp"
        
        echo ""
        echo "♻️  다음 실험 준비 중..."
        cleanup_cache
        
        if [ "$model_name" != "$next_model_name" ]; then
            echo "🔄 다음 실험은 다른 모델 사용 (${next_model_name})"
        else
            echo "♻️  다음 실험도 같은 모델 사용"
        fi
        
        echo "⏳ 다음 실험까지 10초 대기..."
        sleep 10
    fi
done

echo ""
echo "=========================================="
echo "   🎉 모든 실험 완료! ($current/$total)"
echo "   종료 시간: $(TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
