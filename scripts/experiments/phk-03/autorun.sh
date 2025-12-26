#!/bin/bash

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

# 실험 목록 (exp_name:model_name 형식)
experiments=(
    "qwen2.5-32b-it-qlora-v1:unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    "yi-34b-chat-qlora-v1:unsloth/yi-34b-chat-bnb-4bit"
)

total=${#experiments[@]}
current=0
previous_model=""

echo "=========================================="
echo "   🚀 실험 시작"
echo "   총 ${total}개 실험 예정"
echo "=========================================="

echo "🧹 시작 전 기존 캐시 정리..."
cleanup_cache

for exp in "${experiments[@]}"; do
    current=$((current + 1))
    
    # : 기준으로 exp_name과 model_name 분리
    IFS=':' read -r exp_name model_name <<< "$exp"
    
    echo ""
    echo "=========================================="
    echo "   📊 실험 [$current/$total]: $exp_name"
    echo "=========================================="
    echo "모델: $model_name"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 학습 및 추론 실행
    python rebuilding_args.py \
        --exp_name "$exp_name" \
        --model_name "$model_name"
    
    exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "✅ 실험 완료: $exp_name"
    else
        echo "❌ 실험 실패: $exp_name (exit code: $exit_code)"
        # 실패 시 중단하려면 아래 주석 해제
        # exit $exit_code
    fi
    echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 다음 실험이 있는지 확인
    if [ $current -lt $total ]; then
        # 다음 실험의 모델명 가져오기
        next_exp="${experiments[$current]}"
        IFS=':' read -r next_exp_name next_model_name <<< "$next_exp"
        
        # 현재 모델과 다음 모델이 다르면 캐시 정리
        if [ "$model_name" != "$next_model_name" ]; then
            echo "🔄 다음 실험은 다른 모델 사용 (${next_model_name})"
            cleanup_cache
        else
            echo "♻️  다음 실험도 같은 모델 사용 - 캐시 유지"
            echo ""
        fi
        
        echo "⏳ 다음 실험까지 10초 대기..."
        sleep 10
    fi
done

echo ""
echo "=========================================="
echo "   🎉 모든 실험 완료! ($current/$total)"
echo "   종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
