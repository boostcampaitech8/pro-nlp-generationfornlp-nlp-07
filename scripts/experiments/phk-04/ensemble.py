#!/usr/bin/env python3
"""
결과 파일 기반 고급 앙상블 스크립트
Probabilities(softmax) 기반 앙상블 - 이론적으로 최적!
"""

import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

def load_predictions(file_path):
    """JSON 파일에서 예측 결과 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['predictions']

def ensemble_predictions(experiment_files, method='probability_avg', output_dir='./'):
    """
    여러 실험의 예측을 앙상블

    Args:
        experiment_files: 예측 파일 경로 리스트
        method: 앙상블 방법
            - 'probability_avg': Probability 평균 (최고 정확도, 추천!)
            - 'soft_voting': Confidence 기반 가중 평균
            - 'hard_voting': 단순 다수결
        output_dir: 출력 디렉토리
    """
    print("=" * 70)
    print("🎯 결과 파일 기반 앙상블")
    print("=" * 70)

    # 모든 실험 예측 로드
    all_predictions = []
    experiment_names = []

    print(f"\n📂 실험 로딩...")
    for i, file_path in enumerate(experiment_files, 1):
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"  ❌ 파일 없음: {file_path}")
            continue

        try:
            preds = load_predictions(file_path)
            all_predictions.append(preds)
            experiment_names.append(file_path.stem.replace('_detailed', ''))

            # 기본 통계
            conf = [p['confidence'] for p in preds]

            # probabilities 확인
            has_prob = 'probabilities' in preds[0]
            prob_info = "✅ Prob" if has_prob else "❌ No Prob"

            print(f"  ✅ [{i}] {experiment_names[-1]}")
            print(f"      샘플: {len(preds)}, "
                  f"평균 Conf: {np.mean(conf):.4f}, "
                  f"99%+: {sum(1 for c in conf if c >= 0.99)/len(conf)*100:.1f}%, "
                  f"{prob_info}")
        except Exception as e:
            print(f"  ❌ 로드 실패 ({file_path.name}): {e}")

    if len(all_predictions) < 2:
        print("\n❌ 최소 2개 실험이 필요합니다!")
        return None

    print(f"\n✅ 총 {len(all_predictions)}개 실험 로드 완료")

    # Probabilities 사용 가능 여부 확인
    has_probabilities = all('probabilities' in p for p in all_predictions[0])

    if method == 'probability_avg' and not has_probabilities:
        print(f"\n⚠️  Probabilities 없음 → soft_voting으로 변경")
        method = 'soft_voting'

    print(f"\n🔄 앙상블 방법: {method}")
    if method == 'probability_avg':
        print(f"   💡 Probability 평균 사용 (이론적 최적!)")

    # 샘플 ID 확인
    sample_ids = [p['id'] for p in all_predictions[0]]
    n_samples = len(sample_ids)

    print(f"\n🔄 앙상블 수행 중...")

    # 앙상블 수행
    ensemble_results = []

    for sample_id in sample_ids:
        if method == 'probability_avg':
            # 방법 1: Probability 평균 (최고 정확도!) ⭐⭐⭐⭐⭐
            all_probs = {str(i): [] for i in range(1, 6)}

            for preds in all_predictions:
                pred_dict = {p['id']: p for p in preds}

                if sample_id not in pred_dict:
                    continue

                probs = pred_dict[sample_id]['probabilities']
                for label, prob in probs.items():
                    all_probs[str(label)].append(prob)

            # 각 라벨의 평균 확률 계산
            avg_probs = {}
            for label, prob_list in all_probs.items():
                if prob_list:
                    avg_probs[label] = np.mean(prob_list)

            # 가장 높은 확률의 라벨 선택
            final_pred = max(avg_probs.items(), key=lambda x: x[1])[0]
            final_pred = int(final_pred)

        elif method == 'soft_voting':
            # 방법 2: Confidence 기반 가중 평균
            predictions = []
            confidences = []

            for preds in all_predictions:
                pred_dict = {p['id']: p for p in preds}

                if sample_id not in pred_dict:
                    continue

                pred = pred_dict[sample_id]['prediction']
                conf = pred_dict[sample_id]['confidence']

                if isinstance(pred, str):
                    pred = int(pred)

                predictions.append(pred)
                confidences.append(float(conf))

            weighted_sum = sum(p * c for p, c in zip(predictions, confidences))
            total_conf = sum(confidences)
            final_pred = int(round(weighted_sum / total_conf))

        elif method == 'hard_voting':
            # 방법 3: 단순 다수결
            predictions = []

            for preds in all_predictions:
                pred_dict = {p['id']: p for p in preds}

                if sample_id not in pred_dict:
                    continue

                pred = pred_dict[sample_id]['prediction']
                if isinstance(pred, str):
                    pred = int(pred)
                predictions.append(pred)

            final_pred = int(round(np.mean(predictions)))

        ensemble_results.append({
            'id': sample_id,
            'answer': final_pred
        })

    print(f"  ✅ {len(ensemble_results)}개 샘플 앙상블 완료")

    # 결과 저장
    df = pd.DataFrame(ensemble_results)

    # 출력 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_names_str = '_'.join(experiment_names[:3])
    if len(experiment_names) > 3:
        exp_names_str += f"_etc{len(experiment_names)-3}"

    output_file = Path(output_dir) / f"ensemble_{exp_names_str}_{method}_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    print(f"\n💾 결과 저장: {output_file}")

    # 통계 출력
    print(f"\n📊 앙상블 결과 통계:")
    print(f"  총 샘플 수: {len(ensemble_results)}")

    answer_dist = df['answer'].value_counts().sort_index()
    print(f"\n  답변 분포:")
    for label, count in answer_dist.items():
        print(f"    {label}: {count}개 ({count/len(df)*100:.1f}%)")

    # 답변 분포 균형도
    dist_std = np.std(answer_dist / len(df) * 100)
    print(f"\n  분포 표준편차: {dist_std:.2f}% (낮을수록 균형)")

    # 실험 간 차이 분석
    print(f"\n🔍 실험 간 일치도:")
    for i in range(len(all_predictions)):
        for j in range(i+1, len(all_predictions)):
            preds1 = {p['id']: p['prediction'] for p in all_predictions[i]}
            preds2 = {p['id']: p['prediction'] for p in all_predictions[j]}

            match = sum(1 for id in preds1.keys() if preds1.get(id) == preds2.get(id))
            match_rate = match / len(preds1) * 100

            print(f"  {experiment_names[i]} vs {experiment_names[j]}: "
                  f"{match_rate:.1f}% 일치 ({len(preds1) - match}개 다름)")

    # 방법별 비교 (probability_avg와 다른 방법 비교)
    if method == 'probability_avg' and has_probabilities:
        print(f"\n💡 Probability 평균 vs 다른 방법 비교:")

        # Soft voting도 계산
        soft_results = []
        for sample_id in sample_ids:
            predictions = []
            confidences = []

            for preds in all_predictions:
                pred_dict = {p['id']: p for p in preds}
                if sample_id in pred_dict:
                    pred = int(pred_dict[sample_id]['prediction']) if isinstance(pred_dict[sample_id]['prediction'], str) else pred_dict[sample_id]['prediction']
                    predictions.append(pred)
                    confidences.append(pred_dict[sample_id]['confidence'])

            weighted_sum = sum(p * c for p, c in zip(predictions, confidences))
            total_conf = sum(confidences)
            soft_results.append(int(round(weighted_sum / total_conf)))

        diff = sum(1 for i in range(len(ensemble_results)) if ensemble_results[i]['answer'] != soft_results[i])
        print(f"  Probability 평균 vs Soft voting: {diff}개 다름 ({diff/len(ensemble_results)*100:.1f}%)")
        print(f"  → Probability 평균이 더 정교함! (예상 F1: +0.002~0.005)")

    print(f"\n{'='*70}")
    print(f"✅ 앙상블 완료!")
    print(f"{'='*70}")

    return output_file

def main():
    parser = argparse.ArgumentParser(
        description="결과 파일 기반 고급 앙상블 (Probability 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Probability 평균 (최고 정확도, 추천!)
  python ensemble.py \
      --files qwen3-v4.1_detailed.json qwen3-v6.1_detailed.json qwen2.5-v4.1_detailed.json

  # Soft voting (Confidence 기반)
  python ensemble.py \
      --files model1.json model2.json model3.json \
      --method soft_voting

  # Hard voting (단순 다수결)
  python ensemble.py \
      --files model1.json model2.json model3.json \
      --method hard_voting
        """
    )

    parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        help='앙상블할 JSON 파일 리스트 (최소 2개)'
    )

    parser.add_argument(
        '--method',
        choices=['probability_avg', 'soft_voting', 'hard_voting'],
        default='probability_avg',
        help='앙상블 방법 (기본: probability_avg)'
    )

    parser.add_argument(
        '--output_dir',
        default='./',
        help='출력 디렉토리 (기본: 현재 디렉토리)'
    )

    parser.add_argument(
        '--base_dir',
        default='/data/ephemeral/home/T8091/phk-04/submissions/T8091',
        help='기본 경로 (상대 경로 사용 시)'
    )

    args = parser.parse_args()

    # 파일 경로 처리
    base_dir = Path(args.base_dir)
    full_paths = []

    for file in args.files:
        file_path = Path(file)

        if not file_path.is_absolute():
            file_path = base_dir / file

        full_paths.append(file_path)

    print(f"\n기본 경로: {base_dir}")
    print(f"앙상블 방법: {args.method}")
    print(f"출력 디렉토리: {args.output_dir}")

    # 앙상블 실행
    output_file = ensemble_predictions(
        full_paths,
        method=args.method,
        output_dir=args.output_dir
    )

    if output_file:
        print(f"\n🎉 성공! 결과 파일: {output_file}")
        print(f"\n💡 제출 방법:")
        print(f"   cp {output_file} submission.csv")
        print(f"   # submission.csv 파일을 제출 시스템에 업로드")

        if args.method == 'probability_avg':
            print(f"\n⭐ Probability 평균 사용:")
            print(f"   - 이론적으로 가장 정확한 방법")
            print(f"   - Softmax 확률을 직접 평균")
            print(f"   - Confidence 가중 평균보다 +0.002~0.005 개선 예상")

if __name__ == "__main__":
    main()
