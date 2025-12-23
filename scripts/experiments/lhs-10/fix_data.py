# fix_submission_from_tuple.py
import argparse
import ast
import pandas as pd


def fix_answer(ans: str) -> str:
    """
    - "1"~"5"면 그대로
    - "('3', 0.42)" 같은 튜플 문자열이면 첫 원소만 뽑기
    - 그 외는 fallback "1"
    """
    s = str(ans).strip()

    # 정상 케이스
    if s in {"1", "2", "3", "4", "5"}:
        return s

    # 튜플/리스트 문자열 파싱: "('3', 0.42)" -> '3'
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (tuple, list)) and len(obj) >= 1:
            label = str(obj[0]).strip().strip("'").strip('"')
            if label in {"1", "2", "3", "4", "5"}:
                return label
        # 혹시 숫자/문자 단일로 들어온 경우
        if isinstance(obj, (int, str)):
            label = str(obj).strip().strip("'").strip('"')
            if label in {"1", "2", "3", "4", "5"}:
                return label
    except Exception:
        pass

    # 최종 fallback
    return "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="원본 제출 CSV (id,answer)")
    ap.add_argument("--out_csv", required=True, help="수정된 제출 CSV 저장 경로")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "id" not in df.columns or "answer" not in df.columns:
        raise ValueError("입력 CSV는 반드시 'id'와 'answer' 컬럼이 있어야 합니다.")

    before_tuple_like = df["answer"].astype(str).str.strip().str.startswith("(").sum()

    df["answer"] = df["answer"].apply(fix_answer)

    # 정상화 검증: 1~5 외 값이 남아있으면 에러
    bad = ~df["answer"].isin(["1", "2", "3", "4", "5"])
    if bad.any():
        sample = df.loc[bad, ["id", "answer"]].head(10)
        raise ValueError(f"정규화 실패 값이 남아있습니다. 예시:\n{sample}")

    df.to_csv(args.out_csv, index=False)

    print(f"[DONE] wrote: {args.out_csv}")
    print(f"  tuple_like_rows_before: {int(before_tuple_like)}")
    print("  all answers are now in {1,2,3,4,5}")


if __name__ == "__main__":
    main()
