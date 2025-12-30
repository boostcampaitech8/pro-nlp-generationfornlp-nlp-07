import pandas as pd
import matplotlib.pyplot as plt
import ast

TEST_PATH = "data/test/test.csv"

def main():
    df = pd.read_csv(TEST_PATH)
    print("[DATA] shape:", df.shape)
    print("[DATA] columns:", df.columns.tolist())

    # 1) paragraph 길이 (문자 수)
    df["para_len"] = df["paragraph"].fillna("").astype(str).str.len()

    # 2) question + choices까지 포함한 길이도 보자 (문자 수 기준)
    def get_qc_len(problems_str):
        try:
            problems = ast.literal_eval(problems_str)
        except Exception:
            return 0
        q = problems.get("question", "") or ""
        choices = problems.get("choices", []) or []
        choices_str = " ".join(map(str, choices))
        text = q + " " + choices_str
        return len(text)

    df["qc_len"] = df["problems"].apply(get_qc_len)
    df["total_len"] = df["para_len"] + df["qc_len"]

    # 3) 기본 통계
    print("\n[STATS] paragraph length (chars)")
    print(df["para_len"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\n[STATS] question+choices length (chars)")
    print(df["qc_len"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\n[STATS] total_len = paragraph + question+choices")
    print(df["total_len"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    # 4) 히스토그램 대충 보기
    plt.figure()
    df["para_len"].hist(bins=50)
    plt.title("Paragraph length (chars)")
    plt.xlabel("para_len")
    plt.ylabel("freq")
    plt.show()

    plt.figure()
    df["total_len"].hist(bins=50)
    plt.title("Total length = paragraph + question + choices (chars)")
    plt.xlabel("total_len")
    plt.ylabel("freq")
    plt.show()

    # 5) 길이 구간별 개수도 한 번
    bins = [0, 200, 400, 600, 800, 1000, 1500, 2000, 99999]
    labels = ["0-200", "200-400", "400-600", "600-800", "800-1000",
              "1000-1500", "1500-2000", "2000+"]

    df["para_bucket"] = pd.cut(df["para_len"], bins=bins, labels=labels, right=False)
    print("\n[BUCKET] paragraph length buckets (count)")
    print(df["para_bucket"].value_counts().sort_index())

if __name__ == "__main__":
    main()
