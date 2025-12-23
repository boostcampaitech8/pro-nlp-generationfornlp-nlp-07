from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# 1. 초기 설정
api = HfApi()
TOKEN = ""
REPO_ID = "NLP-07-ODQA/train_with_rag_label"  # 반드시 조직명/저장소명 형태여야 함

try:
    # 2. 저장소가 없는 경우 생성 (권한이 있다면 성공함)
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", token=TOKEN, private=True)
        print(f"✨ 저장소 '{REPO_ID}'가 새로 생성되었습니다.")
    except HfHubHTTPError as e:
        if e.response.status_code == 409:
            print(f"ℹ️ 저장소 '{REPO_ID}'가 이미 존재합니다.")
        else:
            raise e

    # 3. 파일 직접 업로드 (PR 없이 직접 푸시)
    print(f"🚀 {REPO_ID}에 직접 업로드를 시작합니다...")
    api.upload_file(
        path_or_fileobj="./data/train_with_rag_label.csv",
        path_in_repo="train_with_rag_label.csv",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=TOKEN,
        create_pr=False  # 직접 푸시를 위해 False 설정
    )
    print(f"✅ 업로드 성공: https://huggingface.co/datasets/{REPO_ID}")

except Exception as e:
    print(f"❌ 근본적 실패 사유: {e}")
    print("💡 팁: 여전히 403 에러가 난다면 조직 관리자에게 'Write' 권한을 요청하세요.")