export LLAMA_URL="http://127.0.0.1:8080"
# 필요시 경로 맞추기
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/submission_llamacpp.csv"
export DUMP_PROBS=1
export DEBUG_LOG=0
export PROBS_OUT_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/submission_llamacpp_probs.csv"
python3 inference_llamacpp_submit.py