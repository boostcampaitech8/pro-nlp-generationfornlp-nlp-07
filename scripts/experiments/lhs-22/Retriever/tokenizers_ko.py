# tokenizers_ko.py
import re

class KoTokenizer:
    def __init__(self, mode="kiwi"):
        self.mode = mode
        self._kiwi = None
        if mode == "kiwi":
            try:
                from kiwipiepy import Kiwi
                self._kiwi = Kiwi()
            except Exception:
                self.mode = "fallback"

    def __call__(self, text: str):
        text = (text or "").strip()
        if not text:
            return []

        if self.mode == "kiwi" and self._kiwi is not None:
            toks = []
            for t in self._kiwi.tokenize(text):
                if t.tag in ("NNG", "NNP", "VV", "VA", "XR"):
                    toks.append(t.form)
            return toks

        words = re.findall(r"[가-힣A-Za-z0-9]+", text)
        bigrams = []
        for w in words:
            if len(w) >= 4:
                bigrams.extend([w[i:i+2] for i in range(len(w)-1)])
        return words + bigrams
