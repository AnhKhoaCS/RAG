import re

def extract_answer(text_respone: str,
                    pattent: str = r"Answer:\s*(.*)"
                    ) -> str:
    match = re.search(pattent, text_respone)
    if match:
        answer_text = match.group(1).strip()
        return answer_text
    else:
        return "Answer not found."