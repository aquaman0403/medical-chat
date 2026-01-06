from core.state import AgentState
import re
from unidecode import unidecode


# Keyword theo nhóm
SYMPTOM_KEYWORDS = {
    # English
    "fever", "pain", "headache", "nausea", "vomiting", "diarrhea", "cough",
    "shortness of breath", "chest pain", "dizziness", "fatigue", "seizure",
    "bleeding", "weight loss", "insomnia",

    # Vietnamese (không dấu)
    "sot", "dau", "dau dau", "buon non", "non", "tieu chay", "ho",
    "kho tho", "dau nguc", "chong mat", "met moi", "co giat",
    "chay mau", "sut can", "mat ngu",
}

DISEASE_KEYWORDS = {
    # English
    "cancer", "diabetes", "hypertension", "heart disease", "stroke",
    "asthma", "copd", "pneumonia", "covid", "epilepsy",

    # Vietnamese
    "ung thu", "tieu duong", "cao huyet ap", "benh tim", "dot quy",
    "hen suyen", "viem phoi", "dong kinh",
}

MEDICAL_TERM_KEYWORDS = {
    # English
    "diagnosis", "treatment", "medication", "surgery", "prescription",
    "side effects", "vaccine", "rehabilitation",

    # Vietnamese
    "chan doan", "dieu tri", "thuoc", "phau thuat",
    "tac dung phu", "vac xin", "phuc hoi",
}

BODY_PART_KEYWORDS = {
    # English
    "heart", "lung", "brain", "liver", "kidney",

    # Vietnamese
    "tim", "phoi", "nao", "gan", "than",
}

GENERIC_MEDICAL_KEYWORDS = {
    "benh", "suc khoe", "y te", "kham", "bac si", "benh vien"
}


# Trọng số
WEIGHTS = {
    "disease": 3,
    "symptom": 2,
    "medical_term": 2,
    "body_part": 1,
    "generic": 0.5,
}


# Ngưỡng để coi là câu hỏi y tế
MEDICAL_SCORE_THRESHOLD = 2.5


# PlannerAgent
def PlannerAgent(state: AgentState) -> AgentState:
    question = state.get("question", "").lower()
    question_normalized = unidecode(question)

    tokens = set(re.findall(r"\w+", question_normalized))

    score = 0.0

    # Match phrase (ưu tiên trước)
    def match_phrase(phrases, weight):
        nonlocal score
        for p in phrases:
            if " " in p and p in question_normalized:
                score += weight

    match_phrase(SYMPTOM_KEYWORDS, WEIGHTS["symptom"])
    match_phrase(DISEASE_KEYWORDS, WEIGHTS["disease"])
    match_phrase(MEDICAL_TERM_KEYWORDS, WEIGHTS["medical_term"])

    # Match token đơn
    for token in tokens:
        if token in DISEASE_KEYWORDS:
            score += WEIGHTS["disease"]
        elif token in SYMPTOM_KEYWORDS:
            score += WEIGHTS["symptom"]
        elif token in MEDICAL_TERM_KEYWORDS:
            score += WEIGHTS["medical_term"]
        elif token in BODY_PART_KEYWORDS:
            score += WEIGHTS["body_part"]
        elif token in GENERIC_MEDICAL_KEYWORDS:
            score += WEIGHTS["generic"]

    # Quyết định routing
    state["current_tool"] = (
        "retriever" if score >= MEDICAL_SCORE_THRESHOLD else "llm_agent"
    )

    # Không reset retry_count nếu đã tồn tại
    state.setdefault("retry_count", 0)

    return state
