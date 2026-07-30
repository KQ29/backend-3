from __future__ import annotations

from app.models.domain import ProbeStrategy, QuestionId, ResponseType

QUESTION_TEXT: dict[QuestionId, str] = {
    QuestionId.CONSENT: (
        "Karibu. Before we start: this interview will be recorded and transcribed, "
        "your responses will be anonymized in reporting, and you can stop at any "
        "point by saying 'stop.' Is it OK to continue?"
    ),
    QuestionId.DEMO_NAME: "Before the interview, could you share your name?",
    QuestionId.DEMO_EMAIL: "What is your email address?",
    QuestionId.DEMO_AGE: "What is your age or date of birth?",
    QuestionId.DEMO_GENDER: "How do you describe your gender?",
    QuestionId.DEMO_COUNTY: "Which county are you based in?",
    QuestionId.DEMO_SUB_COUNTY: "Which sub-county are you based in?",
    QuestionId.DEMO_OCCUPATION: "What is your current occupation or employment status?",
    QuestionId.ANCHOR_1: (
        "Since the training, has anything changed in your income, job role, "
        "responsibilities, or opportunities - even something small? Tell me about it."
    ),
    QuestionId.ANCHOR_1_PROBE: (
        "Was that a specific, datable thing - like a raise, promotion, or new client - "
        "or more of a general sense things have improved?"
    ),
    QuestionId.ANCHOR_2: (
        "What, if anything, is currently getting in the way of using what you "
        "learned for work or income - opportunities, employer support, confidence, "
        "access to tools, or something else?"
    ),
    QuestionId.ANCHOR_2_PROBE: (
        "Can you describe one recent situation when that barrier got in the way?"
    ),
    QuestionId.ANCHOR_3: (
        "For what did change - was it more about doing your current job better, or "
        "about accessing something new: a role, a client, a side income?"
    ),
    QuestionId.ANCHOR_3_PROBE: (
        "What specifically did you do differently that led to that?"
    ),
    QuestionId.ANCHOR_4: (
        "Have you shared anything from the training with colleagues, or changed how "
        "your team works as a result? What happened?"
    ),
    QuestionId.ANCHOR_4_PROBE: (
        "What specifically did you share, and how did it land?"
    ),
    QuestionId.CATCH_ALL: (
        "What is one additional thing the AI literacy training could have provided "
        "to better support your income or opportunities?"
    ),
    QuestionId.WRAP_UP: (
        "Is there anything else you would like to add that we have not yet discussed?"
    ),
    QuestionId.CLOSE: (
        "Asante sana. Thanks - that's everything I needed. Your responses will be "
        "anonymized and used to shape future training."
    ),
}

WRAP_UP_ELABORATION_TEXT = "Please go ahead - what would you like to add?"


DEMOGRAPHIC_QUESTIONS: tuple[QuestionId, ...] = (
    QuestionId.DEMO_NAME,
    QuestionId.DEMO_EMAIL,
    QuestionId.DEMO_AGE,
    QuestionId.DEMO_GENDER,
    QuestionId.DEMO_COUNTY,
    QuestionId.DEMO_SUB_COUNTY,
    QuestionId.DEMO_OCCUPATION,
)


ANCHOR_QUESTIONS: tuple[QuestionId, ...] = (
    QuestionId.ANCHOR_1,
    QuestionId.ANCHOR_2,
    QuestionId.ANCHOR_3,
    QuestionId.ANCHOR_4,
)


PROBE_QUESTIONS: tuple[QuestionId, ...] = (
    QuestionId.ANCHOR_1_PROBE,
    QuestionId.ANCHOR_2_PROBE,
    QuestionId.ANCHOR_3_PROBE,
    QuestionId.ANCHOR_4_PROBE,
)

ALLOWED_PROBE_BY_ANCHOR: dict[QuestionId, QuestionId] = {
    QuestionId.ANCHOR_1: QuestionId.ANCHOR_1_PROBE,
    QuestionId.ANCHOR_2: QuestionId.ANCHOR_2_PROBE,
    QuestionId.ANCHOR_3: QuestionId.ANCHOR_3_PROBE,
    QuestionId.ANCHOR_4: QuestionId.ANCHOR_4_PROBE,
}

ANCHOR_BY_QUESTION: dict[QuestionId, QuestionId] = {
    QuestionId.ANCHOR_1: QuestionId.ANCHOR_1,
    QuestionId.ANCHOR_1_PROBE: QuestionId.ANCHOR_1,
    QuestionId.ANCHOR_2: QuestionId.ANCHOR_2,
    QuestionId.ANCHOR_2_PROBE: QuestionId.ANCHOR_2,
    QuestionId.ANCHOR_3: QuestionId.ANCHOR_3,
    QuestionId.ANCHOR_3_PROBE: QuestionId.ANCHOR_3,
    QuestionId.ANCHOR_4: QuestionId.ANCHOR_4,
    QuestionId.ANCHOR_4_PROBE: QuestionId.ANCHOR_4,
}

ANCHOR_RESEARCH_OBJECTIVE: dict[QuestionId, str] = {
    QuestionId.ANCHOR_1: (
        "Understand whether work, income, responsibilities, or opportunities changed "
        "and what the respondent means by the change."
    ),
    QuestionId.ANCHOR_2: (
        "Understand the respondent's current barrier and how it affects work or "
        "income."
    ),
    QuestionId.ANCHOR_3: (
        "Understand what changed in practice and why that outcome matters to the "
        "respondent."
    ),
    QuestionId.ANCHOR_4: (
        "Understand what the respondent shared or changed with others and what "
        "resulted."
    ),
}

FALLBACK_PROBES_BY_STRATEGY: dict[ProbeStrategy, tuple[str, str]] = {
    ProbeStrategy.CLARITY: (
        "Can you give one recent, specific example of what you mean?",
        "What was the result in that specific example?",
    ),
    ProbeStrategy.DRILL_DOWN: (
        "What did that outcome allow you to do in your work or life?",
        "Which consequence of that outcome mattered most in practice?",
    ),
    ProbeStrategy.TENSION: (
        "You described both a benefit and a difficulty. How do those two parts fit "
        "together for you?",
        "When those two sides pull in different directions, what happens in practice?",
    ),
}


DEMOGRAPHIC_FIELD_BY_QUESTION: dict[QuestionId, str] = {
    QuestionId.DEMO_NAME: "name",
    QuestionId.DEMO_EMAIL: "email",
    QuestionId.DEMO_AGE: "age_or_dob",
    QuestionId.DEMO_GENDER: "gender",
    QuestionId.DEMO_COUNTY: "county",
    QuestionId.DEMO_SUB_COUNTY: "sub_county",
    QuestionId.DEMO_OCCUPATION: "occupation",
}


def response_type_for(question_id: QuestionId | None) -> ResponseType:
    if question_id == QuestionId.CONSENT:
        return ResponseType.CONSENT_CHOICE
    if question_id in {QuestionId.CLOSE, None}:
        return ResponseType.COMPLETE
    return ResponseType.OPEN_TEXT


def interview_progress(question_id: QuestionId, status: str) -> int:
    if status in {"completed", "declined", "stopped", "abandoned"}:
        return 100
    order = (
        QuestionId.CONSENT,
        *DEMOGRAPHIC_QUESTIONS,
        QuestionId.ANCHOR_1,
        QuestionId.ANCHOR_2,
        QuestionId.ANCHOR_3,
        QuestionId.ANCHOR_4,
        QuestionId.CATCH_ALL,
        QuestionId.WRAP_UP,
    )
    base = {
        item: round(index / (len(order) - 1) * 95) for index, item in enumerate(order)
    }
    if question_id in PROBE_QUESTIONS:
        anchor = QuestionId(question_id.value.replace("_probe", ""))
        return min(94, base[anchor] + 3)
    return base.get(question_id, 0)
