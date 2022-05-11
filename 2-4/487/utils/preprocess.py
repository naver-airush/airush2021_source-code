import re
import jamo

DEL_RULES = [
    re.compile(r'[^ .,?!/$%a-zA-Z0-9가-힣<>()\[\]]+'),
    # re.compile(r'[\.,!?]+'),
    # re.compile(r'^\('),
    # re.compile(r'\)$'),
    # re.compile(r'\[.*\]'),
]

SUB_RULES = [
    # (re.compile('구요'), '고요'),
    # (re.compile('려요'), '립니다'),
    # (re.compile('아요'), '습니다'),
    # (re.compile('세요'), '십시오'),
    # (re.compile('해요'), '합니다'),
    # (re.compile('(이에요|예요)'), '입니다'),
    # (re.compile('!'), '.'),
    # (re.compile('₩'), '원'),
    (re.compile(r'[\s]+'), ' '),
]

SPECIAL_RULE = (re.compile('어요'), '습니다')  # ㄹ어요 제외


def preprocess_noisy(sent):
    for rule in DEL_RULES:
        sent = rule.sub('', sent)

    for rule, subst in SUB_RULES:
        sent = rule.sub(subst, sent)

    # sent = special_case1(sent)

    # if re.search(r'[.?!]$', sent) is None:
    #     sent += '.'

    return sent


def special_case1(sent):
    """
    어요 -> 습니다
    `ㄹ어요`는 불규칙, 무시
    """
    rule, subst = SPECIAL_RULE
    match = rule.search(sent)
    if match is None:
        return sent
    start = match.span()[0]
    if start == 0:
        return sent
    prev = sent[start - 1]
    if jamo.j2hcj(jamo.h2j(prev))[-1] == 'ㄹ':
        return sent

    return rule.sub(subst, sent)
