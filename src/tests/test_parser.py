from prompting import CONTINUE, PRUNE, parse_decision


def test_strict_format():
    assert parse_decision("DECISION: CONTINUE") == CONTINUE
    assert parse_decision("DECISION: PRUNE") == PRUNE


def test_case_and_spacing():
    assert parse_decision("decision : prune") == PRUNE
    assert parse_decision("Final answer.\nDECISION:CONTINUE") == CONTINUE


def test_last_occurrence_wins():
    text = "Format is DECISION: CONTINUE or DECISION: PRUNE.\nDECISION: PRUNE"
    assert parse_decision(text) == PRUNE


def test_bare_keyword_fallback():
    assert parse_decision("I think we should prune this run.") == PRUNE
    assert parse_decision("Let's continue training.") == CONTINUE


def test_safe_default_on_garbage():
    assert parse_decision("") == CONTINUE
    assert parse_decision("no idea") == CONTINUE
    assert parse_decision("contains both continue and prune words") == CONTINUE
    assert parse_decision("garbage", default=PRUNE) == PRUNE
