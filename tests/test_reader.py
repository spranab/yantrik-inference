"""Tests that do not need a model.

The parts worth testing without weights are the ones that silently produce wrong
answers rather than errors: question parsing, and the benchmark's own baseline.
"""
from yantrik_inference.reader import Field
from yantrik_inference.tasks import majority_baseline, make_case


def test_field_parse_boolean_default():
    f = Field.parse("Was the card present?")
    assert f.options == ("yes", "no")
    assert f.question == "Was the card present?"


def test_field_parse_enum():
    f = Field.parse("Which region? | domestic/offshore/regional")
    assert f.options == ("domestic", "offshore", "regional")
    assert f.question == "Which region?"


def test_field_parse_rejects_single_option():
    assert Field.parse("Bad question | onlyone") is None
    assert Field.parse("   ") is None


def test_field_parse_keeps_pipes_in_question():
    f = Field.parse("Is a | b the separator? | yes/no")
    assert f.question == "Is a | b the separator?"
    assert f.options == ("yes", "no")


def test_task_is_not_trivially_solvable():
    """The generator draws balanced facts, so always answering the most common
    value should be well short of perfect. If this ever passes near 1.0 the
    benchmark has stopped measuring anything."""
    import random
    cases = [make_case(random.Random(i)) for i in range(60)]
    base = majority_baseline(cases)
    assert 0.45 < base < 0.75, base


def test_every_question_has_a_valid_answer():
    import random
    for i in range(20):
        _, qs = make_case(random.Random(i))
        for q, opts, gold in qs:
            assert gold in opts, (q, gold, opts)
            assert len(opts) >= 2


def test_missing_path_is_an_error_not_a_record():
    """A typo'd path must not be silently used as the record text: the model
    would then answer questions about the filename, confidently."""
    import pytest
    from yantrik_inference.cli import _text_or_file
    from yantrik_inference.engine import EngineError
    with pytest.raises(EngineError):
        _text_or_file("ticket.txt", "record")
    with pytest.raises(EngineError):
        _text_or_file("data/claims/x.md", "record")
    # real text still passes through
    assert _text_or_file("Ticket 4471: the user cannot log in.", "record").startswith("Ticket")
