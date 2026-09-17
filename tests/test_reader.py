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


class _FakeReader:
    """check_options and first_tokens only need a tokenizer, so fake one:
    a token id per distinct string prefix."""
    from yantrik_inference.reader import FieldReader
    check_options = FieldReader.check_options
    first_tokens = FieldReader.first_tokens
    first_token = FieldReader.first_token

    def __init__(self):
        self._first = {}
        self._ids = {}

    def tok(self, s, bos=False):
        # a crude BPE stand-in: the first "token" is the leading 3 characters
        key = s[:3]
        return [self._ids.setdefault(key, len(self._ids) + 1)]


def test_first_tokens_covers_spacing_and_case():
    """A template that ends with a newline makes the model write 'no'; one that
    ends mid-line makes it write ' no'. Scoring one variant reads the wrong
    token: on Llama-3.2-3B that produced 99%-confident wrong answers."""
    r = _FakeReader()
    ids = r.first_tokens("yes")
    assert len(ids) >= 3            # 'yes', ' yes', 'Yes' at least
    assert len(set(ids)) == len(ids)


def test_colliding_options_are_rejected():
    from yantrik_inference.reader import Field
    r = _FakeReader()
    ok = r.check_options([Field("q", ("yes", "no"))])
    assert ok == []
    bad = r.check_options([Field("q", ("approve", "approved"))])
    assert bad and "same token" in bad[0]
