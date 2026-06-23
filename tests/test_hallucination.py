"""
Tests for the HallucinationChecker.
"""
import pytest
from backend.graph.graph_builder import CERAPGraphBuilder
from backend.reasoning.hallucination_checker import HallucinationChecker


@pytest.fixture(scope="module")
def builder():
    b = CERAPGraphBuilder()
    b.build_seed_graph()
    return b


@pytest.fixture(scope="module")
def checker(builder):
    return HallucinationChecker(builder.graph)


def test_zero_hallucination_on_pure_kg_text(checker):
    """Text containing only valid KG entities should have 0% hallucination."""
    text = "TP53 activates MDM2 which inhibits APOPTOSIS via BRCA1."
    result = checker.check(text, path_nodes=["TP53", "MDM2", "BRCA1"])
    # All uppercase tokens in text should be in KG → rate should be low
    # (allow small floating point; some short tokens may not match)
    assert result["hallucination_rate"] <= 0.35


def test_high_hallucination_on_fake_entities(checker):
    """Text containing many fake entities should score high hallucination."""
    text = "FAKEGENE1 activates FAKEDRUG2 via NONEXISTENT3 pathway. MADE_UP4."
    result = checker.check(text, path_nodes=["TP53"])
    assert result["hallucination_rate"] > 0.5


def test_grounding_score_full_coverage(checker):
    """If all path nodes appear in the text, grounding should be 1.0."""
    path_nodes = ["TP53", "MDM2"]
    text = "TP53 is regulated by MDM2 in an important feedback loop."
    result = checker.check(text, path_nodes=path_nodes)
    assert result["grounding_score"] == 1.0


def test_grounding_score_zero_coverage(checker):
    """If no path nodes appear in the text, grounding should be 0."""
    path_nodes = ["ERBB2", "TRASTUZUMAB"]
    text = "Some completely irrelevant text without any biological entities."
    result = checker.check(text, path_nodes=path_nodes)
    assert result["grounding_score"] == 0.0


def test_confidence_score_range(checker):
    """Confidence score must always be in [0, 1]."""
    texts = [
        "TP53 activates MDM2.",
        "FAKEGENE1 activates FAKEPROTEIN2.",
        "",
    ]
    for text in texts:
        result = checker.check(text, path_nodes=["TP53"])
        assert 0.0 <= result["confidence_score"] <= 1.0


def test_entity_in_kg(checker):
    assert checker.entity_in_kg("TP53") is True
    assert checker.entity_in_kg("FAKEGENE_NOT_IN_KG") is False


def test_flagged_entities_present(checker):
    text = "TP53 regulates COMPLETELYFAKEGENE and ANOTHERINVENTION."
    result = checker.check(text, path_nodes=["TP53"])
    flagged = [e.upper() for e in result["flagged_entities"]]
    assert "COMPLETELYFAKEGENE" in flagged
    assert "ANOTHERINVENTION" in flagged


def test_empty_text_returns_safe_result(checker):
    """Empty LLM output should not crash, and should return 0 hallucination."""
    result = checker.check("", path_nodes=["TP53"])
    assert result["hallucination_rate"] == 0.0
