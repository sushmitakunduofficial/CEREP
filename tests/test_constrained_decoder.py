"""
Phase 7 — Test suite for the constraint engine and KG-Trie structures.

Validates:
  - KGTrie character-level entity trie (insert, contains)
  - TokenKGTrie token-level prefix trie (insert_token_sequence, get_valid_tokens, is_complete)
  - TrieCompiler (paths_to_strings, compile)
  - apply_logit_mask with list-based logits
  - ConstraintEngine integration (compile_token_trie)
"""
import pytest
import networkx as nx

from backend.reasoning.constraint_engine import (
    KGTrie,
    TokenKGTrie,
    TrieCompiler,
    ConstraintEngine,
    apply_logit_mask,
)


# ═══════════════════════════════════════════════════════════════════════════════
# KGTrie — Character-level entity trie
# ═══════════════════════════════════════════════════════════════════════════════


class TestKGTrie:
    """Test the character-level entity trie for O(L) lookup."""

    def test_insert_and_contains(self):
        trie = KGTrie()
        trie.insert("TP53")
        assert trie.contains("TP53")

    def test_contains_is_case_insensitive(self):
        trie = KGTrie()
        trie.insert("BRCA1")
        assert trie.contains("brca1")
        assert trie.contains("Brca1")
        assert trie.contains("BRCA1")

    def test_contains_returns_false_for_missing(self):
        trie = KGTrie()
        trie.insert("TP53")
        assert not trie.contains("TP54")
        assert not trie.contains("NONEXISTENT")

    def test_contains_prefix_is_not_complete(self):
        """A prefix of an inserted string should NOT match (not terminal)."""
        trie = KGTrie()
        trie.insert("BRCA1")
        assert not trie.contains("BRC")

    def test_all_entities(self):
        trie = KGTrie()
        trie.insert("TP53")
        trie.insert("BRCA1")
        trie.insert("MDM2")
        entities = trie.all_entities()
        assert len(entities) == 3
        assert "TP53" in entities
        assert "BRCA1" in entities
        assert "MDM2" in entities

    def test_size_bytes_positive(self):
        trie = KGTrie()
        trie.insert("GENE_A")
        assert trie.size_bytes() > 0

    def test_insert_duplicate(self):
        """Inserting the same entity twice should not duplicate in all_entities."""
        trie = KGTrie()
        trie.insert("TP53")
        trie.insert("TP53")
        assert len(trie.all_entities()) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TokenKGTrie — Token-level prefix trie for constrained decoding
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenKGTrie:
    """Test the token-level KG-Trie used during autoregressive decoding."""

    def test_insert_token_sequence_and_path_count(self):
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20, 30], path_index=0)
        assert trie.path_count == 1

    def test_insert_multiple_sequences(self):
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20, 30])
        trie.insert_token_sequence([10, 20, 40])
        trie.insert_token_sequence([50, 60])
        assert trie.path_count == 3

    def test_get_valid_tokens_at_root(self):
        """At the root, valid tokens are the first tokens of all inserted sequences."""
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20])
        trie.insert_token_sequence([30, 40])
        valid = trie.get_valid_tokens([])
        assert valid == {10, 30}

    def test_get_valid_tokens_after_prefix(self):
        """After prefix [10], only valid continuations should be returned."""
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20, 30])
        trie.insert_token_sequence([10, 25, 35])
        valid = trie.get_valid_tokens([10])
        assert valid == {20, 25}

    def test_get_valid_tokens_at_leaf(self):
        """At a leaf node (terminal), there are no further valid tokens."""
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20])
        valid = trie.get_valid_tokens([10, 20])
        assert valid == set()

    def test_get_valid_tokens_diverged_prefix(self):
        """A prefix that diverges from all paths returns an empty set."""
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20])
        valid = trie.get_valid_tokens([99])
        assert valid == set()

    def test_is_complete_true(self):
        """A prefix that matches a complete path should be flagged as complete."""
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20, 30])
        assert trie.is_complete([10, 20, 30]) is True

    def test_is_complete_false_partial(self):
        """A partial prefix should not be considered complete."""
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20, 30])
        assert trie.is_complete([10, 20]) is False

    def test_is_complete_false_diverged(self):
        """A diverged prefix should not be considered complete."""
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20])
        assert trie.is_complete([99]) is False

    def test_node_count(self):
        """Node count should reflect shared prefixes."""
        trie = TokenKGTrie()
        trie.insert_token_sequence([10, 20, 30])
        trie.insert_token_sequence([10, 20, 40])
        # root → 10 → 20 → 30, and 20 → 40: 5 nodes total
        assert trie.node_count == 5

    def test_estimated_memory_mb(self):
        trie = TokenKGTrie()
        trie.insert_token_sequence([1, 2, 3])
        assert trie.estimated_memory_mb() >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# TrieCompiler
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrieCompiler:
    """Test TrieCompiler path conversion and trie compilation."""

    def test_paths_to_strings_simple(self):
        """Convert path dicts with nodes and edges to tokenizable strings."""
        paths = [
            {
                "nodes": ["TP53", "MDM2", "TP53"],
                "edges": [
                    {"predicate": "biolink:positively_regulates"},
                    {"predicate": "biolink:negatively_regulates"},
                ],
            }
        ]
        result = TrieCompiler.paths_to_strings(paths)
        assert len(result) == 1
        assert "TP53" in result[0]
        assert "MDM2" in result[0]
        # Predicates should have "biolink:" stripped
        assert "positively_regulates" in result[0]
        assert "biolink:" not in result[0]

    def test_paths_to_strings_uses_edge_type_fallback(self):
        """When 'predicate' is missing, fall back to 'edge_type'."""
        paths = [
            {
                "nodes": ["A", "B"],
                "edges": [{"edge_type": "biolink:targets"}],
            }
        ]
        result = TrieCompiler.paths_to_strings(paths)
        assert "targets" in result[0]

    def test_paths_to_strings_empty(self):
        assert TrieCompiler.paths_to_strings([]) == []

    def test_compile_produces_valid_trie(self):
        """TrieCompiler.compile() should produce a TokenKGTrie with correct path count."""
        compiler = TrieCompiler(tokenizer=None)  # uses hash-based fallback
        path_strings = [
            "TP53 positively_regulates MDM2",
            "BRCA1 participates_in DNA_REPAIR",
        ]
        trie = compiler.compile(path_strings)
        assert isinstance(trie, TokenKGTrie)
        assert trie.path_count == 2

    def test_compile_trie_with_shared_prefix(self):
        """Paths with a shared prefix should share trie nodes."""
        compiler = TrieCompiler(tokenizer=None)
        path_strings = [
            "TP53 regulates MDM2",
            "TP53 regulates BAX",
        ]
        trie = compiler.compile(path_strings)
        assert trie.path_count == 2
        # "TP53" and "regulates" are shared → fewer total nodes than 6
        assert trie.node_count < 7  # at most: root + TP53 + regulates + MDM2 + BAX = 5

    def test_compile_empty_list(self):
        compiler = TrieCompiler(tokenizer=None)
        trie = compiler.compile([])
        assert trie.path_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# apply_logit_mask — Pure Python (list) fallback
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyLogitMask:
    """Test the logit masking function with list-based inputs."""

    def test_apply_logit_mask_basic(self):
        """Valid tokens keep their logit value; invalid tokens get -inf."""
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        valid_ids = {1, 3}
        vocab_size = 5
        masked = apply_logit_mask(logits, valid_ids, vocab_size)
        assert masked[1] == 2.0  # valid
        assert masked[3] == 4.0  # valid
        assert masked[0] == -1e9  # invalid
        assert masked[2] == -1e9  # invalid
        assert masked[4] == -1e9  # invalid

    def test_apply_logit_mask_all_valid(self):
        """When all tokens are valid, no masking should occur."""
        logits = [1.0, 2.0, 3.0]
        valid_ids = {0, 1, 2}
        masked = apply_logit_mask(logits, valid_ids, 3)
        assert masked == logits

    def test_apply_logit_mask_none_valid(self):
        """When no tokens are valid, all should be masked."""
        logits = [1.0, 2.0, 3.0]
        masked = apply_logit_mask(logits, set(), 3)
        assert all(v == -1e9 for v in masked)

    def test_apply_logit_mask_custom_neg_inf(self):
        """Custom neg_inf value should be used for masking."""
        logits = [10.0, 20.0]
        masked = apply_logit_mask(logits, {0}, 2, neg_inf=-999.0)
        assert masked[0] == 10.0
        assert masked[1] == -999.0

    def test_apply_logit_mask_preserves_length(self):
        logits = [0.1] * 100
        masked = apply_logit_mask(logits, {50}, 100)
        assert len(masked) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# ConstraintEngine — Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstraintEngine:
    """Test ConstraintEngine initialization and trie compilation."""

    @pytest.fixture
    def small_graph(self) -> nx.DiGraph:
        """Build a minimal DiGraph for the ConstraintEngine."""
        g = nx.DiGraph()
        g.add_node("TP53", aliases=["p53"], category="biolink:Gene")
        g.add_node("MDM2", aliases=[], category="biolink:Gene")
        g.add_node("BRCA1", aliases=["BRCA-1"], category="biolink:Gene")
        g.add_edge("TP53", "MDM2", predicate="biolink:positively_regulates", weight=0.9)
        g.add_edge("MDM2", "TP53", predicate="biolink:negatively_regulates", weight=0.9)
        g.add_edge("BRCA1", "TP53", predicate="biolink:physically_interacts_with", weight=0.7)
        return g

    def test_engine_builds_entity_trie(self, small_graph: nx.DiGraph):
        engine = ConstraintEngine(small_graph)
        assert engine.is_valid_entity("TP53")
        assert engine.is_valid_entity("MDM2")
        assert engine.is_valid_entity("BRCA1")

    def test_engine_validates_aliases(self, small_graph: nx.DiGraph):
        engine = ConstraintEngine(small_graph)
        assert engine.is_valid_entity("p53")
        assert engine.is_valid_entity("BRCA-1")

    def test_engine_validates_edges(self, small_graph: nx.DiGraph):
        engine = ConstraintEngine(small_graph)
        assert engine.is_valid_edge("TP53", "MDM2")
        assert not engine.is_valid_edge("MDM2", "BRCA1")

    def test_engine_validate_path_sequence(self, small_graph: nx.DiGraph):
        engine = ConstraintEngine(small_graph)
        result = engine.validate_path_sequence(["TP53", "MDM2"])
        assert result["valid"] is True

        result = engine.validate_path_sequence(["TP53", "NONEXISTENT"])
        assert result["valid"] is False
        assert len(result["issues"]) > 0

    def test_compile_token_trie(self, small_graph: nx.DiGraph):
        """compile_token_trie should produce a TokenKGTrie from path dicts."""
        engine = ConstraintEngine(small_graph)
        paths = [
            {
                "nodes": ["TP53", "MDM2"],
                "edges": [{"predicate": "biolink:positively_regulates"}],
            },
            {
                "nodes": ["BRCA1", "TP53"],
                "edges": [{"predicate": "biolink:physically_interacts_with"}],
            },
        ]
        trie = engine.compile_token_trie(paths)
        assert isinstance(trie, TokenKGTrie)
        assert trie.path_count == 2

    def test_constrain_entity_list(self, small_graph: nx.DiGraph):
        engine = ConstraintEngine(small_graph)
        result = engine.constrain_entity_list(["TP53", "FAKE_GENE", "MDM2"])
        assert "TP53" in result["valid"]
        assert "MDM2" in result["valid"]
        assert "FAKE_GENE" in result["invalid"]

    def test_extract_entities_from_text(self, small_graph: nx.DiGraph):
        engine = ConstraintEngine(small_graph)
        text = "The TP53 gene interacts with MDM2 to regulate apoptosis."
        found = engine.extract_entities_from_text(text)
        assert "TP53" in found
        assert "MDM2" in found

    def test_build_prompt_context(self, small_graph: nx.DiGraph):
        engine = ConstraintEngine(small_graph)
        paths = [
            {
                "nodes": ["TP53", "MDM2"],
                "readable": "TP53 --[positively_regulates]--> MDM2",
            }
        ]
        ctx = engine.build_prompt_context(paths)
        assert "KNOWLEDGE GRAPH CONSTRAINTS" in ctx
        assert "TP53" in ctx
