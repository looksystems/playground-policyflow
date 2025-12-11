"""Unit tests for TransformNode."""

import pytest

from policyflow.nodes.transform import TransformNode


class TestTransformNodeBasicTransforms:
    """Tests for basic transformation operations."""

    def test_lowercase(self):
        """lowercase transform should convert to lowercase."""
        node = TransformNode(transforms=["lowercase"])
        shared = {"input_text": "Hello WORLD"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "default"
        assert shared["input_text"] == "hello world"

    def test_uppercase(self):
        """uppercase transform should convert to uppercase."""
        node = TransformNode(transforms=["uppercase"])
        shared = {"input_text": "Hello World"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "HELLO WORLD"

    def test_strip_html(self):
        """strip_html transform should remove HTML tags."""
        node = TransformNode(transforms=["strip_html"])
        shared = {"input_text": "<p>This is <b>bold</b> text</p>"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "This is bold text"

    def test_normalize_whitespace(self):
        """normalize_whitespace should collapse multiple spaces."""
        node = TransformNode(transforms=["normalize_whitespace"])
        shared = {"input_text": "Too   many    spaces"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Too many spaces"

    def test_normalize_whitespace_newlines(self):
        """normalize_whitespace should collapse newlines to single space."""
        node = TransformNode(transforms=["normalize_whitespace"])
        shared = {"input_text": "Line1\n\nLine2\n\n\nLine3"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Line1 Line2 Line3"

    def test_strip_urls(self):
        """strip_urls should remove http and https URLs."""
        node = TransformNode(transforms=["strip_urls"])
        shared = {"input_text": "Visit https://example.com or http://test.org for info"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Visit  or  for info"

    def test_strip_emails(self):
        """strip_emails should remove email addresses."""
        node = TransformNode(transforms=["strip_emails"])
        shared = {"input_text": "Contact support@example.com or admin@test.org"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Contact  or "

    def test_truncate(self):
        """truncate:N should limit text to N characters."""
        node = TransformNode(transforms=["truncate:10"])
        shared = {"input_text": "Hello World, this is a long message"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Hello Worl"
        assert len(shared["input_text"]) == 10

    def test_trim(self):
        """trim should remove leading/trailing whitespace."""
        node = TransformNode(transforms=["trim"])
        shared = {"input_text": "   Hello World   "}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Hello World"


class TestTransformNodeChainedTransforms:
    """Tests for multiple transforms applied in sequence."""

    def test_chained_transforms(self):
        """Multiple transforms should be applied in order."""
        node = TransformNode(
            transforms=["strip_html", "lowercase", "normalize_whitespace", "trim"]
        )
        shared = {"input_text": "  <p>Hello   WORLD</p>  "}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "hello world"

    def test_transform_order_matters(self):
        """Transform order should affect the result."""
        # truncate then lowercase
        node1 = TransformNode(transforms=["truncate:5", "lowercase"])
        shared1 = {"input_text": "HELLO WORLD"}

        prep_res = node1.prep(shared1)
        exec_res = node1.exec(prep_res)
        node1.post(shared1, prep_res, exec_res)

        assert shared1["input_text"] == "hello"  # truncated to "HELLO", then lowercased

        # lowercase then truncate
        node2 = TransformNode(transforms=["lowercase", "truncate:5"])
        shared2 = {"input_text": "HELLO WORLD"}

        prep_res = node2.prep(shared2)
        exec_res = node2.exec(prep_res)
        node2.post(shared2, prep_res, exec_res)

        assert shared2["input_text"] == "hello"  # lowercased, then truncated


class TestTransformNodeCustomKeys:
    """Tests for custom input/output key configuration."""

    def test_custom_input_key(self):
        """Custom input_key should read from specified key."""
        node = TransformNode(
            transforms=["lowercase"],
            input_key="raw_text",
        )
        shared = {"raw_text": "HELLO", "input_text": "WORLD"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "hello"
        assert shared["raw_text"] == "HELLO"  # unchanged

    def test_custom_output_key(self):
        """Custom output_key should write to specified key."""
        node = TransformNode(
            transforms=["lowercase"],
            output_key="processed_text",
        )
        shared = {"input_text": "HELLO"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["processed_text"] == "hello"
        assert shared["input_text"] == "HELLO"  # unchanged

    def test_custom_input_and_output_keys(self):
        """Both custom keys should work together."""
        node = TransformNode(
            transforms=["uppercase"],
            input_key="source",
            output_key="destination",
        )
        shared = {"source": "hello"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["destination"] == "HELLO"
        assert shared["source"] == "hello"


class TestTransformNodeEdgeCases:
    """Edge case tests for TransformNode."""

    def test_unknown_transform(self):
        """Unknown transform should be ignored (text unchanged)."""
        node = TransformNode(transforms=["unknown_transform"])
        shared = {"input_text": "Hello World"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Hello World"

    def test_empty_transforms_list(self):
        """Empty transforms list should leave text unchanged."""
        node = TransformNode(transforms=[])
        shared = {"input_text": "Hello World"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Hello World"

    def test_empty_input(self):
        """Empty input should remain empty."""
        node = TransformNode(transforms=["lowercase", "trim"])
        shared = {"input_text": ""}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == ""

    def test_missing_input_key(self):
        """Missing input_key should default to empty string."""
        node = TransformNode(transforms=["lowercase"])
        shared = {}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == ""

    def test_truncate_without_param(self):
        """truncate without parameter should leave text unchanged."""
        node = TransformNode(transforms=["truncate"])
        shared = {"input_text": "Hello World"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Hello World"

    def test_truncate_longer_than_text(self):
        """truncate with length longer than text should not change it."""
        node = TransformNode(transforms=["truncate:100"])
        shared = {"input_text": "Short"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Short"

    def test_returns_default_action(self):
        """TransformNode should always return 'default' action."""
        node = TransformNode(transforms=["lowercase"])
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "default"

    def test_strip_html_nested_tags(self):
        """strip_html should handle nested HTML tags."""
        node = TransformNode(transforms=["strip_html"])
        shared = {"input_text": "<div><p>Nested <span>tags</span></p></div>"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Nested tags"

    def test_strip_html_self_closing_tags(self):
        """strip_html should handle self-closing tags."""
        node = TransformNode(transforms=["strip_html"])
        shared = {"input_text": "Line1<br/>Line2<hr/>End"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Line1Line2End"

    def test_unicode_text(self):
        """Transforms should handle unicode correctly."""
        node = TransformNode(transforms=["lowercase"])
        shared = {"input_text": "HELLO 世界 ПРИВЕТ"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "hello 世界 привет"

    def test_truncate_unicode(self):
        """truncate should count unicode characters correctly."""
        node = TransformNode(transforms=["truncate:5"])
        shared = {"input_text": "Hello世界"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["input_text"] == "Hello"
        assert len(shared["input_text"]) == 5
