"""
Unit Tests for the Translation Data Pipeline
--------------------------------------------------
Tests prompt formatting and dataset tokenization/collation -- the parts of
the translation pipeline that don't require GPU or gated Aya-23/Llama-3.1
access. End-to-end generation (`translate_with_aya`/`translate_with_llama`)
is exercised on the A100 training/inference environment instead, not in
unit tests.
"""

import unittest

from transformers import AutoTokenizer

from src.datasets.builder import InstructionDatasetBuilder
from src.datasets.collator import CausalLMDataCollator
from src.task_generation.prompt_templates import available_directions, format_completion_prompt


class TestPromptFormatting(unittest.TestCase):
    """Test translation prompt template formatting."""

    def test_format_completion_prompt_contains_source_text(self) -> None:
        """The formatted prompt must embed the exact source sentence."""
        prompt = format_completion_prompt("English", "Ekegusii", "Wash your hands.")
        self.assertIn("Wash your hands.", prompt)

    def test_unsupported_direction_raises(self) -> None:
        """A direction with no configured template must raise KeyError, not
        silently produce a malformed prompt."""
        with self.assertRaises(KeyError):
            format_completion_prompt("French", "Klingon", "Bonjour.")

    def test_available_directions_nonempty(self) -> None:
        """At least the six core directions must be discoverable from config."""
        directions = available_directions()
        self.assertGreaterEqual(len(directions), 4)


class TestTranslationDatasetPipeline(unittest.TestCase):
    """Test tokenization + collation of translation instruction pairs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load a small stand-in tokenizer once for the whole test class."""
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.cls_token
        cls.tokenizer.padding_side = "left"

    def test_builder_masks_prompt_labels(self) -> None:
        """Prompt-token labels must be masked with -100 so loss is computed
        only on the response portion."""
        import pandas as pd

        df = pd.DataFrame({"prompt": ["Translate: hello"], "response": ["mgusii-hello"]})
        builder = InstructionDatasetBuilder(self.tokenizer, max_length=64)
        dataset = builder.build(df)

        labels = dataset[0]["labels"]
        self.assertIn(-100, labels)
        self.assertTrue(any(label != -100 for label in labels))

    def test_builder_respects_max_length(self) -> None:
        """No tokenized example may exceed the configured max_length, even
        when the raw prompt+response would otherwise be longer."""
        import pandas as pd

        long_text = " ".join(["word"] * 200)
        df = pd.DataFrame({"prompt": [long_text], "response": [long_text]})
        builder = InstructionDatasetBuilder(self.tokenizer, max_length=32)
        dataset = builder.build(df)

        self.assertLessEqual(len(dataset[0]["input_ids"]), 32)

    def test_collator_pads_batch_to_same_length(self) -> None:
        """All examples in a collated batch must share the same sequence length."""
        import pandas as pd

        df = pd.DataFrame({"prompt": ["short", "a much longer prompt here"], "response": ["ok", "a longer response too"]})
        builder = InstructionDatasetBuilder(self.tokenizer, max_length=64)
        dataset = builder.build(df)

        collator = CausalLMDataCollator(tokenizer=self.tokenizer)
        batch = collator([dataset[0], dataset[1]])

        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["input_ids"].shape[1], batch["labels"].shape[1])


if __name__ == "__main__":
    unittest.main()
