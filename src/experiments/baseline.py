"""
E0: Zero-Shot Baseline Experiment
--------------------------------------
No fine-tuning: evaluates the raw pretrained Aya-23-8B / Llama-3.1-8B
models directly against the fixed master test split. Establishes the
floor every other experiment (E1-E8) must beat to justify its added
resources -- if E1 (bilingual fine-tuning) doesn't clearly outperform E0,
the resource/technique isn't earning its keep.
"""

import logging

from src.experiments.base import BaseExperiment
from src.models.aya.inference import translate_with_aya
from src.models.llama.inference import translate_with_llama

logger = logging.getLogger(__name__)


class BaselineExperiment(BaseExperiment):
    """E0: zero-shot evaluation of both base models, no adapter trained."""

    experiment_id = "E0_Baseline"

    def build_training_tasks(self):
        """E0 trains nothing -- returns an empty DataFrame.

        Raises:
            NotImplementedError: Always -- call `run()` instead, which
                skips training entirely for this experiment.
        """
        raise NotImplementedError("E0_Baseline does not train; call run() to evaluate zero-shot only.")

    def run(self, source_lang: str = "English", target_lang: str = "Ekegusii") -> dict:
        """Evaluate both base models zero-shot against the fixed test split.

        Args:
            source_lang: Source language for the test direction.
            target_lang: Target language for the test direction.

        Returns:
            dict: {"aya": {...metrics...}, "llama": {...metrics...}}.
        """
        test_pairs = self.build_test_pairs(source_lang, target_lang)
        sources = test_pairs["source"].tolist()
        references = test_pairs["target"].tolist()

        logger.info(f"[{self.experiment_id}] Evaluating {len(sources):,} zero-shot pairs for both models.")

        aya_predictions = translate_with_aya(sources, source_lang, target_lang, adapter_path=None)
        llama_predictions = translate_with_llama(sources, source_lang, target_lang, adapter_path=None)

        results = {
            "aya": self.evaluate_predictions(aya_predictions, references),
            "llama": self.evaluate_predictions(llama_predictions, references),
        }
        self.save_results(results)
        return results
