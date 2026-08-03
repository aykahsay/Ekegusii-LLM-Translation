"""
Unit Tests for InstructionTaskGenerator using Standard unittest
----------------------------------------------------------------
"""

import unittest
import pandas as pd

from src.task_generation.instruction_generator import InstructionTaskGenerator


class TestInstructionGenerator(unittest.TestCase):
    """Test suite for 6-Way InstructionTaskGenerator."""

    def setUp(self) -> None:
        self.sample_df = pd.DataFrame([
            {
                "concept_id": 1,
                "English": "The government helps people.",
                "Kiswahili": "Serikali inasaidia watu.",
                "Ekegusii": "Eserikari yakonyere abanto.",
            }
        ])

    def test_task_expansion(self) -> None:
        """Test expansion of 1 trilingual concept into 6 bidirectional tasks."""
        generator = InstructionTaskGenerator()
        tasks_df = generator.generate_tasks_from_dataframe(self.sample_df)

        self.assertIsInstance(tasks_df, pd.DataFrame)
        self.assertEqual(len(tasks_df), 6)
        self.assertEqual(
            set(tasks_df["task_type"]),
            {
                "ENG_to_EKE",
                "EKE_to_ENG",
                "SWA_to_EKE",
                "EKE_to_SWA",
                "ENG_to_SWA",
                "SWA_to_ENG",
            },
        )
        for col in ["concept_id", "task_type", "prompt", "response"]:
            self.assertIn(col, tasks_df.columns)


if __name__ == "__main__":
    unittest.main()
