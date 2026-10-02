import unittest
from unittest.mock import MagicMock, patch
from src.generator import GeminiGenerator, AUTOSAR_SYSTEM_PROMPT

class TestGenerator(unittest.TestCase):
    def test_missing_api_key_message(self):
        generator = GeminiGenerator(api_key="")
        response = generator.generate_answer("What is CanIf?", "Context: CanIf provides interface.")
        self.assertIn("GEMINI_API_KEY is not configured", response)
        self.assertIn("Context: CanIf provides interface", response)

    def test_system_prompt_structure(self):
        self.assertIn("AUTOSAR", AUTOSAR_SYSTEM_PROMPT)
        self.assertIn("Citations", AUTOSAR_SYSTEM_PROMPT)
        self.assertIn("Strict Adherence", AUTOSAR_SYSTEM_PROMPT)

    @patch("src.generator.GeminiGenerator._init_client")
    def test_mock_generation(self, mock_init):
        generator = GeminiGenerator(api_key="AIzaFakeKeyTest")
        generator._sdk_type = "legacy"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "According to [SWS_Can_00017], Can_SetControllerMode switches modes."
        mock_client.GenerativeModel().generate_content.return_value = mock_response
        generator._client = mock_client

        answer = generator.generate_answer(
            query="Explain Can_SetControllerMode",
            context="[SWS_Can_00017] Can_SetControllerMode(Controller, Transition)"
        )
        self.assertIn("SWS_Can_00017", answer)

if __name__ == "__main__":
    unittest.main()
