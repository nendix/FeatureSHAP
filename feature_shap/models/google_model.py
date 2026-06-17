from feature_shap.models.model_base import ModelBase
import os
from google import genai
from typing import Union, List


class GoogleModel(ModelBase):
    """
    A wrapper for Google's Gemini API that conforms to ModelBase.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.5-flash",
        generation_args: dict | None = None,
    ):
        """
        Initializes the GoogleModel.

        Args:
            api_key (str, optional): Your Google API key. Falls back to GOOGLE_API_KEY env var.
            model_name (str, optional): The name of the Google model to use.
            generation_args (dict | None, optional): A dictionary of arguments for the Google API.
        """
        self.model_name = model_name

        try:
            if api_key is None:
                api_key = os.getenv("GOOGLE_API_KEY")

            self.client = genai.Client(api_key=api_key)
        except Exception:
            raise ValueError("Failed to initialize Google GenAI client. Please check your API key and environment variables.")

        self.generation_args = generation_args or {
            "max_output_tokens": 32768,
        }

    def generate(self, batch: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generate text from a prompt or batch of prompts using the Google Gemini API.

        Args:
            batch (Union[str, List[str]]): A single prompt or list of prompts.

        Returns:
            Union[str, List[str]]: The generated text or list of generated texts.
        """
        prompts = [batch] if isinstance(batch, str) else batch

        config = genai.types.GenerateContentConfig(**self.generation_args)

        outputs = []
        for prompt in prompts:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            outputs.append(response.text.strip())

        return outputs if isinstance(batch, list) else outputs[0]
