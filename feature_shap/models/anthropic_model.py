from feature_shap.models.model_base import ModelBase
import os
from anthropic import Anthropic
from typing import Union, List


class AnthropicModel(ModelBase):
    """
    A wrapper for Anthropic's Messages API that conforms to ModelBase.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "claude-sonnet-4-20250514",
        generation_args: dict | None = None,
    ):
        """
        Initializes the AnthropicModel.

        Args:
            api_key (str, optional): Your Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            model_name (str, optional): The name of the Anthropic model to use.
            generation_args (dict | None, optional): A dictionary of arguments for the Anthropic API.
        """
        self.model_name = model_name

        try:
            if api_key is None:
                api_key = os.getenv("ANTHROPIC_API_KEY")

            self.client = Anthropic(api_key=api_key)
        except Exception:
            raise ValueError("Failed to initialize Anthropic client. Please check your API key and environment variables.")

        self.generation_args = generation_args or {
            "max_tokens": 32768,
        }

    def generate(self, batch: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generate text from a prompt or batch of prompts using the Anthropic API.

        Args:
            batch (Union[str, List[str]]): A single prompt or list of prompts.

        Returns:
            Union[str, List[str]]: The generated text or list of generated texts.
        """
        prompts = [batch] if isinstance(batch, str) else batch

        outputs = []
        for prompt in prompts:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.generation_args,
            )

            outputs.append(response.content[0].text.strip())

        return outputs if isinstance(batch, list) else outputs[0]
