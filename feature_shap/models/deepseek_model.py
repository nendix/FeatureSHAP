from feature_shap.models.model_base import ModelBase
import os
from openai import OpenAI
from typing import Union, List


class DeepSeekModel(ModelBase):
    """
    A wrapper for DeepSeek's OpenAI-compatible API that conforms to ModelBase.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "deepseek-chat",
        generation_args: dict | None = None,
    ):
        """
        Initializes the DeepSeekModel.

        Args:
            api_key (str, optional): Your DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
            model_name (str, optional): The name of the DeepSeek model to use.
            generation_args (dict | None, optional): A dictionary of arguments for the API.
        """
        self.model_name = model_name

        try:
            if api_key is None:
                api_key = os.getenv("DEEPSEEK_API_KEY")

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        except Exception:
            raise ValueError("Failed to initialize DeepSeek client. Please check your API key and environment variables.")

        self.generation_args = generation_args or {
            "max_tokens": 4096,
        }

    def generate(self, batch: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generate text from a prompt or batch of prompts using the DeepSeek API.

        Args:
            batch (Union[str, List[str]]): A single prompt or list of prompts.

        Returns:
            Union[str, List[str]]: The generated text or list of generated texts.
        """
        prompts = [batch] if isinstance(batch, str) else batch

        outputs = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.generation_args,
            )

            outputs.append(response.choices[0].message.content.strip())

        return outputs if isinstance(batch, list) else outputs[0]
