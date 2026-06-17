from feature_shap.models.model_base import ModelBase
import os
from openai import OpenAI
from typing import Union, List


class OpenAIModel(ModelBase):
    """
    A wrapper for OpenAI's Completions API that conforms to ModelBase.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        generation_args: dict | None = None,
    ):
        """
        Initializes the OpenAIModel.

        Args:
            api_key (str): Your OpenAI API key.
            model_name (str, optional): The name of the OpenAI model to use.
            generation_args (dict | None, optional): A dictionary of arguments for the OpenAI API.
        """
        self.model_name = model_name

        try:
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")

            self.client = OpenAI(api_key=api_key)
        except Exception:
            raise ValueError("Failed to initialize OpenAI client. Please check your API key and environment variables.")

        # Default generation args
        self.generation_args = generation_args or {
            "max_output_tokens": 32768,
            # "temperature": 0.0,
        }


    def generate(self, batch: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generate text from a prompt or batch of prompts using the OpenAI API.

        Args:
            batch (Union[str, List[str]]): A single prompt or list of prompts.

        Returns:
            Union[str, List[str]]: The generated text or list of generated texts.
        """
        prompts = [batch] if isinstance(batch, str) else batch

        outputs = []
        for prompt in prompts:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                **self.generation_args,
            )

            # Extract generated text
            outputs.append(response.output_text.strip())

        return outputs if isinstance(batch, list) else outputs[0]
