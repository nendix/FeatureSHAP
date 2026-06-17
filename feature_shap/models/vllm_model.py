from feature_shap.models import ModelBase

from typing import Union, List
from vllm import LLM, SamplingParams


class VLLMModel(ModelBase):
    """
    A model wrapper that leverages the vLLM library for faster inference of HuggingFace models.
    """
    def __init__(self, model_name_or_path, generation_args=None, tensor_parallel_size=1, max_len=1024):
        """
        Initializes the vLLM-powered model and tokenizer.

        Args:
            model_name_or_path (str): The identifier for a HuggingFace model (e.g., "meta-llama/Llama-2-7b-chat-hf")
                                      or a local path to a model checkpoint.
            generation_args (dict, optional): A dictionary of arguments to control text generation.
                                              These are mapped to vLLM's `SamplingParams`. If not provided,
                                              default values for deterministic generation are used.
                                              Example: {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}.
            tensor_parallel_size (int, optional): The number of GPUs to use for tensor parallelism.
                                                  Defaults to 1.
            max_len (int, optional): The maximum total sequence length (prompt + generated tokens)
                                     the model can handle. Defaults to 1024.
        """
        self.model_name_or_path = model_name_or_path
        self.max_len = max_len

        # Initialize vLLM model
        print('loading', model_name_or_path)
        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=self.max_len,
            dtype="bfloat16",  # Equivalent to torch.bfloat16
            device_map="auto"
        )
        self.tokenizer = self.model.get_tokenizer()

        # Default generation args
        self.generation_args = generation_args or {
            "max_new_tokens": 256,
            "num_beams": 1,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }


    def generate(self, batch: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generates text from a given prompt or batch of prompts using the vLLM engine.

        Args:
            batch (Union[str, List[str]]): A single prompt or a list of prompts.

        Returns:
            Union[str, List[str]]: The generated text or a list of generated texts.
        """
        # Convert generation args to vLLM's SamplingParams
        sampling_params = SamplingParams(
            max_tokens=self.generation_args.get("max_new_tokens", 32768),
            seed=42,        # set your seed here
            temperature=self.generation_args.get("temperature", 0.0),
            truncate_prompt_tokens=self.max_len-1
        )

        # Handle a single prompt or batch
        prompts = [batch] if isinstance(batch, str) else batch

        # Handle prompts longer than max_len
        truncated = []
        for p in prompts:
            toks = self.tokenizer.encode(p, add_special_tokens=False)

            if len(toks) > self.max_len:
                toks = toks[-(self.max_len-2):]   # keep only the last max_prompt tokens

            # decode back to text
            truncated.append(self.tokenizer.decode(toks, skip_special_tokens=True))

        prompts = truncated

        # Generate completions with vLLM's batch processing
        outputs = self.model.generate(prompts, sampling_params=sampling_params)

        # Extract generated texts
        generated_texts = []
        for output in outputs:
            if output.outputs:
                generated_texts.append(output.outputs[0].text.strip())
            else:
                generated_texts.append("")
        
        # Return string for the single input, list for multiple inputs
        if isinstance(batch, list):
            return generated_texts
        else:
            return generated_texts[0]
