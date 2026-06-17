from feature_shap.models.model_base import ModelBase
from feature_shap.modifiers import ModifierBase
from feature_shap.splitters.splitter_base import SplitterBase
from feature_shap.comparators.comparator_base import ComparatorBase

from typing import Optional, List, Tuple, Dict
import logging
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import re

logger = logging.getLogger(__name__)


class FeatureSHAP:
    """
    A class to perform FeatureSHAP analysis on text generation models.

    This class orchestrates the process of splitting a prompt into features,
    modifying them, generating responses from a model, comparing the responses
    to a baseline, and calculating Shapley values to determine feature importance.
    """
    def __init__(self,
                 model: ModelBase = None,
                 splitter: SplitterBase = None,
                 modifier: ModifierBase = None,
                 comparator: Optional[ComparatorBase] = None,
                 instruction: str = '',
                 debug: bool = False,
                 batch_size: int = 16):
        """
        Initialize the FeatureSHAP analyzer.

        Args:
            model (ModelBase, optional): The language model to analyze.
            splitter (SplitterBase, optional): The strategy to split the prompt into features.
            modifier (ModifierBase, optional): The strategy to modify/perturb features.
            comparator (ComparatorBase, optional): The metric to compare generated responses.
            instruction (str, optional): An instruction to prepend to the prompt (e.g., system prompt).
            debug (bool, optional): Whether to print debug information. Defaults to False.
            batch_size (int, optional): The batch size for model generation. Defaults to 16.
        """
        self.shapley_values = None
        self.baseline_texts = None
        self.model = model
        self.splitter = splitter
        self.modifier = modifier
        self.comparator = comparator
        self.debug = debug
        self.instruction = instruction
        self.batch_size = batch_size
        self.tokenizer = self.model.tokenizer if hasattr(self.model, 'tokenizer') else None


    def _debug_print(self, message):
        """
        Print a debug message if debug mode is enabled.

        Args:
            message (str): The message to print.
        """
        if self.debug:
            print(message)

    def _extract_backtick_content(self, text: str) -> str:
        """
        Extracts content from the first triple backtick block in the text.

        If no triple backticks are found, it tries to find single backticks or returns the original text.

        Args:
            text (str): The text to extract content from.

        Returns:
            str: The extracted content or the original text.
        """
        if not isinstance(text, str):
            return text

        match = re.search(r'```(?:[a-zA-Z0-9]+)?\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r'```(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Unterminated triple backticks
        match = re.search(r'```(?:[a-zA-Z0-9]+)?\n(.*)$', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text

    def _format(self, data):
        """
        Formats the input data with the instruction, potentially using a chat template.

        Args:
            data (str): The input text data.

        Returns:
            str: The formatted prompt string.
        """
        if self.tokenizer and getattr(self.tokenizer, 'chat_template', None):
            msgs = [{'role': 'user', 
                     'content': f'```\n{data}\n```\n{self.instruction}'}]
            return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        else:
            return f'```\n{data}\n```\n{self.instruction}'

    def _generate_random_combinations(self, clean_features, perturbed_features, k, exclude_combinations_set):
        """
        Generates random combinations of clean and perturbed features.

        Args:
            clean_features (List[str]): The list of original features.
            perturbed_features (List[str]): The list of perturbed features.
            k (int): The number of combinations to generate.
            exclude_combinations_set (set): A set of combinations to exclude (e.g., already generated).

        Returns:
            List[Tuple[Tuple[str], Tuple[int]]]: A list of tuples, where each tuple contains the combined features and the indexes of clean features.
        """
        n = len(clean_features)
        sampled_combinations_set = set()
        max_attempts = k * 100
        attempts = 0

        while len(sampled_combinations_set) < k and attempts < max_attempts:
            attempts += 1
            rand_int = random.randint(1, 2 ** n - 2)
            bin_str = bin(rand_int)[2:].zfill(n)

            combination = []
            for i in range(n):
                if bin_str[i] == '0':
                    combination.append(perturbed_features[i])
                else:
                    combination.append(clean_features[i])

            indexes = tuple([i + 1 for i in range(n) if bin_str[i] == '1'])
            if indexes not in exclude_combinations_set and indexes not in sampled_combinations_set:
                sampled_combinations_set.add((tuple(combination), indexes))

        return list(sampled_combinations_set)

    def _get_combinations_for_prompt(self, prompt: str, sampling_ratio: float) -> List[Tuple[List, Tuple]]:
        """
        Generate all feature combinations for a single prompt.
        
        Args:
            prompt (str): The input prompt.
            sampling_ratio (float): The ratio of additional random combinations to sample.

        Returns:
            List[Tuple[List, Tuple]]: A list of (combination, indexes) tuples.
        """
        clean_features = self.splitter.split(prompt)
        n = len(clean_features)
        self._debug_print(f"Number of features: {n}")

        perturbed_features = [self.modifier.modify(clean_features[i], prompt) for i in range(n)]

        # Essential combinations (one feature perturbed at a time)
        essential_combinations = []
        essential_combinations_set = set()

        for i in range(n):
            perturbed_sample = clean_features[:i] + [perturbed_features[i]] + clean_features[i + 1:]
            indexes = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((perturbed_sample, indexes))
            essential_combinations_set.add(indexes)

        # Additional-sampled combinations
        num_total_combinations = 2 ** n - 1
        num_sampled_combinations = int(num_total_combinations * sampling_ratio)
        num_additional_samples = max(0, num_sampled_combinations - len(essential_combinations))

        sampled_combinations = []
        if num_additional_samples > 0:
            sampled_combinations = self._generate_random_combinations(
                clean_features, perturbed_features, num_additional_samples, essential_combinations_set
            )

        return essential_combinations + sampled_combinations

    def _batch_generate(self, texts: List[str]) -> List[str]:
        """
        Generate responses for all texts in batches.

        Args:
            texts (List[str]): A list of input texts.

        Returns:
            List[str]: A list of generated responses.
        """
        if not texts:
            return []
        
        # Handle single text
        if len(texts) == 1:
            result = self.model.generate(texts[0])
            results = [result] if isinstance(result, str) else result
            return [self._extract_backtick_content(r) for r in results]

        results = []
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        for batch in tqdm(batches, desc="Batch generation"):
            batch_outputs = self.model.generate(batch)
            logger.debug("Batch outputs:\n%s", "\n---\n".join(str(o) for o in batch_outputs))
            if isinstance(batch_outputs, str):
                batch_outputs = [batch_outputs]
            
            processed_outputs = [self._extract_backtick_content(r) for r in batch_outputs]
            results.extend(processed_outputs)

        return results

    def _get_result_per_feature_combination(self, prompt: str, sampling_ratio: float) -> Dict:
        """
        Process a single prompt using batch infrastructure.
        
        Args:
            prompt (str): The input prompt.
            sampling_ratio (float): The sampling ratio for combinations.

        Returns:
            Dict: A dictionary mapping prompt keys to (response, indexes) tuples.
        """
        all_combinations = self._get_combinations_for_prompt(prompt, sampling_ratio)

        # Collect all texts
        all_texts = []
        metadata = []
        for combination, indexes in all_combinations:
            text = self.splitter.join(combination)
            all_texts.append(self._format(text))
            metadata.append({
                'indexes': indexes,
                'original_text': text
            })

        # Batch generate
        all_responses = self._batch_generate(all_texts)

        # Build prompt_responses dict
        prompt_responses = {}
        for response, meta in zip(all_responses, metadata):
            indexes = meta['indexes']
            original_text = meta['original_text']
            prompt_key = original_text + '_' + ','.join(str(idx) for idx in indexes)
            prompt_responses[prompt_key] = (response, indexes)

        return prompt_responses

    def _get_result_per_feature_combination_batch(self, prompts: List[str], sampling_ratio: float) -> List[Dict]:
        """
        Process multiple prompts, collecting all combinations and batching model calls.
        
        Args:
            prompts (List[str]): A list of input prompts.
            sampling_ratio (float): The sampling ratio for combinations.

        Returns:
            List[Dict]: A list of prompt_responses dicts (one per input prompt).
        """
        all_texts = []
        text_metadata = []

        # Step 1: Generate all combinations for all prompts
        for prompt_idx, prompt in enumerate(prompts):
            all_combinations = self._get_combinations_for_prompt(prompt, sampling_ratio)

            for combination, indexes in all_combinations:
                text = self.splitter.join(combination)
                all_texts.append(self._format(text))
                text_metadata.append({
                    'prompt_idx': prompt_idx,
                    'indexes': indexes,
                    'original_text': text
                })

        # Step 2: Batch-generate all responses
        all_responses = self._batch_generate(all_texts)

        # Step 3: Map results back to per-prompt dictionaries
        results_per_prompt = [{} for _ in prompts]

        for response, meta in zip(all_responses, text_metadata):
            prompt_idx = meta['prompt_idx']
            indexes = meta['indexes']
            original_text = meta['original_text']

            prompt_key = original_text + '_' + ','.join(str(idx) for idx in indexes)
            results_per_prompt[prompt_idx][prompt_key] = (response, indexes)

        return results_per_prompt

    def _get_df_per_feature_combination(self, prompt_responses: Dict, baseline_text: str) -> pd.DataFrame:
        """
        Converts prompt responses into a DataFrame and calculates similarities.

        Args:
            prompt_responses (Dict): The dictionary of prompt responses.
            baseline_text (str): The baseline text for comparison.

        Returns:
            pd.DataFrame: A DataFrame containing prompts, responses, feature indexes, and similarity scores.
        """
        df = pd.DataFrame(
            [(prompt.split('_')[0], response[0], response[1])
             for prompt, response in prompt_responses.items()],
            columns=['Prompt', 'Response', 'Feature_Indexes']
        )

        # Compute similarity
        similarities = []
        for response in df["Response"]:
            similarity = self.comparator.compare(baseline_text, response)
            similarities.append(similarity)

        df["Similarity"] = similarities

        return df

    def _calculate_shapley_values(self, df_per_feature_combination: pd.DataFrame, prompt: str) -> Dict:
        """
        Calculates Shapley values for the features based on the similarity scores.

        Args:
            df_per_feature_combination (pd.DataFrame): The DataFrame with feature combinations and similarities.
            prompt (str): The original prompt.

        Returns:
            Dict: A dictionary mapping feature identifiers to their normalized Shapley values.
        """
        def normalize_shapley_values(shapley_values, power=1):
            min_value = min(shapley_values.values())
            shifted_values = {k: v - min_value for k, v in shapley_values.items()}
            powered_values = {k: v ** power for k, v in shifted_values.items()}
            total = sum(powered_values.values())
            if total == 0:
                return {k: 1 / len(powered_values) for k in powered_values}
            normalized_values = {k: v / total for k, v in powered_values.items()}
            return normalized_values

        samples = self.splitter.split(prompt)
        n = len(samples)
        shapley_values = {}

        for i, sample in enumerate(samples, start=1):
            with_sample = np.average(
                df_per_feature_combination[
                    df_per_feature_combination["Feature_Indexes"].apply(lambda x: i in x)
                ]["Similarity"].values
            )
            without_sample = np.average(
                df_per_feature_combination[
                    df_per_feature_combination["Feature_Indexes"].apply(lambda x: i not in x)
                ]["Similarity"].values
            )

            shapley_values[sample + str(i)] = with_sample - without_sample

        return normalize_shapley_values(shapley_values)

    def _generate_baseline_batch(self, prompts: List[str]) -> List[str]:
        """
        Generates baseline responses for a batch of prompts.

        Args:
            prompts (List[str]): A list of prompts.

        Returns:
            List[str]: A list of baseline responses.
        """
        formatted_prompts = []
        for p in prompts:
            if isinstance(p, list):
                p = self.splitter.join(p)
            formatted_prompts.append(self._format(p))

        return self._batch_generate(formatted_prompts)

    # ==================== Main Analysis Methods ====================
    def analyze(
            self,
            prompt: str,
            sampling_ratio: float = 0.0,
            baseline: str = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Analyze a single prompt for feature importance.
        
        Args:
            prompt (str): The input prompt to analyze.
            sampling_ratio (float): Ratio of additional combinations to sample (0.0 to 1.0).
            baseline (str, optional): The baseline text to use. If None, it is generated.

        Returns:
            Tuple[Dict, pd.DataFrame]: A tuple containing the Shapley values dictionary and the DataFrame with detailed results.
        """
        prompts = [prompt]

        # Generate or use the provided baseline
        if baseline is not None:
            self.baseline_texts = [baseline]
        else:
            self.baseline_texts = self._generate_baseline_batch(prompts)

        # Get all feature combination results (batched)
        all_feature_results = self._get_result_per_feature_combination_batch(prompts, sampling_ratio)

        # Process each prompt's results
        all_dfs = []
        all_shapley_values = []

        for prompt_idx, prompt in enumerate(prompts):
            prompt_responses = all_feature_results[prompt_idx]
            baseline_text = self.baseline_texts[prompt_idx]

            df = self._get_df_per_feature_combination(prompt_responses, baseline_text)
            df["Baseline"] = baseline_text
            shapley_values = self._calculate_shapley_values(df, prompt)

            all_dfs.append(df)
            all_shapley_values.append(shapley_values)

        return all_shapley_values[0], all_dfs[0]


    def analyze_batch(
            self,
            prompts: List[str],
            sampling_ratio: float = 0.0,
            baselines: List[str] = None
    ) -> Tuple[List[Dict], List[pd.DataFrame]]:
        """
        Analyze a batch of prompts for feature importance.

        Args:
            prompts (List[str]): A list of input prompts to analyze.
            sampling_ratio (float): Ratio of additional combinations to sample (0.0 to 1.0).
            baselines (List[str], optional): A list of baseline texts. If None, they are generated.

        Returns:
            Tuple[List[Dict], List[pd.DataFrame]]: A tuple containing a list of Shapley values dictionaries and a list of DataFrames with detailed results.
        """
        # Generate baselines
        if baselines is not None:
            self.baseline_texts = baselines
        else:
            self.baseline_texts = self._generate_baseline_batch(prompts)

        # Get all feature combination results (batched)
        all_feature_results = self._get_result_per_feature_combination_batch(prompts, sampling_ratio)

        # Process each prompt's results
        all_dfs = []
        all_shapley_values = []

        for prompt_idx, prompt in enumerate(prompts):
            prompt_responses = all_feature_results[prompt_idx]
            baseline_text = self.baseline_texts[prompt_idx]

            df = self._get_df_per_feature_combination(prompt_responses, baseline_text)
            df["Baseline"] = baseline_text
            shapley_values = self._calculate_shapley_values(df, prompt)

            all_dfs.append(df)
            all_shapley_values.append(shapley_values)

        return all_shapley_values, all_dfs
