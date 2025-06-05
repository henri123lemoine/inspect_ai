import asyncio
import logging
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from typing_extensions import override

from inspect_ai._util.constants import DEFAULT_MAX_TOKENS
from inspect_ai.tool import ToolChoice, ToolInfo

from .._chat_message import ChatMessage, ChatMessageAssistant
from .._generate_config import GenerateConfig
from .._model import ModelAPI, simple_input_messages
from .._model_output import ChatCompletionChoice, ModelOutput, ModelUsage

# Environment variable names
# VLLM_BASE_URL = "VLLM_BASE_URL"
# VLLM_API_KEY = "VLLM_API_KEY"
VLLM_DEFAULT_SERVER_ARGS = "VLLM_DEFAULT_SERVER_ARGS"
VLLM_CONFIGURE_LOGGING = "VLLM_CONFIGURE_LOGGING"

# Set up logger for this module
logger = logging.getLogger(__name__)

# Type for ensemble combination function (operates on logits)
EnsembleCombineFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class VLLMEnsembleAPI(ModelAPI):
    """Ensemble model using two VLLM models with logit-level combination.

    This implementation performs true ensemble generation by:
    1. Getting logit distributions from both models at each token position
    2. Combining the logits using the specified combination function
    3. Sampling from the combined distribution
    4. Repeating until generation is complete
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # model_name should be in format "model1,model2"
        parts = model_name.split(",")
        if len(parts) != 2:
            raise ValueError(
                "Ensemble model name must be two model names separated by comma, e.g., 'model1,model2'"
            )

        self.model1_name = parts[0].strip()
        self.model2_name = parts[1].strip()

        if not self.model1_name or not self.model2_name:
            raise ValueError(
                "Ensemble model name must be two model names separated by comma"
            )

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=[],
            config=config,
        )

        # Collect model arguments
        def collect_model_arg(name: str) -> Any | None:
            return model_args.get(name, None)

        # Import VLLM here to check availability
        try:
            from vllm import LLM  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Error importing VLLM (required for vllm-ensemble provider): {e}"
            )

        # Common model arguments
        model_kwargs = {
            "tokenizer": collect_model_arg("tokenizer"),
            "tokenizer_mode": collect_model_arg("tokenizer_mode") or "auto",
            "trust_remote_code": collect_model_arg("trust_remote_code") or False,
            "tensor_parallel_size": collect_model_arg("tensor_parallel_size") or 1,
            "dtype": collect_model_arg("dtype") or "auto",
            "seed": collect_model_arg("seed") or 0,
            "device": collect_model_arg("device") or "auto",
        }

        # Initialize both models
        self.model1 = LLM(model=self.model1_name, **model_kwargs)
        self.model2 = LLM(model=self.model2_name, **model_kwargs)

        # Store chat template for later use
        self.chat_template = collect_model_arg("chat_template")

        # Get combination function (default to average)
        self.combine_fn: EnsembleCombineFn = (
            collect_model_arg("combine_fn") or ensemble_combine_average
        )

        # Use the tokenizer from the first model
        self.tokenizer = self.model1.get_tokenizer()

        # Verify models have compatible tokenizers
        tokenizer2 = self.model2.get_tokenizer()
        if len(self.tokenizer) != len(tokenizer2):
            raise ValueError(
                f"Models have different vocab sizes: {len(self.tokenizer)} vs {len(tokenizer2)}. "
                "Ensemble models must share the same tokenizer and vocabulary."
            )

    @override
    def close(self) -> None:
        """Close both models."""
        # VLLM doesn't provide an explicit close method, but we can delete the models
        del self.model1
        del self.model2

    @override
    def max_connections(self) -> int:
        return 1  # Only allow one request at a time for ensemble

    @override
    def collapse_user_messages(self) -> bool:
        return True

    @override
    def collapse_assistant_messages(self) -> bool:
        return True

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # Apply chat template
        prompt = self.apply_chat_template(input, tools)

        # Generate token by token with ensemble
        start_time = time.monotonic()
        output_text, input_tokens, output_tokens = await self._generate_ensemble(
            prompt, config
        )
        generation_time = time.monotonic() - start_time

        # Create model output
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=output_text,
                        model=self.model_name,
                        source="generate",
                    )
                )
            ],
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            time=generation_time,
        )

    def apply_chat_template(
        self, messages: list[ChatMessage], tools: list[ToolInfo]
    ) -> str:
        # Handle system message and consecutive user messages
        messages = simple_input_messages(messages)

        # Use custom chat template if provided
        if self.chat_template:
            # Simple template application - just concatenate message content
            return "".join([msg.text for msg in messages])
        # Use the tokenizer's chat template if available
        elif hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": msg.text} for msg in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback to simple concatenation
            return "".join([msg.text for msg in messages])

    async def _generate_ensemble(
        self, prompt: str, config: GenerateConfig
    ) -> tuple[str, int, int]:
        """Generate text using ensemble method with token-by-token logit combination."""
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tokens = len(input_ids)

        # Convert to tensor (use CPU to avoid device issues)
        current_ids = torch.tensor([input_ids])

        # Generation parameters
        max_tokens = config.max_tokens or DEFAULT_MAX_TOKENS
        temperature = config.temperature or 1.0
        top_p = config.top_p or 1.0
        top_k = config.top_k or -1

        # Stop tokens
        stop_token_ids = set()
        if config.stop_seqs:
            for stop_seq in config.stop_seqs:
                stop_ids = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                stop_token_ids.update(stop_ids)

        # Add EOS token to stop tokens
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)

        generated_tokens = []

        for _ in range(max_tokens):
            # Get logits from both models for the current sequence
            logits1 = await self._get_next_token_logits(
                self.model1, current_ids[0].tolist()
            )
            logits2 = await self._get_next_token_logits(
                self.model2, current_ids[0].tolist()
            )

            # Combine logits using the combination function
            combined_logits = self.combine_fn(logits1, logits2)

            # Apply temperature
            if temperature != 1.0:
                combined_logits = combined_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(combined_logits, top_k)
                combined_logits = torch.full_like(combined_logits, float("-inf"))
                combined_logits.scatter_(0, top_k_indices, top_k_logits)

            # Convert to probabilities
            probs = F.softmax(combined_logits, dim=-1)

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

                # Set probabilities to 0 for tokens to remove
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0
                # Renormalize
                probs = probs / probs.sum()

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Check for stop condition
            if next_token in stop_token_ids:
                break

            generated_tokens.append(next_token)

            # Update current sequence
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]])], dim=1)

        # Decode the generated tokens
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_tokens = len(generated_tokens)

        return output_text, input_tokens, output_tokens

    async def _get_next_token_logits(self, model, input_ids: list[int]) -> torch.Tensor:
        """Get logits for the next token from a VLLM model using top-k logprobs."""
        from vllm import SamplingParams

        def _get_logits():
            # Use VLLM's generate method with logprobs (limited to 20 by VLLM)
            sampling_params = SamplingParams(
                max_tokens=1,
                temperature=1.0,
                logprobs=20,  # VLLM's maximum allowed logprobs
            )

            # Convert input_ids back to text for generation
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            outputs = model.generate([prompt], sampling_params)

            # Extract logprobs from the output
            output = outputs[0]
            if output.outputs and output.outputs[0].logprobs:
                # Get the logprobs for the first generated token
                token_logprobs = output.outputs[0].logprobs[0]

                # Create a tensor with very low logits for all vocabulary tokens
                vocab_size = len(self.tokenizer)
                logits = torch.full(
                    (vocab_size,), -100.0
                )  # Very low logits for unseen tokens

                # Fill in the available logprobs (top 20)
                for token_id, logprob in token_logprobs.items():
                    # Extract the actual logprob value (logprob is a Logprob object)
                    logprob_value = (
                        logprob.logprob
                        if hasattr(logprob, "logprob")
                        else float(logprob)
                    )
                    logits[token_id] = logprob_value

                return logits
            else:
                # Fallback: return uniform distribution
                vocab_size = len(self.tokenizer)
                return torch.zeros(vocab_size)

        # Run in thread to avoid blocking
        return await asyncio.to_thread(_get_logits)


# Ensemble combination functions operating on logits


def ensemble_combine_average(
    logits1: torch.Tensor, logits2: torch.Tensor
) -> torch.Tensor:
    """Average the logits from both models (default behavior)."""
    return (logits1 + logits2) / 2.0


def ensemble_combine_weighted(
    weight1: float = 0.5, weight2: float = 0.5
) -> EnsembleCombineFn:
    """Create a weighted combination function with custom weights."""

    def combine(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        return weight1 * logits1 + weight2 * logits2

    return combine


def ensemble_combine_max(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """Take the maximum logit for each token across both models."""
    return torch.maximum(logits1, logits2)


def ensemble_combine_multiplicative(
    logits1: torch.Tensor, logits2: torch.Tensor
) -> torch.Tensor:
    """Multiply probabilities (equivalent to adding log probabilities)."""
    # Convert to probabilities, multiply, then back to logits
    probs1 = F.softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)
    combined_probs = probs1 * probs2
    # Renormalize and convert back to logits
    combined_probs = combined_probs / combined_probs.sum()
    return torch.log(combined_probs + 1e-10)  # Add small epsilon to avoid log(0)


def ensemble_combine_filtered(threshold: float = 0.02) -> EnsembleCombineFn:
    """Filter tokens from first model based on probability threshold.

    Then use second model's distribution over remaining tokens.
    """

    def combine(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        # Convert first model's logits to probabilities
        probs1 = F.softmax(logits1, dim=-1)

        # Create mask for tokens above threshold in model1
        high_prob_mask = probs1 >= threshold

        # Start with second model's logits
        combined_logits = logits2.clone()

        # Mask out tokens that don't meet the threshold
        combined_logits[~high_prob_mask] = float("-inf")

        return combined_logits

    return combine
