import pytest
from test_helpers.utils import skip_if_github_action, skip_if_no_vllm

from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.model._providers.vllm_ensemble import (
    ensemble_combine_filtered,
    ensemble_combine_max,
    ensemble_combine_weighted,
)


@pytest.mark.anyio
@skip_if_github_action
@skip_if_no_vllm
async def test_vllm_ensemble_api() -> None:
    model = get_model(
        "vllm-ensemble/gpt2,gpt2",
        config=GenerateConfig(
            max_tokens=5,
            seed=42,
            temperature=0.7,
            top_p=0.9,
            top_k=None,
        ),
        device=0,
        # this allows us to run base models with the chat message scaffolding:
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    message = ChatMessageUser(content="Lorem ipsum dolor")
    response = await model.generate(input=[message])
    print(f">>> {response}")
    assert len(response.completion) >= 1


@pytest.mark.anyio
@skip_if_github_action
@skip_if_no_vllm
async def test_vllm_ensemble_weighted() -> None:
    model = get_model(
        "vllm-ensemble/gpt2,gpt2",
        config=GenerateConfig(
            max_tokens=3,
            seed=42,
            temperature=0.7,
        ),
        device=0,
        combine_fn=ensemble_combine_weighted(0.7, 0.3),
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    message = ChatMessageUser(content="The quick brown")
    response = await model.generate(input=[message])
    assert len(response.completion) >= 1


@pytest.mark.anyio
@skip_if_github_action
@skip_if_no_vllm
async def test_vllm_ensemble_max() -> None:
    model = get_model(
        "vllm-ensemble/gpt2,gpt2",
        config=GenerateConfig(
            max_tokens=3,
            seed=42,
            temperature=0.7,
        ),
        device=0,
        combine_fn=ensemble_combine_max,
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    message = ChatMessageUser(content="The quick brown")
    response = await model.generate(input=[message])
    assert len(response.completion) >= 1


@pytest.mark.anyio
@skip_if_github_action
@skip_if_no_vllm
async def test_vllm_ensemble_filtered() -> None:
    model = get_model(
        "vllm-ensemble/gpt2,gpt2",
        config=GenerateConfig(
            max_tokens=3,
            seed=42,
            temperature=0.7,
        ),
        device=0,
        combine_fn=ensemble_combine_filtered(threshold=0.05),
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    message = ChatMessageUser(content="The quick brown")
    response = await model.generate(input=[message])
    assert len(response.completion) >= 1


@pytest.mark.anyio
@skip_if_github_action
@skip_if_no_vllm
async def test_vllm_ensemble_vocab_mismatch() -> None:
    """Test that models with different vocabularies raise an error."""
    with pytest.raises(ValueError, match="Models have different vocab sizes"):
        # Using different models with different tokenizers to test vocab mismatch
        get_model(
            "vllm-ensemble/gpt2,facebook/opt-125m",  # GPT-2 and OPT have different vocabularies
            device=0,
        )
        # The error should be raised during initialization, not generation
