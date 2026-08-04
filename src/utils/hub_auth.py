"""
HuggingFace Hub Gated-Repo Error Guidance
----------------------------------------------
Neither of this project's current base models -- Qwen2.5-7B-Instruct and
Mistral-7B-Instruct-v0.3 -- is gated on HuggingFace Hub (both are fully
open, no authentication needed). This module's predecessors filled this
role for Aya-23-8B and then Llama-3.1-8B-Instruct, both of which were
replaced specifically to avoid gated-repo friction blocking iteration on
Kineses Cloud. The guidance mechanism is kept as a generic safety net in
case a gated model is ever swapped in again: it turns the resulting
`OSError`/`GatedRepoError` (which surfaces as a long, generic-looking
traceback) into a short, actionable message pointing at exactly what to
do, rather than requiring the user to read HuggingFace's internals to
figure out what's wrong.
"""

from typing import NoReturn


def raise_with_access_guidance(exc: Exception, model_id: str) -> NoReturn:
    """Re-raise `exc` with actionable guidance if it's a gated-repo access error.

    Args:
        exc: The exception caught from a `from_pretrained()` call.
        model_id: The HuggingFace Hub repo ID that was being loaded (e.g.
            a gated model swapped in place of one of this project's
            default open models).

    Raises:
        OSError: Always -- either a rewritten, actionable version of `exc`
            (if it looks like a gated-repo/auth error) or `exc` itself
            unchanged (for any other failure, e.g. no internet access).
    """
    message = str(exc).lower()
    if "gated repo" in message or "access to model" in message or "401" in message:
        raise OSError(
            f"'{model_id}' is a gated model on HuggingFace Hub -- this is not a bug, "
            f"it requires two one-time steps:\n"
            f"  1. Visit https://huggingface.co/{model_id}, log in, and request/accept access "
            f"(most providers approve within minutes to a day).\n"
            f"  2. Authenticate THIS environment: run `huggingface-cli login` in a terminal/shell "
            f"cell and paste a token from https://huggingface.co/settings/tokens, or set the "
            f"HF_TOKEN environment variable before starting the kernel.\n"
            f"Original error: {exc}"
        ) from exc
    raise exc
