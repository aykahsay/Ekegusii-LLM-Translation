"""
HuggingFace Hub Gated-Repo Error Guidance
----------------------------------------------
Llama-3.1-8B-Instruct is a gated model: HuggingFace requires a logged-in
account to have requested/accepted access on the model's page, AND the
local environment to be authenticated (an `HF_TOKEN` env var or
`huggingface-cli login`), before `from_pretrained()` will download
anything. (Qwen2.5-7B-Instruct, the project's other base model, is fully
open and never hits this path -- Aya-23-8B originally filled this role but
was replaced specifically to avoid this friction; Llama's license terms
still require it regardless.) Neither of Llama's two access steps can be
automated from code -- this module exists only to turn the resulting
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
            "meta-llama/Llama-3.1-8B-Instruct").

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
            f"(Cohere/Meta typically approve within minutes to a day).\n"
            f"  2. Authenticate THIS environment: run `huggingface-cli login` in a terminal/shell "
            f"cell and paste a token from https://huggingface.co/settings/tokens, or set the "
            f"HF_TOKEN environment variable before starting the kernel.\n"
            f"Original error: {exc}"
        ) from exc
    raise exc
