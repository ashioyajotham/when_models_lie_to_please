import os
import transformers

# Save references to original transformers from_pretrained classmethods
_orig_model_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
_orig_tokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained


def patched_model_from_pretrained(cls, *args, **kwargs):
    if os.environ.get("MOCK_PIPELINE") == "true":
        from src.utils.mock_utils import MockModel
        return MockModel.from_pretrained(*args, **kwargs)
    return _orig_model_from_pretrained(*args, **kwargs)


def patched_tokenizer_from_pretrained(cls, *args, **kwargs):
    if os.environ.get("MOCK_PIPELINE") == "true":
        from src.utils.mock_utils import MockTokenizer
        return MockTokenizer.from_pretrained(*args, **kwargs)
    return _orig_tokenizer_from_pretrained(*args, **kwargs)


# Apply the classmethods unconditionally at import time
transformers.AutoModelForCausalLM.from_pretrained = classmethod(patched_model_from_pretrained)
transformers.AutoTokenizer.from_pretrained = classmethod(patched_tokenizer_from_pretrained)

