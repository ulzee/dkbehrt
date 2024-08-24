
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers.utils import PaddingStrategy

TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

def _pad(
    self,
    encoded_inputs,
    max_length: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
) -> dict:
    """
    Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

    Args:
        encoded_inputs:
            Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
        max_length: maximum length of the returned list and optionally padding length (see below).
            Will truncate by taking into account the special tokens.
        padding_strategy: PaddingStrategy to use for padding.

            - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
            - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
            - PaddingStrategy.DO_NOT_PAD: Do not pad
            The tokenizer padding sides are defined in self.padding_side:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
        pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
            `>= 7.5` (Volta).
        return_attention_mask:
            (optional) Set to False to avoid returning attention mask (default: set to model specifics)
    """
    # Load from model defaults
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in self.model_input_names

    required_input = encoded_inputs[self.model_input_names[0]]

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(required_input)

    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

    # Initialize attention mask if not present.
    if return_attention_mask and "attention_mask" not in encoded_inputs:
        encoded_inputs["attention_mask"] = [1] * len(required_input)

    if needs_to_be_padded:
        difference = max_length - len(required_input)

        if self.padding_side == "right":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                )
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = (
                    encoded_inputs["position_ids"] + [0] * difference
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
            encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
        else:
            raise ValueError(f"Not implemented")

    return encoded_inputs


@dataclass
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    The default data MLM collator sucks when trying to pass anything other than default set of inputs.
    Supports passing more information to the model.
    """

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        # NOTE: NEW
        ilen = batch['input_ids'].shape[-1]
        batch['position_ids'] = torch.tensor([s['position_ids'][:ilen] + [0]*(max(0, ilen - len(s['position_ids']))) for s in examples])

        return batch

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    # NOTE: not modified
    # https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/data/data_collator.py

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    # NOTE: not modified
    # https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/data/data_collator.py
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result