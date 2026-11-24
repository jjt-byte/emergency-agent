#!/usr/bin/env python3
"""Offline conversion for the downloaded, tied-embedding Qwen2.5-3B model.

Uses llama.cpp's GGUF writer and tensor mapping, plus safetensors/PyTorch.
The known working Qwen GGUF verifies the BPE vocabulary and merge table.
This deliberately supports only this architecture, without Transformers.
"""
import argparse
import os
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

LLAMA = Path(os.environ.get('LLAMA_CPP_DIR', str(Path(__file__).resolve().parents[1]/'vendor/llama.cpp')))
sys.path.insert(0, str(LLAMA / 'gguf-py'))
import gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source
    config = json.loads((source / 'config.json').read_text())
    required = dict(model_type='qwen2', num_hidden_layers=36, hidden_size=2048,
                    num_attention_heads=16, num_key_value_heads=2,
                    vocab_size=151936, tie_word_embeddings=True)
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported configuration: {key}')
    if config.get('use_sliding_window') or config.get('rope_scaling'):
        raise ValueError('Unsupported attention/rope configuration')
    tokenizer = json.loads((source / 'tokenizer.json').read_text())
    regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    expected_pre = {'type': 'Sequence', 'pretokenizers': [
        {'type': 'Split', 'pattern': {'Regex': regex}, 'behavior': 'Isolated', 'invert': False},
        {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False}]}
    if tokenizer['model']['type'] != 'BPE' or tokenizer['pre_tokenizer'] != expected_pre:
        raise ValueError('Unsupported tokenizer')
    reference = gguf.GGUFReader(args.reference)
    if reference.fields['tokenizer.ggml.pre'].contents() != 'qwen2':
        raise ValueError('Reference must use qwen2 BPE')
    reference_tokens = reference.fields['tokenizer.ggml.tokens'].contents()
    tokens = [f'[PAD{i}]' for i in range(config['vocab_size'])]
    types = [gguf.TokenType.UNUSED] * len(tokens)
    for token, i in tokenizer['model']['vocab'].items():
        if reference_tokens[i] != token:
            raise ValueError(f'Vocabulary mismatch at {i}')
        tokens[i], types[i] = token, gguf.TokenType.NORMAL
    for entry in tokenizer['added_tokens']:
        token, i = entry['content'], entry['id']
        if reference_tokens[i] != token:
            raise ValueError(f'Added token mismatch at {i}')
        special = entry['special'] or (token.startswith('<|') and token.endswith('|>'))
        tokens[i] = token
        types[i] = gguf.TokenType.CONTROL if special else gguf.TokenType.USER_DEFINED
    if reference.fields['tokenizer.ggml.merges'].contents() != tokenizer['model']['merges']:
        raise ValueError('BPE merge mismatch')
    del reference

    index = json.loads((source / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    if len(weight_map) != 434 or 'lm_head.weight' in weight_map:
        raise ValueError('Unexpected tensor set')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial.gguf')
    if temporary.exists():
        raise FileExistsError(temporary)
    writer = gguf.GGUFWriter(temporary, 'qwen2')
    writer.add_type('model')
    writer.add_name('Qwen2.5-3B-Instruct')
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_block_count(config['num_hidden_layers'])
    writer.add_context_length(config['max_position_embeddings'])
    writer.add_embedding_length(config['hidden_size'])
    writer.add_feed_forward_length(config['intermediate_size'])
    writer.add_head_count(config['num_attention_heads'])
    writer.add_head_count_kv(config['num_key_value_heads'])
    writer.add_rope_freq_base(config['rope_theta'])
    writer.add_layer_norm_rms_eps(config['rms_norm_eps'])
    writer.add_tokenizer_model('gpt2')
    writer.add_tokenizer_pre('qwen2')
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    gguf.SpecialVocab(source, load_merges=True).add_to_gguf(writer)
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN2, config['num_hidden_layers'])
    seen = set()
    manifest = []
    torch.set_num_threads(8)
    for shard in sorted(set(weight_map.values())):
        with safe_open(source / shard, framework='pt', device='cpu') as weights:
            expected = {name for name, filename in weight_map.items() if filename == shard}
            if set(weights.keys()) != expected:
                raise ValueError(f'Shard tensor index mismatch: {shard}')
            for name in sorted(weights.keys()):
                target = mapping.get_name(name, try_suffixes=('.weight', '.bias'))
                if target is None or target in seen:
                    raise ValueError(f'Unmapped/duplicate tensor: {name}')
                tensor = weights.get_tensor(name)
                dtype = torch.float32 if tensor.ndim == 1 else torch.float16
                array = tensor.to(dtype).numpy()
                writer.add_tensor(target, array)
                manifest.append(dict(source=name, target=target, shape=list(array.shape),
                                     dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest()))
                seen.add(target)
        print(f'Prepared {shard}: {len(seen)}/434 tensors', flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    check = gguf.GGUFReader(temporary)
    tensors = {tensor.name: tensor for tensor in check.tensors}
    if set(tensors) != seen:
        raise ValueError('Written tensor inventory mismatch')
    for entry in manifest:
        tensor = tensors[entry['target']]
        if list(tensor.data.shape) != entry['shape']:
            raise ValueError(f'Written tensor shape mismatch: {entry["target"]}')
        if hashlib.sha256(tensor.data.tobytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Written tensor data mismatch: {entry["target"]}')
    del check
    temporary.rename(args.output)
    args.output.with_suffix('.conversion.json').write_text(json.dumps({
        'source': str(source), 'output': str(args.output),
        'converter': str(Path(__file__).resolve()), 'tensors': manifest,
        'verification': 'all 434 written tensors verified against converted source arrays',
        'tokenizer': 'qwen2; base/added tokens and merges match working reference; chat template from source',
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'Conversion and full tensor verification complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
