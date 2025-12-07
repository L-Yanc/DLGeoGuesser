"""
Inference engine for efficient text generation with KV caching.
"""

import torch
import torch.nn.functional as F


class KVCache:
    """
    KV cache for efficient autoregressive generation.
    Stores keys and values from attention layers to avoid recomputation.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        Args:
            batch_size: Batch size
            num_heads: Number of KV heads
            seq_len: Maximum sequence length
            head_dim: Dimension of each attention head
            num_layers: Number of transformer layers
        """
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0  # current position in the cache

    def reset(self):
        """Reset cache position to beginning."""
        self.pos = 0

    def get_pos(self):
        """Get current cache position."""
        return self.pos

    def prefill(self, other):
        """
        Prefill from another KV cache.
        Used for multi-sample generation from single prefill.
        """
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"

        # Validate shapes
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:  # num_layers, k/v, num_heads, head_dim
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:  # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:  # seq_len
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"

        # Initialize and copy
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        """
        Insert keys and values into cache for given layer.

        Args:
            layer_idx: Index of transformer layer
            k: Keys tensor (B, H, T, D)
            v: Values tensor (B, H, T, D)

        Returns:
            Tuple of (full_keys, full_values) up to current position
        """
        # Lazy initialize
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)

        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add

        # Dynamically grow cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024
            t_needed = (t_needed + 1023) & ~1023  # round up to nearest 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape

        # Insert into cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v

        # Return views
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]

        # Update position after last layer
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1

        return key_view, value_view


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    Sample next token from logits.

    Args:
        logits: Logits tensor (B, vocab_size)
        rng: Random number generator
        temperature: Sampling temperature (0.0 = greedy)
        top_k: Top-k sampling parameter

    Returns:
        Sampled token IDs (B, 1)
    """
    assert temperature >= 0.0, "temperature must be non-negative"

    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


class Engine:
    """
    Efficient inference engine with KV caching for text generation.
    """

    def __init__(self, model, tokenizer):
        """
        Args:
            model: GPT model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Generate tokens autoregressively with KV caching.

        Args:
            tokens: Initial token IDs (list of ints)
            num_samples: Number of samples to generate in parallel
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            seed: Random seed

        Yields:
            Tuple of (token_column, token_masks) for each generation step
            - token_column: List of token IDs (one per sample)
            - token_masks: List of masks (1 = sampled, 0 = forced)
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get special tokens
        bos = self.tokenizer.get_bos_token_id()

        # Prefill with batch size 1
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer
        }
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), **kv_model_kwargs)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)
        sampled_tokens = next_ids[:, 0].tolist()

        # Replicate KV cache for each sample
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, **kv_model_kwargs)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # Track completion for each sample
        completed = [False] * num_samples

        # Main generation loop
        num_generated = 0
        first_iteration = True

        while True:
            # Stop conditions
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(completed):
                break

            # Get sampled tokens
            if first_iteration:
                sampled_tokens = [sampled_tokens[0]] * num_samples
                first_iteration = False
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)
                logits = logits[:, -1, :]
                next_ids = sample_next_token(logits, rng, temperature, top_k)
                sampled_tokens = next_ids[:, 0].tolist()

            # Process each sample
            token_column = []
            token_masks = []

            for i in range(num_samples):
                if completed[i]:
                    token_column.append(bos)  # dummy token
                    token_masks.append(0)
                else:
                    token = sampled_tokens[i]
                    token_column.append(token)
                    token_masks.append(1)  # all tokens are sampled in this simple version

                    # Check for completion (bos token marks end)
                    if token == bos:
                        completed[i] = True

            yield token_column, token_masks
            num_generated += 1

            # Prepare for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation.

        Args:
            tokens: Initial token IDs (list of ints)
            num_samples: Number of samples to generate
            **kwargs: Additional arguments for generate()

        Returns:
            Tuple of (results, masks):
            - results: List of token sequences (list of lists)
            - masks: List of masks for each token (list of lists)
        """
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)

            if all(completed):
                break

        return results, masks
