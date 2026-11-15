"""UDiT blocks with feature input/output and fixed mixture conditioning."""
import torch

from .audio import lengths_to_mask
from .backbone.udit import UDiT


class MelUDiT(UDiT):
    def __init__(self, mel_dim=100, hidden_size=1024, depth=16, num_heads=16,
                 mlp_ratio=4.0, use_checkpoint=True, conditioning_mode="channel_concat",
                 attention_mode="bidirectional", skip_redundant_mask=True,
                 vectorized_packing=True, compile_blocks=False, compile_mode="default",
                 compile_batch_size=None, feature_dim=None):
        if depth < 2 or depth % 2:
            raise ValueError("UDiT depth must be an even number >=2; actual blocks = depth+1")
        if conditioning_mode not in {"channel_concat", "time_concat", "flowtse"}:
            raise ValueError("Unknown conditioning_mode")
        if attention_mode not in {"bidirectional", "enrollment_readonly"}:
            raise ValueError("Unknown attention_mode")
        if compile_mode not in {"default", "max-autotune-no-cudagraphs"}:
            raise ValueError("Unknown compile_mode")
        channels = mel_dim if feature_dim is None else feature_dim
        if type(channels) is not int or channels < 1:
            raise ValueError('Feature dimension must be a positive integer')
        super().__init__(input_dim=channels, output_dim=channels, hidden_size=hidden_size,
                         depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         pos_method="none", use_checkpoint=use_checkpoint,
                         mixture_condition_mode=("time_prefix" if conditioning_mode == "time_concat"
                                                 else "channel_concat"))
        self.conditioning_mode = conditioning_mode
        self.attention_mode = attention_mode
        self.skip_redundant_mask = skip_redundant_mask
        self.vectorized_packing = vectorized_packing
        self.compile_blocks = compile_blocks
        self.compile_mode = compile_mode
        if compile_batch_size is not None and (type(compile_batch_size) is not int or compile_batch_size < 1):
            raise ValueError("compile_batch_size must be a positive integer or None")
        self.compile_batch_size = compile_batch_size
        self._compile_rope_signature = None
        self.last_packing_path = None
        if compile_blocks and use_checkpoint:
            raise ValueError("Block compilation currently requires use_checkpoint=False")
        if conditioning_mode == "flowtse":
            # Shared [state, condition] projection; enrollment is [0, E].
            self.enrollment_proj = None
        # Instantaneous flow matching uses a single time coordinate.
        del self.delta_embedder

    def forward(self, state, time, enrollment, mixture, mixture_lengths, enrollment_lengths):
        if (self.vectorized_packing and state.ndim == 3 and state.shape == mixture.shape
                and state.shape[1] == self.input_dim and enrollment.shape[:2] == state.shape[:2]
                and state.shape[-1] > 0 and enrollment.shape[-1] > 0
                and mixture_lengths.shape == (state.shape[0],)
                and enrollment_lengths.shape == (state.shape[0],)):
            # Two batched length reads replace per-example GPU scalar reads.
            # Only physically full input tensors qualify; ragged tensors retain
            # compact packing and their original masks, including NaN padding.
            ml = mixture_lengths.tolist()
            el = enrollment_lengths.tolist()
            if (all(n == state.shape[-1] for n in ml)
                    and all(n == enrollment.shape[-1] for n in el)):
                self.last_packing_path = "vectorized"
                return self._forward_uniform(state, time, enrollment, mixture)
        self.last_packing_path = "reference"
        return self._forward_reference(state, time, enrollment, mixture,
                                       mixture_lengths, enrollment_lengths)

    def _forward_uniform(self, state, time, enrollment, mixture):
        if self.conditioning_mode == "time_concat":
            enroll_tokens = self.input_proj(enrollment.transpose(1, 2))
            mix_tokens = self.input_proj(mixture.transpose(1, 2))
            state_tokens = self.input_proj(state.transpose(1, 2))
            tokens = torch.cat((enroll_tokens, mix_tokens, state_tokens), dim=1)
            offset = enrollment.shape[-1] + mixture.shape[-1]
        else:
            state_tokens = self.input_proj(torch.cat((state, mixture), dim=1).transpose(1, 2))
            enroll_tokens = (self.input_proj(torch.cat((torch.zeros_like(enrollment), enrollment), dim=1).transpose(1, 2))
                             if self.conditioning_mode == "flowtse"
                             else self.enrollment_proj(enrollment.transpose(1, 2)))
            tokens = torch.cat((enroll_tokens, state_tokens), dim=1)
            offset = enrollment.shape[-1]
        token_mask = None
        if not self.skip_redundant_mask or self.attention_mode == "enrollment_readonly":
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        attention_mask = None
        if self.attention_mode == "enrollment_readonly":
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
            is_enrollment = (positions < enrollment.shape[-1]).expand(tokens.shape[0], -1)
            attention_mask = ~is_enrollment[:, :, None] | is_enrollment[:, None, :]
        tokens = self._transformer(tokens, time, token_mask, attention_mask,
                                   allow_masked_compile=True)
        return tokens[:, offset:offset + state.shape[-1]].transpose(1, 2).contiguous()

    def _forward_reference(self, state, time, enrollment, mixture, mixture_lengths, enrollment_lengths):
        if state.shape != mixture.shape or state.ndim != 3 or state.shape[1] != self.input_dim:
            raise ValueError("State and fixed mixture must have matching [B,C,T] shapes")
        if enrollment.shape[:2] != state.shape[:2]:
            raise ValueError("Enrollment must match batch and feature dimensions")
        mixture_lengths = mixture_lengths.to(state.device, torch.long)
        enrollment_lengths = enrollment_lengths.to(state.device, torch.long)
        for lengths, maximum in [(mixture_lengths, state.shape[-1]), (enrollment_lengths, enrollment.shape[-1])]:
            if lengths.shape != (state.shape[0],) or bool(((lengths < 1) | (lengths > maximum)).any()):
                raise ValueError("Invalid token lengths")
        mix_mask = lengths_to_mask(mixture_lengths, state.shape[-1])
        enroll_mask = lengths_to_mask(enrollment_lengths, enrollment.shape[-1])
        state = state.masked_fill(~mix_mask[:, None], 0)
        mixture = mixture.masked_fill(~mix_mask[:, None], 0)
        enrollment = enrollment.masked_fill(~enroll_mask[:, None], 0)
        if self.conditioning_mode != "time_concat":
            state_tokens = self.input_proj(torch.cat((state, mixture), dim=1).transpose(1, 2))
        if self.conditioning_mode == "time_concat":
            state_tokens = self.input_proj(state.transpose(1, 2))
            mix_tokens = self.input_proj(mixture.transpose(1, 2))
            enroll_tokens = self.input_proj(enrollment.transpose(1, 2))
            state_offsets = enrollment_lengths + mixture_lengths
        elif self.conditioning_mode == "flowtse":
            enroll_tokens = self.input_proj(
                torch.cat((torch.zeros_like(enrollment), enrollment), dim=1).transpose(1, 2))
            state_offsets = enrollment_lengths
        else:
            enroll_tokens = self.enrollment_proj(enrollment.transpose(1, 2))
            state_offsets = enrollment_lengths
        total_lengths = state_offsets + mixture_lengths
        total_sizes = total_lengths.tolist()
        max_total = max(total_sizes)
        tokens = state_tokens.new_zeros((state.shape[0], max_total, state_tokens.shape[-1]))
        # Pack each example's real segments contiguously: padding must not shift
        # the state's rotary positions relative to single-item use.
        for index, (el, ml) in enumerate(zip(enrollment_lengths.tolist(), mixture_lengths.tolist())):
            tokens[index, :el] = enroll_tokens[index, :el]
            offset = int(state_offsets[index])
            if self.conditioning_mode == "time_concat":
                tokens[index, el:offset] = mix_tokens[index, :ml]
            tokens[index, offset:offset + ml] = state_tokens[index, :ml]
        token_mask = lengths_to_mask(total_lengths, tokens.shape[1])
        if (self.skip_redundant_mask and self.attention_mode == "bidirectional"
                and all(size == max_total for size in total_sizes)):
            # Packed sequences have no gaps. With no padded keys/queries, the
            # all-True padding mask and per-block padding fills are redundant.
            # Keep the explicit mask path for real padding and for FlowTSE's
            # directional policy. SDPA may select FlashAttention on this path.
            token_mask = None
        attention_mask = None
        if self.attention_mode == "enrollment_readonly":
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
            is_enrollment = positions < enrollment_lengths[:, None]
            # True means visible. Enrollment queries see only enrollment keys;
            # all remaining queries see every valid key, without a causal mask.
            # Only this mode allocates a dense pairwise boolean mask.
            attention_mask = ~is_enrollment[:, :, None] | is_enrollment[:, None, :]
        tokens = self._transformer(tokens, time, token_mask, attention_mask)
        output = tokens.new_zeros(state.shape)
        for index, (offset, ml) in enumerate(zip(state_offsets.tolist(), mixture_lengths.tolist())):
            output[index, :, :ml] = tokens[index, offset:offset + ml].transpose(0, 1)
        return output

    def _transformer(self, tokens, time, token_mask, attention_mask, allow_masked_compile=False):
        # Uniform tensors have stable mask shapes. Compile the original masked
        # computation too; ragged packing and evaluation retain eager execution.
        compile_masked = self.compile_blocks and allow_masked_compile
        # Preserve every partial batch but avoid paying a second static-shape
        # autotune pass for its rare size on an otherwise full-batch epoch.
        allow_compile = self.compile_batch_size is None or tokens.shape[0] == self.compile_batch_size
        if (self.compile_blocks and allow_compile and self.training and torch.is_grad_enabled()
                and (compile_masked or (token_mask is None and attention_mask is None))):
            blocks = list(self.in_blocks) + [self.mid_block] + list(self.out_blocks)
            signature = (tokens.shape[1], tokens.device)
            invalid_cache = any(
                block.attn.rotary._seq_len_cached != tokens.shape[1]
                or block.attn.rotary._cos_cached is None
                or block.attn.rotary._cos_cached.device != tokens.device
                or block.attn.rotary._cos_cached.dtype != torch.float32
                for block in blocks)
            if self._compile_rope_signature != signature or invalid_cache:
                # Keep rotary-table cache mutation outside the compiled region.
                shape = (1, 1, tokens.shape[1], tokens.shape[-1] // self.num_heads)
                dummy = tokens.new_empty(shape, dtype=torch.float32)
                for block in blocks:
                    block.attn.rotary._update_cos_sin_tables(dummy)
                self._compile_rope_signature = signature
            for block in blocks:
                if block._compiled_forward is None:
                    block._compiled_forward = torch.compile(
                        block._forward, fullgraph=True, dynamic=False, mode=self.compile_mode)
        condition = self.t_embedder(time * 1000)
        skips = []
        for block in self.in_blocks:
            tokens = block(tokens, condition, token_mask=token_mask, attention_mask=attention_mask,
                           compile_masked=compile_masked, allow_compile=allow_compile)
            skips.append(tokens)
        tokens = self.mid_block(tokens, condition, token_mask=token_mask, attention_mask=attention_mask,
                                compile_masked=compile_masked, allow_compile=allow_compile)
        for block in self.out_blocks:
            tokens = block(tokens, condition, skips.pop(), token_mask=token_mask, attention_mask=attention_mask,
                           compile_masked=compile_masked, allow_compile=allow_compile)
        tokens = self.final_layer(tokens, condition)
        return tokens

