from numpy import concat, require
import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights

class Engine:
    """
    A class to manage the generation engine.
    """
    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
        self.__name__ = "DifferentPrefill"
        self.weight_path = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
        self.weight_path = "Llama-3.2-1B"
        self.head_dim = 64         # Dimensionality of each attention head
        self.num_qo_heads = 32      # Total number of query/output heads
        self.num_kv_heads = 8       # Total number of key/value heads
        self.layers = 16            # Number of transformer layers

        # Load the tokenizer for text processing
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weight_path,
            local_files_only=True
        )

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)

        self.kv_cache = {}

    def run(self, input_ids_list, prefill=True):
        """
        Run a batched forward pass through the model with KV caching,
        handling variable-length sequences.

        Args:
            input_ids_list: List of 1D tensors, each containing token IDs for one sequence.
                           Sequences can have different lengths.
            prefill: If True, this is the initial prefill phase.
                     If False, this is a decode step using KV cache.

        Returns:
            1D tensor of shape (batch_size,) with the next token for each sequence.
        """
        ### Modified from transformer.py in lecture. ###

        # Unpack weights for convenient reference
        embedding = self.weights["embedding"] # token_ids -> embedding

        # Attention weights, they are per layer. We'll index them by layer number
        layernormAttn_weight = self.weights["layernormAttn_weight"]
        self_attn_q_proj_weight = self.weights["self_attn_q_proj_weight"]
        self_attn_k_proj_weight = self.weights["self_attn_k_proj_weight"]
        self_attn_v_proj_weight = self.weights["self_attn_v_proj_weight"]
        o_proj_weight = self.weights["o_proj_weight"]

        # FFN weights
        layernormFFN_weight = self.weights["layernormFFN_weight"]
        up_proj_weight = self.weights["up_proj_weight"]
        gate_proj_weight = self.weights["gate_proj_weight"]
        down_proj_weight = self.weights["down_proj_weight"]

        # Final layer normalization
        model_layernorm_weight = self.weights["model_layernorm_weight"]

        # Final vocabulary projection. Here, "head" is not the same as head in multi-head attention.
        # "head" often refers to the final layer or component that maps the model's internal
        # hidden representations (embeddings) to the output space
        lm_head_weight = self.weights["lm_head_weight"]

        # Concatenate all variable-length sequences into a flat tensor and embed
        batch_size = len(input_ids_list)
        seq_lens = [len(ids) for ids in input_ids_list]
        all_ids = torch.cat([ids.to(device='cuda') for ids in input_ids_list])
        hidden_state = embedding[all_ids]  # (total_tokens, hidden_dim)

        #####################
        ### BEGIN Prefill ###
        #####################

        # Compute per-sequence position offsets for RoPE
        if prefill:
            # Reset KV cache for new batch of sequences
            self.kv_cache = {}
            position_offsets = [0] * batch_size
        else:
            # Each sequence may have a different number of cached tokens
            position_offsets = [self.kv_cache[0]["k"][b].shape[0] for b in range(batch_size)]

        ###################
        ### END Prefill ###
        ###################

        # Process each transformer layer
        for layer in range(self.layers):
            ### Attention Block ###

            # RMSNorm for each vector of user requests (per-token, works on flat tensor)
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = normalized_x * layernormAttn_weight[layer]

            # QKV projections (per-token, works on flat tensor)
            q = x.matmul(self_attn_q_proj_weight[layer].t())  # (total_tokens, num_qo_heads * head_dim)
            k = x.matmul(self_attn_k_proj_weight[layer].t())  # (total_tokens, num_kv_heads * head_dim)
            v = x.matmul(self_attn_v_proj_weight[layer].t())  # (total_tokens, num_kv_heads * head_dim)

            # Split Q, K, V per sequence for RoPE and attention (different lengths/offsets)
            q_list = list(torch.split(q, seq_lens))
            k_list = list(torch.split(k, seq_lens))
            v_list = list(torch.split(v, seq_lens))

            # RoPE (Rotary Positional Embedding) - per sequence with its own position offset
            for b in range(batch_size):
                apply_rope(q_list[b], output=q_list[b], head_dim=self.head_dim, offset=position_offsets[b])
                apply_rope(k_list[b], output=k_list[b], head_dim=self.head_dim, offset=position_offsets[b])

            ######################
            ### BEGIN KV Cache ###
            ######################

            if prefill:
                # Initialize per-sequence cache for this layer (clone to decouple from flat tensor)
                self.kv_cache[layer] = {
                    "k": [k_b.clone() for k_b in k_list],
                    "v": [v_b.clone() for v_b in v_list]
                }
            else:
                # Append new K, V to each sequence's cache
                for b in range(batch_size):
                    self.kv_cache[layer]["k"][b] = torch.cat([self.kv_cache[layer]["k"][b], k_list[b]], dim=0)
                    self.kv_cache[layer]["v"][b] = torch.cat([self.kv_cache[layer]["v"][b], v_list[b]], dim=0)

            ####################
            ### END KV Cache ###
            ####################

            # Attention computed per sequence (different q_len, kv_len, and causal masks)
            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            attn_outputs = []

            for b in range(batch_size):
                q_b = q_list[b]                                    # (q_len_b, num_qo_heads * head_dim)
                k_cache_b = self.kv_cache[layer]["k"][b]           # (kv_len_b, num_kv_heads * head_dim)
                v_cache_b = self.kv_cache[layer]["v"][b]           # (kv_len_b, num_kv_heads * head_dim)

                # Compute sub-components of q, k, v for each head
                sub_q = q_b.view(-1, self.num_qo_heads, self.head_dim)        # (q_len_b, num_qo_heads, head_dim)
                sub_k = k_cache_b.view(-1, self.num_kv_heads, self.head_dim)  # (kv_len_b, num_kv_heads, head_dim)
                sub_v = v_cache_b.view(-1, self.num_kv_heads, self.head_dim)  # (kv_len_b, num_kv_heads, head_dim)

                # The below is needed to obtain the dimensions of the attention score matrix
                n_q = sub_q.shape[0] # Query sequence length for this sequence
                n_k = sub_k.shape[0] # Key sequence length (full cached length for this sequence)

                # Replicate sub_k and sub_v for each group of query heads (GQA)
                sub_k = sub_k.repeat_interleave(group_size, dim=1)
                sub_v = sub_v.repeat_interleave(group_size, dim=1)

                # Rearrange to (num_qo_heads, seq_len, head_dim)
                sub_q_t = sub_q.permute(1, 0, 2)  # (num_qo_heads, q_len_b, head_dim)
                sub_k_t = sub_k.permute(1, 0, 2)  # (num_qo_heads, kv_len_b, head_dim)

                # Compute attention scores: (num_qo_heads, q_len_b, kv_len_b)
                scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale

                # Create causal mask for this sequence
                causal_mask = torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)

                ###################################
                ### START KV Cache Prefill Mask ###
                ###################################

                # Only attend to positions 0 to abs_pos (inclusive)
                for i in range(n_q):
                    abs_pos = position_offsets[b] + i # Absolute position for this sequence
                    causal_mask[i, abs_pos + 1:] = False # Invert later (still)

                #################################
                ### END KV Cache Prefill Mask ###
                #################################

                # https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                # Apply softmax along the last dimension (for each query against all keys) to get attention weights
                attn_weights = torch.softmax(scores, dim=-1)  # (num_qo_heads, q_len_b, kv_len_b)

                # Compute attention output by multiplying weights with the values
                v_t = sub_v.permute(1, 0, 2)  # (num_qo_heads, kv_len_b, head_dim)
                attn_out = torch.matmul(attn_weights, v_t)  # (num_qo_heads, q_len_b, head_dim)

                # Go back to single-head from multi-head attention. We combine the outputs from all heads
                attn_out = attn_out.permute(1, 0, 2)
                attn_out = attn_out.reshape(-1, self.num_qo_heads * self.head_dim)  # (q_len_b, hidden_dim)
                attn_outputs.append(attn_out)

            # Concatenate attention outputs back to flat tensor
            attn_output = torch.cat(attn_outputs, dim=0)  # (total_tokens, hidden_dim)

            # Output projection and residual connection
            o_proj_residual = attn_output.matmul(o_proj_weight[layer].t()) + hidden_state

            # --- Feed-Forward Network (FFN) ---

            # RMSNorm before FFN (per-token, works on flat tensor)
            rms = torch.sqrt(torch.mean(o_proj_residual ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = o_proj_residual / rms
            layernormFFN_output = normalized_x.to(torch.float16) * layernormFFN_weight[layer]

            # Up projection and gate projection
            up_proj_output = layernormFFN_output.matmul(up_proj_weight[layer].t())
            gate_proj_output = layernormFFN_output.matmul(gate_proj_weight[layer].t())

            # SwiGLU activation
            activation_output = up_proj_output * torch.nn.functional.silu(gate_proj_output)

            # Down projection with residual connection
            down_proj_output = activation_output.matmul(down_proj_weight[layer].t())
            hidden_state = down_proj_output + o_proj_residual

        # --- Final Layer Norm and Projection ---

        # Final RMSNorm
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        model_output = normalized_x.to(torch.float16) * model_layernorm_weight

        # Project to vocabulary
        logits = model_output.matmul(lm_head_weight.t())  # (total_tokens, vocab_size)

        # Extract the last token of each sequence from the flat tensor
        cumsum = 0
        last_indices = []
        for s in seq_lens:
            last_indices.append(cumsum + s - 1)
            cumsum += s

        next_tokens = torch.argmax(logits[last_indices], dim=-1)  # (batch_size,)

        return next_tokens.cpu()

    def generate_batched(self, input_string, rounds=20):
        input_ids_list = []
        for input_string in input_string:
            input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids[0]
            input_ids_list.append(input_ids)

        output_ids_list = input_ids_list
        new_token = self.run(input_ids_list)
        for i in range(len(input_ids_list)):
            output_ids_list[i] = torch.cat((output_ids_list[i], new_token[i:i+1]), dim=0)

        for round in range(rounds - 1):
            print(f"Round {round}")
            input_ids_list = []
            for output_ids in output_ids_list:
                input_ids_list.append(output_ids[-1:])
            new_token = self.run(input_ids_list, prefill=False)

            for i in range(len(input_ids_list)):
                output_ids_list[i] = torch.cat((output_ids_list[i], new_token[i:i+1]), dim=0)
        output_text_list = []
        for output_ids in output_ids_list:
            output_text_list.append(self.tokenizer.decode(output_ids, skip_special_tokens=True))
        return output_text_list

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "The University of Washington is located in"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)
