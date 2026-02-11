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
        self.__name__ = "SingleBatch"
        self.weight_path = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
        self.weight_path = "Llama-3.2-1B"
        self.head_dim = 64          # Dimensionality of each attention head
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

    def run(self, input_ids, prefill=True):
        """
        Run a forward pass through the model.

        Args:
            input_ids: List of token IDs to process
            prefill: If True, this is the initial prefill phase (process all tokens).
                     If False, this is a decode step (process single new token using KV cache).

        Returns:
            The next token predicted by the model.
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

        # Convert input_ids to tensor and get embeddings
        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        hidden_state = embedding[input_tensor]  # (seq_len, hidden_dim)

        #####################
        ### BEGIN Prefill ###
        #####################

        # Compute position offset for RoPE
        if prefill:
            # Reset KV cache for new sequence
            self.kv_cache = {}
            position_offset = 0
        else:
            # Get offset from existing cache (number of cached tokens)
            position_offset = self.kv_cache[0]["k"].shape[0]

        ###################
        ### END Prefill ###
        ###################

        # Process each transformer layer
        for layer in range(self.layers):
            ### Attention Block ###

            # RMSNorm for each vector of user requests
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = normalized_x * layernormAttn_weight[layer]

            # QKV projections
            q = x.matmul(self_attn_q_proj_weight[layer].t())  # (seq_len, num_qo_heads * head_dim)
            k = x.matmul(self_attn_k_proj_weight[layer].t())  # (seq_len, num_kv_heads * head_dim)
            v = x.matmul(self_attn_v_proj_weight[layer].t())  # (seq_len, num_kv_heads * head_dim)

            # RoPE (Rotary Positional Embedding)
            apply_rope(q, output=q, head_dim=self.head_dim, offset=position_offset)
            apply_rope(k, output=k, head_dim=self.head_dim, offset=position_offset)

            ######################
            ### BEGIN KV Cache ###
            ######################

            if prefill:
                # Initialize cache for this layer
                self.kv_cache[layer] = {"k": k, "v": v}
                k_cache = k
                v_cache = v
            else:
                # Append new K, V to cache
                k_cache = torch.cat([self.kv_cache[layer]["k"], k], dim=0)
                v_cache = torch.cat([self.kv_cache[layer]["v"], v], dim=0)
                self.kv_cache[layer] = {"k": k_cache, "v": v_cache}

            ####################
            ### END KV Cache ###
            ####################

            # Compute sub-components of q, k, v for each head
            # I: (seq_len, num_qo_heads * head_dim)
            # -1 allows torch to infer the first dimension (seq_len)
            sub_q = q.view(-1, self.num_qo_heads, self.head_dim)        # (seq_len, num_qo_heads, head_dim)
            sub_k = k_cache.view(-1, self.num_kv_heads, self.head_dim)  # (seq_len, num_kv_heads, head_dim)
            sub_v = v_cache.view(-1, self.num_kv_heads, self.head_dim)  # (seq_len, num_kv_heads, head_dim)

            # Compute some attention-related values
            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            # The sequence length for q and k is needed to compute the causal mask [slide]
            # The below is needed to obtain the dimensions of the attention score matrix
            n_q = sub_q.shape[0] # seq_len: Query sequence length
            n_k = sub_k.shape[0] # seq_len: Key sequence length (full cached length)

            # Replicate sub_k and sub_v for each group of query heads
            # The underlying KV values are shared, but each query will attend to them differently
            # I: (seq_len, num_kv_heads, head_dim): (seq_len, 8, 128)
            # O: (seq_len, num_qo_heads, head_dim): (seq_len, 32, 128) # num_qo_heads = num_kv_heads * group_size
            sub_k = sub_k.repeat_interleave(group_size, dim=1)
            sub_v = sub_v.repeat_interleave(group_size, dim=1)

            # Rearrange q and k so the shapes are (num_qo_heads, seq_len, head_dim):
            # This is because the computation of attention scores requires taking the dot product between
            # each query vector and each key vector across the sequence dimension. Specifically, for
            # multi-head attention, the operation must independently occur across each attention head.
            sub_q_t = sub_q.permute(1, 0, 2)  # (num_qo_heads, q_len, head_dim)
            sub_k_t = sub_k.permute(1, 0, 2)  # (num_qo_heads, kv_len, head_dim)

            # Compute attention scores: (num_qo_heads, q_len, kv_len)
            # I_1/q: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
            # I_2/k: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
            # O: (num_qo_heads, seq_len, seq_len): (32, seq_len, seq_len)
            # We take the transpose of the keys to align the dimensions for matrix multiplication
            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale

            # Create causal mask
            # For decode: query only attends to all previous keys (including itself)
            # Shape: (q_len, kv_len) where each query position can attend to positions <= its absolute position
            causal_mask = torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)

            ###################################
            ### START KV Cache Prefill Mask ###
            ###################################

            # Only attend to positions 0 to abs_pos (inclusive)
            for i in range(n_q):
                abs_pos = position_offset + i # Absolute position, inclusive of prefill tokens
                causal_mask[i, abs_pos + 1:] = False # Invert later (still)

            #################################
            ### END KV Cache Prefill Mask ###
            #################################

            # https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            # Apply softmax along the last dimension (for each query against all keys) to get attention weights
            # (num_qo_heads, seq_len, seq_len) corresponds to (heads, queries, keys)
            attn_weights = torch.softmax(scores, dim=-1)  # (num_qo_heads, q_len, kv_len)

            # Compute attention output by multiplying weights with the values. sub_v has shape (seq_len, num_qo_heads, head_dim)
            # Transpose the sub_v tensor to get the shape (num_qo_heads, seq_len, head_dim)
            v_t = sub_v.permute(1, 0, 2)  # (num_qo_heads, kv_len, head_dim)

            # I_1/w: (num_qo_heads, seq_len, seq_len): (32, seq_len, seq_len)
            # I_2/v: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
            # O: (num_qo_heads, seq_len, head_dim): (32, seq_len, 128)
            attn_output = torch.matmul(attn_weights, v_t)  # (num_qo_heads, q_len, head_dim)

            # Go back to single-head from multi-head attention. We combine the outputs from all heads
            attn_output = attn_output.permute(1, 0, 2)
            attn_output = attn_output.reshape(-1, self.num_qo_heads * self.head_dim)

            # Output projection and residual connection. There is a residual connection in the attention block
            # So we add the original input to output projection. Seminal paper: https://arxiv.org/abs/1512.03385
            # I_1/attn: (seq_len, hidden_dim): (seq_len, 4096)
            # I_2/w: (hidden_dim, hidden_dim): (4096, 4096)
            # O: (seq_len, hidden_dim): (seq_len, 4096)
            o_proj_residual = attn_output.matmul(o_proj_weight[layer].t()) + hidden_state

            # --- Feed-Forward Network (FFN) ---

            # RMSNorm before FFN
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
        logits = model_output.matmul(lm_head_weight.t())  # (seq_len, vocab_size)

        # Get the next token (argmax of the last position)
        next_token = torch.argmax(logits, dim=1)[-1].item()

        return next_token

    def generate(self, input_string, rounds=20):
        input_ids = self.tokenizer.encode(input_string)

        print("Token IDs:", input_ids)
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)

        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

    def bench(self, input_string, rounds=20):
        """
        Basically the same as generate() but modified for benchmarking as to not break any potential TA testing.
        """
        input_ids = self.tokenizer.encode(input_string)

        #print("Token IDs:", input_ids)
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(rounds - 1):
            #print(f"Round {round}")
            new_token = self.run(output_ids[-1:], prefill=True)
            output_ids.append(new_token)

        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return input_ids, output_text

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    engine = Engine()
    output_text = engine.generate(input_string, rounds=20)
    print("Generated Text:", output_text)
