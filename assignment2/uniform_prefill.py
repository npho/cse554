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
        self.__name__ = "UniformPrefill"
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

    def run(self, input_ids, prefill=True):
        """
        Run a batched forward pass through the model with KV caching.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            prefill: If True, this is the initial prefill phase.
                     If False, this is a decode step using KV cache.

        Returns:
            Tensor of shape (batch_size, 1) with the next token for each sequence.
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
        hidden_state = embedding[input_tensor]  # (batch, seq_len, hidden_dim)
        batch_size = hidden_state.shape[0]

        #####################
        ### BEGIN Prefill ###
        #####################

        # Compute position offset for RoPE
        if prefill:
            # Reset KV cache for new batch of sequences
            self.kv_cache = {}
            position_offset = 0
        else:
            # Get offset from existing cache (number of cached tokens per sequence)
            position_offset = self.kv_cache[0]["k"].shape[1]

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

            # QKV projections (matmul broadcasts over batch dim)
            q = x.matmul(self_attn_q_proj_weight[layer].t())  # (batch, seq_len, num_qo_heads * head_dim)
            k = x.matmul(self_attn_k_proj_weight[layer].t())  # (batch, seq_len, num_kv_heads * head_dim)
            v = x.matmul(self_attn_v_proj_weight[layer].t())  # (batch, seq_len, num_kv_heads * head_dim)

            # RoPE (Rotary Positional Embedding) - apply per batch element since apply_rope expects 2D
            # All sequences share the same position_offset (uniform prefill lengths)
            for b in range(batch_size):
                apply_rope(q[b], output=q[b], head_dim=self.head_dim, offset=position_offset)
                apply_rope(k[b], output=k[b], head_dim=self.head_dim, offset=position_offset)

            ######################
            ### BEGIN KV Cache ###
            ######################

            if prefill:
                # Initialize cache for this layer
                self.kv_cache[layer] = {"k": k, "v": v}
                k_cache = k
                v_cache = v
            else:
                # Append new K, V to cache along the sequence dimension (dim=1)
                k_cache = torch.cat([self.kv_cache[layer]["k"], k], dim=1)
                v_cache = torch.cat([self.kv_cache[layer]["v"], v], dim=1)
                self.kv_cache[layer] = {"k": k_cache, "v": v_cache}

            ####################
            ### END KV Cache ###
            ####################

            # Compute sub-components of q, k, v for each head
            # I: (batch, seq_len, num_qo_heads * head_dim)
            sub_q = q.view(batch_size, -1, self.num_qo_heads, self.head_dim)        # (batch, q_len, num_qo_heads, head_dim)
            sub_k = k_cache.view(batch_size, -1, self.num_kv_heads, self.head_dim)  # (batch, kv_len, num_kv_heads, head_dim)
            sub_v = v_cache.view(batch_size, -1, self.num_kv_heads, self.head_dim)  # (batch, kv_len, num_kv_heads, head_dim)

            # Compute some attention-related values
            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            # The sequence length for q and k is needed to compute the causal mask [slide]
            # The below is needed to obtain the dimensions of the attention score matrix
            n_q = sub_q.shape[1] # Query sequence length
            n_k = sub_k.shape[1] # Key sequence length (full cached length)

            # Replicate sub_k and sub_v for each group of query heads
            # I: (batch, kv_len, num_kv_heads, head_dim)
            # O: (batch, kv_len, num_qo_heads, head_dim)
            sub_k = sub_k.repeat_interleave(group_size, dim=2)
            sub_v = sub_v.repeat_interleave(group_size, dim=2)

            # Rearrange to (batch, num_qo_heads, seq_len, head_dim) for batched attention
            sub_q_t = sub_q.permute(0, 2, 1, 3)  # (batch, num_qo_heads, q_len, head_dim)
            sub_k_t = sub_k.permute(0, 2, 1, 3)  # (batch, num_qo_heads, kv_len, head_dim)

            # Compute attention scores: (batch, num_qo_heads, q_len, kv_len)
            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale

            # Create causal mask
            # Same for all batch elements since all sequences have the same length (uniform prefill)
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
            # Unsqueeze twice: once for batch dim, once for heads dim -> (1, 1, q_len, kv_len)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Apply softmax along the last dimension (for each query against all keys) to get attention weights
            attn_weights = torch.softmax(scores, dim=-1)  # (batch, num_qo_heads, q_len, kv_len)

            # Compute attention output by multiplying weights with the values
            v_t = sub_v.permute(0, 2, 1, 3)  # (batch, num_qo_heads, kv_len, head_dim)
            attn_output = torch.matmul(attn_weights, v_t)  # (batch, num_qo_heads, q_len, head_dim)

            # Go back to single-head from multi-head attention. We combine the outputs from all heads
            attn_output = attn_output.permute(0, 2, 1, 3)  # (batch, q_len, num_qo_heads, head_dim)
            attn_output = attn_output.reshape(batch_size, -1, self.num_qo_heads * self.head_dim)

            # Output projection and residual connection
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
        logits = model_output.matmul(lm_head_weight.t())  # (batch, seq_len, vocab_size)

        # Get the next token for each batch element (argmax of the last position)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # (batch,)

        return next_token.unsqueeze(1).cpu()  # (batch, 1)

    def generate_batched(self, input_string, rounds=20):
        input_ids_list = self.tokenizer(input_string, return_tensors="pt", padding=False).input_ids
        print("Input String:", input_string)

        print("Token IDs:", input_ids_list)
        output_ids_list = input_ids_list

        new_token = self.run(output_ids_list)
        print("New Token Shape:", new_token.shape)
        output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids_list[:, -1:], prefill=False)
            output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        output_text = self.tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)
        return output_text

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "Hi, how are you?"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)
