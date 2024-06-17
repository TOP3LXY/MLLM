import torch
import math
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig



class LlamaConfig(PretrainedConfig):
    def __init__(self, 
        hidden_size=4096,
        num_heads=32,
        num_hidden_layer=32,
        attention_bias=False,
        intermediate_size=11008,    
        mlp_bias=False, 
        vocab_size=32000, 
        rsm_norm_eps=1e-6,  
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_bias = attention_bias
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.rsm_norm_eps = rsm_norm_eps
        self.vocab_size = vocab_size
        self.num_hidden_layer = num_hidden_layer

        super().__init__()


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(hidden_size)
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keep_dim=True)
        out = x*torch.rsqrt(variance+self.variance_epsilon)*self.weight
        out = out.to(input_dtype)

        return out



class LlamaAttention(nn.Module):
    def __init__(self, config:LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size//self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias=config.attention_bias)
        
    def forward(self, hidden_states):
        batch, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch, q_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch, q_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch, q_len, self.num_heads, self.head_dim).transpose(1,2)

        attn_weights = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.head_dim)
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        output_states = torch.matmul(attn_weights, v)

        output_states = output_states.transpose(1,2).contiguous()
        output_states = output_states.view(batch, q_len, -1)

        return output_states, attn_weights





class LlamaMLP(nn.Module):
    def __init__(self, config:LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaEncoderLayer(nn.Module):
    def __init__(self, config:LlamaConfig) -> None:
        super().__init__()
  
        self.attan = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_lnorm = LlamaRMSNorm(config.hidden_size, eps=config.rsm_norm_eps)
        self.post_attn_lnorm = LlamaRMSNorm(config.hidden_size, eps=config.rsm_norm_eps)

    def forward(self, hidden_states):
        #res_attan
        residual = hidden_states
        hidden_states = self.input_lnorm(hidden_states)
        hidden_states, attn_score = self.attan(hidden_states)
        hidden_states += residual

        #res_ff
        residual = hidden_states
        hidden_states = self.post_attn_lnorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states



class LlamaModel(PreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.id_emb = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LlamaEncoderLayer(config) for _ in range(config.num_hidden_layer)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rsm_norm_eps)

    def forward(self, id):
        hidden_states = self.id_emb(id)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        out_states = self.norm(hidden_states)

        return out_states
            