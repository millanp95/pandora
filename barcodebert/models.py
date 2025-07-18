import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForTokenClassification

class FastBertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # ————— same param checks —————
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of "
                f"the number of attention heads ({config.num_attention_heads})"
            )
        self.num_attention_heads   = config.num_attention_heads
        self.attention_head_size   = config.hidden_size // config.num_attention_heads
        self.all_head_size         = self.num_attention_heads * self.attention_head_size

        # ————— Q/K/V projections —————
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # ————— dropout & settings —————
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = (
            position_embedding_type
            or getattr(config, "position_embedding_type", "absolute")
        )
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1,
                self.attention_head_size,
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) → (B, H, L, head_size)
        B, L, _ = x.size()
        x = x.view(B, L, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask:       torch.FloatTensor = None,
        head_mask:            torch.FloatTensor = None,
        encoder_hidden_states:torch.FloatTensor = None,
        encoder_attention_mask:torch.FloatTensor = None,
        past_key_value:       tuple = None,
        output_attentions:    bool  = False,
    ):
        # 1) project Q
        mixed_query = self.query(hidden_states)

        # 2) handle cross-attention / caching for K, V
        is_cross = encoder_hidden_states is not None
        if is_cross and past_key_value is not None:
            # reuse cross-attn K/V
            key_layer, value_layer = past_key_value
            attention_mask = encoder_attention_mask
        elif is_cross:
            key_layer   = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            # decoder self-attn caching
            prev_k, prev_v = past_key_value
            new_k = self.transpose_for_scores(self.key(hidden_states))
            new_v = self.transpose_for_scores(self.value(hidden_states))
            key_layer   = torch.cat([prev_k, new_k], dim=2)
            value_layer = torch.cat([prev_v, new_v], dim=2)
        else:
            key_layer   = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 3) always project Q last
        query_layer = self.transpose_for_scores(mixed_query)

        # 4) save new cache if decoder
        present_key_value = (key_layer, value_layer) if self.is_decoder else None

        # 5) compute raw scores via fused kernel
        #    scaled_dot_product_attention handles scaling, masking, dropout
        dp = self.dropout.p if self.training else 0.0
        context = F.scaled_dot_product_attention(
            query_layer,  # (B,H,L,dh)
            key_layer,   
            value_layer,
            attn_mask = attention_mask,        # HF’s [B,1,1,L] additive mask
            dropout_p = dp,
            is_causal = False,
        )  # → (B, H, L, dh)

        # 6) relative positions (if any)
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            B, H, L, dh = query_layer.size()
            # build distance matrix
            positions_l = torch.arange(L, device=hidden_states.device).view(-1,1)
            positions_r = torch.arange(key_layer.size(2), device=hidden_states.device).view(1,-1)
            distances = positions_l - positions_r
            distances = distances + self.max_position_embeddings - 1
            rel_emb = self.distance_embedding(distances)   # (L, L, dh)
            rel_emb = rel_emb.to(query_layer.dtype)

            # einsum to add relative scores
            if self.position_embedding_type == "relative_key":
                rel_scores = torch.einsum("bhld,lrd->bhlr", query_layer, rel_emb)
                # need to rerun softmax? approximate by adding before?
                # fallback to default: simply add after
                context = context + rel_scores.unsqueeze(-1)  
            else:  # relative_key_query
                qr = torch.einsum("bhld,lrd->bhlr", query_layer, rel_emb)
                kr = torch.einsum("bhrd,lrd->bhlr", key_layer,     rel_emb)
                context = context + (qr+kr).unsqueeze(-1)
                
        if self.training:
            context = self.dropout(context)

        # 7) reshape back to (B, L, D)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(hidden_states.size(0), -1, self.all_head_size)

        # 8) return exactly HF expects
        outputs = (context, )
        if output_attentions:
            # HF normally returns (context, attn_probs)
            outputs = outputs + (None,)
        if present_key_value is not None:
            outputs = outputs + (present_key_value,)
        return outputs


class MyBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.bert.encoder.layer:
            layer.attention.self = FastBertSelfAttention(config)

