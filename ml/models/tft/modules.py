import torch
import torch.nn as nn

from typing import Tuple, List, Optional

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size : int, hidden_layer_size : int, dropout_rate : float, activation : nn.Module = nn.Identity()) -> None:
        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation = activation

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.W4 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = nn.Linear(self.input_size, self.hidden_layer_size)

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.dropout_rate:
            x = self.dropout(x)
        
        out = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
     
        return out

class GatedAddNormNetwork(nn.Module):
    def __init__(self, input_size : int, hidden_layer_size : int, dropout_rate : float, activation : nn.Module = nn.Identity()) -> None:
        super(GatedAddNormNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.glu = GatedLinearUnit(self.input_size, self.hidden_layer_size, self.dropout_rate, self.activation)

        self.layer_norm = nn.LayerNorm(self.hidden_layer_size)

    def forward(self, x : torch.Tensor, skip : torch.Tensor) -> torch.Tensor:
        out = self.layer_norm(self.glu(x) + skip)

        return out

class GatedResidualNetwork(nn.Module):
    def __init__(self, hidden_layer_size : int, input_size : int, output_size : Optional[int] = None, additional_context : Optional[int] = None, dropout_rate : float = 0.5, return_gate : bool = False) -> None:
        super(GatedResidualNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context
        self.return_gate = return_gate

        self.W1 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.W2 = nn.Linear(self.input_size, self.hidden_layer_size)

        if self.additional_context:
            self.W3 = nn.Linear(self.additional_context, self.hidden_layer_size, bias=False)

        if self.output_size:
            self.skip_linear = nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GatedAddNormNetwork(self.hidden_layer_size, self.output_size, self.dropout_rate)
        else:
            self.glu_add_norm = GatedAddNormNetwork(self.hidden_layer_size, self.hidden_layer_size, self.dropout_rate)

        self.init_weights()

    
    def init_weights(self):
        for name, p in self.named_parameters():
            if ('W2' in name or 'W3' in name) and 'bias' not in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif ('skip_linear' in name or 'W1' in name) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.zeros_(p)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.additional_context:
            x, context = x
            n2 = torch.nn.functional.elu(self.W2(x) + self.W3(context))
        else:
            n2 = torch.nn.functional.elu(self.W2(x))

        n1 = self.W1(n2)

        if self.output_size:
            out = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            out = self.glu_add_norm(n1, x)

        return out

    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout : float = 0.0, scale : bool = True) -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, mask : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.bmm(q, k.permute(0,2,1))

        if self.scale:
            denom = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / denom

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)

        return out, attn

    
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head : int, d_model : int, dropout : float) -> None:
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v, bias=False)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q, bias=False) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_head)]
        )
        self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

        
    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p) 

    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, mask : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []

        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layers[i](v)

            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outs = torch.mean(head, dim=2) if self.n_head > 1 else head

        outs = self.w_h(outs)
        outs = self.dropout(outs)

        return outs, attn


class VariableSelectionNetwork(nn.Module):
    def __init__(self, hidden_layer_size : int, dropout_rate : float, output_size : int, input_size : Optional[int] = None, additional_context : Optional[int] = None) -> None:
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context

        self.flattened_grn = GatedResidualNetwork(self.hidden_layer_size, input_size = self.input_size, output_size = self.output_size, dropout_rate = self.dropout_rate, additional_context = self.additional_context)

        self.per_feature_grn = nn.ModuleList([
            GatedResidualNetwork(self.hidden_layer_size, input_size=self.hidden_layer_size, dropout_rate=self.dropout_rate) for i in range(self.output_size)
        ])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.additional_context:
            embedding, static_context = x

            time_steps = embedding.shape[1]
            flatten = embedding.view(-1, time_steps, self.hidden_layer_size * self.output_size)

            static_context = static_context.unsqueeze(1)

            # Non-linear transformation with gated residual network
            mlp_outputs = self.flattened_grn((flatten, static_context))

            sparse_weights = torch.nn.functional.softmax(mlp_outputs, dim=-1)
            sparse_weights = sparse_weights.unsqueeze(2)

            trans_emb_list = []
            for i in range(self.output_size):
                e = self.per_feature_grn[i](embedding[..., i, :])
                trans_emb_list.append(e)
            transformed_embedding = torch.stack(trans_emb_list, dim=-1)

            combined = sparse_weights * transformed_embedding
            temporal_ctxt = torch.sum(combined, dim=-1)
        else:
            embedding = x

            flatten = torch.flatten(embedding, start_dim=1)

            mlp_outputs = self.flattened_grn(flatten)

            sparse_weights = torch.nn.functional.softmax(mlp_outputs, dim=-1)
            sparse_weights = sparse_weights.unsqueeze(-1)

            trans_emb_list = []
            for i in range(self.output_size):
                e = self.per_feature_grn[i](embedding[:, i:i+1, :])
                trans_emb_list.append(e)
            transformed_embedding = torch.cat(trans_emb_list, dim=1)

            combined = sparse_weights * transformed_embedding
            temporal_ctxt = torch.sum(combined, dim=1)

        return temporal_ctxt, sparse_weights