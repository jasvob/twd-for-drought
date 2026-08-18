import torch
import torch.nn as nn

from typing import Tuple, List, Optional, Dict

from ml.models.tft.modules import *
from ml.models.revin import RevInstanceNorm

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_time_variables : int, num_time_categorical_variables : int, num_static_variables : int, num_static_categorical_variables : int, time_category_counts : List[int], stat_category_counts : List[int], hist_variable_idxs : List[int], futr_variable_idxs : List[int], seq_len : int, input_size : int, output_size : int, hidden_layer_size : int, dropout_rate : float, num_encoder_steps : int, num_heads : int, output_activation : nn.Module = nn.Identity()) -> None:
        super(TemporalFusionTransformer, self).__init__()

        self.num_time_variables = num_time_variables
        self.num_static_variables = num_static_variables
        self.num_time_categorical_variables = num_time_categorical_variables
        self.num_static_categorical_variables = num_static_categorical_variables
        self.time_category_counts = time_category_counts
        self.stat_category_counts = stat_category_counts
        self.hist_variable_idxs = hist_variable_idxs
        self.futr_variable_idxs = futr_variable_idxs

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.num_encoder_steps = num_encoder_steps
        self.num_heads = num_heads

        # Build embeddings
        self.stat_categorical_var_embeddings = nn.ModuleList([
            nn.Embedding(self.stat_category_counts[i], self.hidden_layer_size) for i in range(self.num_static_categorical_variables)
        ])
        self.static_var_embeddings = nn.ModuleList([
            nn.Linear(1, self.hidden_layer_size) for i in range(self.num_static_variables)
        ])
        self.time_categorical_var_embeddings = nn.ModuleList([
            nn.Embedding(self.time_category_counts[i], self.hidden_layer_size) for i in range(self.num_time_categorical_variables)
        ])
        self.time_var_embeddings = nn.ModuleList([
            nn.Linear(1, self.hidden_layer_size) for i in range(self.num_time_variables)
        ])

        # Build variable selection networks
        self.static_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size * (self.num_static_variables + self.num_static_categorical_variables),
            output_size=(self.num_static_variables + self.num_static_categorical_variables),
            dropout_rate=self.dropout_rate
        )
        self.time_hist_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size * len(self.hist_variable_idxs),
            output_size=len(self.hist_variable_idxs),
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size
        )
        self.time_futr_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size * len(self.futr_variable_idxs),
            output_size=len(self.futr_variable_idxs),
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size
        )

        # Build static context networks
        self.static_context_variable_selection_grn = GatedResidualNetwork(self.hidden_layer_size, input_size=self.hidden_layer_size, dropout_rate=self.dropout_rate)
        self.static_context_enrichment_grn = GatedResidualNetwork(self.hidden_layer_size, input_size=self.hidden_layer_size, dropout_rate=self.dropout_rate)
        self.static_context_state_h_grn = GatedResidualNetwork(self.hidden_layer_size, input_size=self.hidden_layer_size, dropout_rate=self.dropout_rate)
        self.static_context_state_c_grn = GatedResidualNetwork(self.hidden_layer_size, input_size=self.hidden_layer_size, dropout_rate=self.dropout_rate)

        # Build LSTM
        self.historical_lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            batch_first=True
        )
        self.future_lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            batch_first=True
        )

        # Build post LSTM gate add norm
        self.post_seq_encoder_gate_add_norm = GatedAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=nn.Identity()
        )

        # Build static enrichment
        self.static_enrichment = GatedResidualNetwork(
            self.hidden_layer_size,
            input_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size
        )

        # Build temporal self attention
        self.self_attn_layer = InterpretableMultiHeadAttention(
            n_head=self.num_heads,
            d_model=self.hidden_layer_size,
            dropout=self.dropout_rate
        )
        self.post_attn_gate_add_norm = GatedAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=nn.Identity()
        )

        # Build position wise feed forward network
        self.grn_positionwise = GatedResidualNetwork(
            self.hidden_layer_size,
            input_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate
        )
        self.post_tfd_gate_add_norm = GatedAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=nn.Identity()
        )

        # Build output feed forward
        self.output_feed_forward = nn.Linear(self.hidden_layer_size, self.output_size)

        # Output activation
        self.output_activation = output_activation

        # Reverse instance normalization for the target variable
        self.revin_target = RevInstanceNorm(num_features=1)

        # Interpretability parameters
        self.interpretability_params = []

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ('lstm' in name and 'ih' in name) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            elif ('lstm' in name and 'hh' in name) and 'bias' not in name:
                torch.nn.init.orthogonal_(p)
            elif 'lstm' in name and 'bias' in name:
                torch.nn.init.zeros_(p)

    def get_decoder_mask(self, self_attn_inputs : torch.Tensor) -> torch.Tensor:
        seq_len = self_attn_inputs.shape[1]
        btsz = self_attn_inputs.shape[0]
        mask = torch.cumsum(torch.eye(seq_len), 0)
        mask = mask.repeat(btsz, 1, 1).to(torch.float32)
        return mask.to(self_attn_inputs.device)

    def forward(self, time_inputs : torch.Tensor, time_categorical_inputs : torch.Tensor, stat_inputs : torch.Tensor, stat_categorical_inputs : torch.Tensor) -> List[torch.Tensor]:       
        # Use normalization from non-stationary transformer
        # Assume that the last variable is the target
        past_targets, future_targets = time_inputs[:, :self.num_encoder_steps, -1:], time_inputs[:, self.num_encoder_steps:, -1:]
        past_targets = self.revin_target(past_targets, mode='norm')
        time_inputs = torch.cat([time_inputs[..., :-1], torch.cat([past_targets, future_targets], dim=1)], dim=-1)

        if self.num_time_variables > 0:
            time_inputs_emb = torch.stack([self.time_var_embeddings[i](time_inputs[..., i:i+1]) for i in range(self.num_time_variables)], dim=-2)
        else:
            time_inputs_emb = torch.zeros((time_categorical_inputs.shape[0], time_categorical_inputs.shape[1], 0, self.hidden_layer_size), device=time_categorical_inputs.device)
        if self.num_time_categorical_variables > 0:
            time_categorical_inputs_emb = torch.stack([self.time_categorical_var_embeddings[i](time_categorical_inputs[..., i:i+1]) for i in range(self.num_time_categorical_variables)], dim=-2)
        else:
            time_categorical_inputs_emb = torch.zeros((time_inputs.shape[0], time_inputs.shape[1], 0, self.hidden_layer_size), device=time_inputs.device)
        comb_inputs_emb = torch.cat([time_inputs_emb, time_categorical_inputs_emb], dim=-2)
        hist_inputs_emb = comb_inputs_emb[..., :self.num_encoder_steps, self.hist_variable_idxs, :]
        futr_inputs_emb = comb_inputs_emb[..., self.num_encoder_steps:, self.futr_variable_idxs, :]

        if self.num_static_variables > 0 or self.num_static_categorical_variables > 0:
            if self.num_static_variables > 0:
                stat_inputs_emb = torch.cat([self.static_var_embeddings[i](stat_inputs[..., i:i+1]).unsqueeze(1) for i in range(self.num_static_variables)], dim=1)
            else:
                stat_inputs_emb = torch.zeros((stat_categorical_inputs.shape[0], 0, self.hidden_layer_size), device=stat_categorical_inputs.device)
            if self.num_static_categorical_variables > 0:
                stat_categorical_inputs_emb = torch.cat([self.stat_categorical_var_embeddings[i](stat_categorical_inputs[..., i:i+1]) for i in range(self.num_static_categorical_variables)], dim=1)
            else:
                stat_categorical_inputs_emb = torch.zeros((stat_inputs.shape[0], 0, self.hidden_layer_size), device=stat_inputs.device)

            comb_stat_inputs_emb = torch.cat([stat_inputs_emb, stat_categorical_inputs_emb], dim=1)
            #print(comb_stat_inputs_emb.shape)
            
            stat_encoder, sparse_weights = self.static_vsn(comb_stat_inputs_emb)
        else:
            stat_encoder = torch.zeros((hist_inputs_emb.shape[0], self.hidden_layer_size), device=hist_inputs_emb.device)
            sparse_weights = torch.zeros((hist_inputs_emb.shape[0], 0, 1), device=hist_inputs_emb.device)

        static_context_variable_selection = self.static_context_variable_selection_grn(stat_encoder)
        static_context_enrichment = self.static_context_enrichment_grn(stat_encoder)
        static_context_state_h = self.static_context_state_h_grn(stat_encoder)
        static_context_state_c = self.static_context_state_c_grn(stat_encoder)

        hist_feats, hist_flags = self.time_hist_vsn((hist_inputs_emb, static_context_variable_selection))
        futr_feats, futr_flags = self.time_futr_vsn((futr_inputs_emb, static_context_variable_selection))

        hist_lstm, (state_h, state_c) = self.historical_lstm(hist_feats, (static_context_state_h.unsqueeze(0), static_context_state_c.unsqueeze(0)))
        futr_lstm, _ = self.future_lstm(futr_feats, (state_h, state_c))

        # Apply gated skip connection
        input_embeddings = torch.cat([hist_feats, futr_feats], dim=1)
        lstm_layer = torch.cat([hist_lstm, futr_lstm], dim=1)
        time_feat_layer = self.post_seq_encoder_gate_add_norm(lstm_layer, input_embeddings)

        # Static enrichment layers
        enriched = self.static_enrichment((time_feat_layer, static_context_enrichment.unsqueeze(1)))

        # Decoder self attention
        x, self_att = self.self_attn_layer(enriched, enriched, enriched, mask=self.get_decoder_mask(enriched))
        x = self.post_attn_gate_add_norm(x, enriched)

        # Non-linear processing on outputs
        decoder = self.grn_positionwise(x)

        # Final skip connection
        transformer_layer = self.post_tfd_gate_add_norm(decoder, time_feat_layer)
        outputs = self.output_feed_forward(transformer_layer[..., self.num_encoder_steps:, :])
        outputs = self.output_activation(outputs)
        
        # Normalize forecast back into the original range
        outputs = self.revin_target(outputs, mode='denorm')
        
        # Store interpretability parameters
        self.interpretability_params = {
            "history_vsn_weights": hist_flags,
            "future_vsn_weights": futr_flags,
            "static_encoder_sparse_weights": sparse_weights,
            "attn_weights": self_att,
        }

        return outputs
    
    def feature_importance(self) -> Dict[str, torch.Tensor]:
        """
        Compute the feature importances for historical, fiuture and static features.

        Returns:
            dict: A dictionary containing the fetaure importances for each feature type.
            The keys are 'hist_vsn', 'future_vsn', 'static_vsn' and the values are 
            torch.Tensors with the corresponding feature impotances.
        """
        if not self.interpretability_params:
            raise ValueError(
                'No interpretability_params available. Make a prediction using the model to generate them.'
            )
        
        importances = {}
        # Historical feature importances
        history_vsn_weights = self.interpretability_params['history_vsn_weights']
        importances['history_var_weights'] = history_vsn_weights.mean(dim=0)

        # Future feature importances
        future_vsn_weights = self.interpretability_params['future_vsn_weights']
        importances['future_var_weights'] = future_vsn_weights.mean(dim=0)

        # Future feature importances
        static_encoder_sparse_weights = self.interpretability_params['static_encoder_sparse_weights']
        importances['static_var_weights'] = static_encoder_sparse_weights

        return importances

    def attention_weights(self) -> torch.Tensor:
        """
        Batch average attention weights.

        Returns:
            torch.Tensor: 1D array containing the attention weights for each time-step.
        """

        attention = self.interpretability_params['attn_weights'].mean(dim=0).mean(dim=0)
        return attention