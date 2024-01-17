import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl

from frame2seq.utils.rigid_utils import Rigid
from frame2seq.openfold.model.primitives import LayerNorm
from frame2seq.openfold.model.structure_module import InvariantPointAttention, StructureModuleTransition
from frame2seq.model.edge_update import EdgeTransition
from frame2seq.utils.featurize import make_s_init, make_z_init

import torch
from torch import Tensor
from typing import List, Optional, Tuple

class frame2seq(pl.LightningModule):
    """
    A PyTorch Lightning module that implements the frame2seq model for protein structure prediction.

    Attributes:
        config (dict): Configuration dictionary containing model hyperparameters.
        sequence_dim (int): Dimensionality of the sequence embeddings.
        single_dim (int): Dimensionality of the single residue embeddings.
        pair_dim (int): Dimensionality of the pairwise residue embeddings.
        layers (nn.ModuleList): List of layers comprising the frame2seq model.
        ... (other attributes)
    """

    def __init__(self, config: dict):
        """
        Initializes the frame2seq model with the given configuration.

        Args:
            config (dict): Configuration dictionary containing model hyperparameters.
        """
        super(frame2seq, self).__init__()
        self.save_hyperparameters()
        config = self.hparams.config
        self.config = config

        # Extracting hyperparameters from the configuration
        ipa_depth = config['ipa_depth']
        ipa_dim = config['ipa_dim']
        ipa_heads = config['ipa_heads']
        ipa_pairwise_repr_dim = config['ipa_pairwise_repr_dim']
        self.st_mod_tsit_factor = config['st_mod_tsit_factor']
        self.sequence_dim = config['sequence_dim']
        self.single_dim = config['single_dim']

        # Define constants for torsion and distance binning
        self.torsion_bin_width = 8
        self.torsion_bins = 360 // self.torsion_bin_width
        self.relpos_k = 32
        self.dist_bin_width = 0.5
        self.dist_bins = 24

        # Calculate the dimensionality of the pairwise residue embeddings
        self.pair_dim = 16 * self.dist_bins + 2 * self.relpos_k + 1

        # Define linear transformations for different types of embeddings
        self.sequence_to_single = nn.Linear(6 + self.single_dim, self.single_dim)
        self.edge_to_pair = nn.Linear(self.pair_dim, ipa_pairwise_repr_dim)
        self.single_to_sequence = nn.Linear(self.single_dim, self.sequence_dim)

        # Initialize layers of the model
        self.layers = nn.ModuleList([])
        for i in range(ipa_depth):
            # Invariant Point Attention (IPA) layer
            ipa = InvariantPointAttention(
                ipa_dim,
                ipa_pairwise_repr_dim,
                ipa_dim // ipa_heads,
                ipa_heads,
                4,
                8,
            )
            # Dropout and LayerNorm for the IPA layer
            ipa_dropout = nn.Dropout(0.1)
            layer_norm_ipa = LayerNorm(ipa_dim)

            # Transition layers for the structure module
            if self.st_mod_tsit_factor > 1:
                pre_transit = nn.Linear(ipa_dim, ipa_dim * self.st_mod_tsit_factor)
                post_transit = nn.Linear(ipa_dim * self.st_mod_tsit_factor, ipa_dim)
            transition = StructureModuleTransition(
                ipa_dim * self.st_mod_tsit_factor,
                1,
                0.1,
            )

            # Edge transition layer, not present in the last layer
            edge_transition = None if i == ipa_depth - 1 else EdgeTransition(
                ipa_dim,
                ipa_pairwise_repr_dim,
                ipa_pairwise_repr_dim,
                num_layers=2,
            )

            # Append the constructed layers to the module list
            if self.st_mod_tsit_factor > 1:
                self.layers.append(
                    nn.ModuleList([
                        ipa, ipa_dropout, layer_norm_ipa, pre_transit,
                        transition, post_transit, edge_transition
                    ]))
            else:
                self.layers.append(
                    nn.ModuleList([
                        ipa, ipa_dropout, layer_norm_ipa, transition,
                        edge_transition
                    ]))

        # Dropout layers for the sequence and pairwise embeddings
        self.s_dropout = nn.Dropout(0.1)
        self.z_dropout = nn.Dropout(0.1)

        # Input transformations for sequence embeddings
        self.input_sequence_to_single = nn.Linear(self.sequence_dim, self.single_dim)
        self.input_sequence_layer_norm = nn.LayerNorm(self.single_dim)

    def forward(self, X: Tensor, seq_mask: Tensor, input_S: Tensor) -> Tensor:
        """
        Forward pass of the frame2seq model.

        Args:
            X (Tensor): Input tensor containing the coordinates of the residues.
            seq_mask (Tensor): Sequence mask indicating valid positions.
            input_S (Tensor): Input tensor containing the initial sequence embeddings.

        Returns:
            Tensor: Predicted sequence embeddings after processing through the model.
        """
        training_bool = self.training
        X = X.to(self.device)
        seq_mask = seq_mask.to(self.device)
        input_S = input_S.to(self.device)

        # Compute rigid transformations for the input coordinates
        r = Rigid.from_3_points(X[:, :, 0], X[:, :, 1], X[:, :, 2])

        # Initialize s and z embeddings
        s, in_S = make_s_init(self, X, input_S, seq_mask)
        s = self.sequence_to_single(s)
        s = s + self.input_sequence_layer_norm(in_S)
        z = make_z_init(self, X)
        z = self.edge_to_pair(z)
        seq_mask = seq_mask.long()

        # Set attention dropout rate based on training or evaluation mode
        attn_drop_rate = 0.2 if training_bool else 0.0

        # Apply dropout to s and z embeddings if in training mode
        if training_bool:
            s = self.s_dropout(s)
            z = self.z_dropout(z)

        # Process through each layer of the model
        for layer in self.layers:
            ipa, ipa_dropout, layer_norm_ipa, *transit_layers, edge_transition = layer
            s = s + ipa(s, z, r, seq_mask, attn_drop_rate=attn_drop_rate)
            s = ipa_dropout(s)
            s = layer_norm_ipa(s)

            # Apply transition layers if configured
            if self.st_mod_tsit_factor > 1:
                pre_transit, transition, post_transit = transit_layers
                s = pre_transit(s)
                s = transition(s)
                s = post_transit(s)
            else:
                transition = transit_layers[0]
                s = transition(s)

            # Apply edge transition if present
            if edge_transition is not None:
                z = checkpoint(edge_transition, s, z)

        # Transform the final single residue embeddings to sequence embeddings
        pred_seq = self.single_to_sequence(s)

        return pred_seq