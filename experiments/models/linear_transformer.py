import torch
from torch import nn


def attention(P, Q, Z, activation = None):
    """Apply a single linear attention block.

    Parameters:
        P: Value matrix with shape `(d, d)`.
        Q: Query-key product matrix with shape `(d, d)`.
        Z: Prompt tensor with shape `(B, N + 1, d + 1)`.
        activation: Optional activation applied to the attention scores.
    """

    batch_size = Z.shape[0]
    context_length = Z.shape[1] - 1
    dimension = Z.shape[2] - 1
    device = Z.device
    dtype = Z.dtype

    P_full = torch.zeros(dimension + 1, dimension + 1, device = device, dtype = dtype)
    Q_full = torch.zeros(dimension + 1, dimension + 1, device = device, dtype = dtype)
    P_full[:dimension, :dimension] = P
    Q_full[:dimension, :dimension] = Q
    P_full[dimension, dimension] = 1.0

    mask = torch.eye(context_length + 1, device = device, dtype = dtype)
    mask[context_length, context_length] = 0.0
    attn_scores = torch.einsum("BNi,ij,BMj->BNM", Z, Q_full, Z)
    if activation is not None:
        attn_scores = activation(attn_scores)

    key_tensor = torch.einsum("ij,BNj->BNi", P_full, Z)
    output = torch.einsum("BNM,ML,BLi->BNi", attn_scores, mask, key_tensor)
    return output / context_length


class TransformerF(nn.Module):
    """Linear transformer used throughout the reproduction suite.

    Parameters:
        n_layer: Number of attention layers.
        n_head: Number of heads per layer.
        dimension: Covariate dimension.
        init_var: Standard deviation used for parameter initialization.
    """

    def __init__(self, n_layer, n_head, dimension, init_var):
        """Initialize the trainable tensor stack.

        Parameters:
            n_layer: Number of attention layers.
            n_head: Number of heads per layer.
            dimension: Covariate dimension.
            init_var: Standard deviation used for parameter initialization.
        """

        super().__init__()
        self.register_parameter("allparam", nn.Parameter(torch.zeros(n_layer, n_head, 2, dimension, dimension)))
        with torch.no_grad():
            self.allparam.normal_(0.0, init_var)
        self.n_layer = n_layer
        self.n_head = n_head

    def forward(self, Z):
        """Run the residual linear transformer forward pass.

        Parameters:
            Z: Prompt tensor with shape `(B, N + 1, d + 1)`.
        """

        output = Z
        for layer_index in range(self.n_layer):
            residual = 0.0
            layer_input = output
            for head_index in range(self.n_head):
                P_matrix = self.allparam[layer_index, head_index, 0, :, :]
                Q_matrix = self.allparam[layer_index, head_index, 1, :, :]
                residual = residual + attention(P_matrix, Q_matrix, layer_input)
            output = layer_input + residual
        return output

    def zero_p(self):
        """Zero every `P` matrix in place.

        Parameters:
            None: This method does not accept runtime parameters.
        """

        for layer_index in range(self.n_layer):
            for head_index in range(self.n_head):
                with torch.no_grad():
                    self.allparam[layer_index, head_index, 0, :, :].zero_()


def in_context_loss(model, Z, y):
    """Compute the in-context regression loss used by the reference experiments.

    Parameters:
        model: Transformer model to evaluate.
        Z: Prompt tensor with shape `(B, N + 1, d + 1)`.
        y: Query labels with shape `(B,)`.
    """

    context_length = Z.shape[1] - 1
    dimension = Z.shape[2] - 1
    output = model(Z)
    diff = output[:, context_length, dimension] + y
    return (diff ** 2).mean()
