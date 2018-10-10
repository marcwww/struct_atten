from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchChildSumTreeLSTMCell(nn.Module):
    """Child-Sum Tree-LSTM Cell implementation for mini batches.

    Based on https://arxiv.org/abs/1503.00075.
    Equations on p.3 as follows.

    .. math::

        \begin{array}{ll}
          \tilde{h_j} = \sum_{k \in C(j)} h_k \\
          i_j = \mathrm{sigmoid}(W^{(i)} x_j + U^{(i)} \tilde{h}_j + b^{(i)}) \\
          f_{jk} = \mathrm{sigmoid}(W^{(f)} x_j + U^{(f)} h_k + b^{(f)}) \\
          o_j = \mathrm{sigmoid}(W^{(o)} x_j + U^{(o)} \tilde{h}_j + b^{(o)}) \\
          u_j = \tanh(W^{(u)} x_j + U^{(u)} \tilde{h}_j + b^{(u)}) \\
          c_j = i_j \circ u_j + \sum_{k \in C(j)} f_{jk} \circ c_k \\
          h_j = o_j \circ \tanh(c_j)
        \end{array}
    """

    def __init__(self, input_size, hidden_size, p_dropout):
        """Create a new ChildSumTreeLSTMCell.

        Args:
          input_size: Integer, the size of the input vector.
          hidden_size: Integer, the size of the hidden state to return.
          dropout: torch.nn.Dropout module.
        """
        super(BatchChildSumTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=p_dropout)
        self.W_combined = nn.Parameter(
            torch.Tensor(input_size + hidden_size, 3 * hidden_size),
            requires_grad=True)
        self.b_combined = nn.Parameter(
            torch.zeros(1, 3 * hidden_size),
            requires_grad=True)
        self.W_f = nn.Parameter(
            torch.Tensor(input_size, hidden_size),
            requires_grad=True)
        self.U_f = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size),
            requires_grad=True)
        self.b_f = nn.Parameter(
            torch.zeros(1, hidden_size),
            requires_grad=True)
        nn.init.xavier_uniform_(self.W_combined, gain=1.0)
        nn.init.xavier_uniform_(self.W_f, gain=1.0)
        nn.init.xavier_uniform_(self.U_f, gain=1.0)

    def forward(self, inputs, previous_states):
        """Calculate the next hidden state given the inputs.

        This is for custom control over a batch, designed for efficiency.
        I hope it is efficient...

        Args:
          inputs: List of Tensors of shape (1, input_size) - row vectors.
          previous_states: Tuple of (List, List), being cell_states and
            hidden_states respectively. Inside the lists, for nodes with
            multiple children, we expect they are already concatenated into
            matrices.

        Returns:
          cell_states, hidden_states: being state tuple, where both states are
             row vectors of length hidden_size.
        """
        # prepare the inputs
        cell_states = previous_states[0]
        hidden_states = previous_states[1]
        inputs_mat = inputs
        h_tilde_mat = torch.cat([torch.sum(h, 0).expand(1, self.hidden_size)
                                 for h in hidden_states],
                                dim=0)
        prev_c_mat = torch.cat(cell_states, 0)
        big_cat_in = torch.cat([inputs_mat, h_tilde_mat], 1)

        # process in parallel those parts we can
        big_cat_out = big_cat_in.mm(self.W_combined) + self.b_combined.expand(
            big_cat_in.size()[0],
            3 * self.hidden_size)
        z_i, z_o, z_u = big_cat_out.split(self.hidden_size, 1)

        # apply dropout to u, like the Fold boys
        z_u = self.dropout(z_u)

        # forget gates
        f_inputs = inputs_mat.mm(self.W_f)
        # we can concat the matrices along the row axis,
        # but we need to calculate cumsums for splitting after

        # NOTE: I could probably pass this information from pre-processing
        # yes, I think that's the idea: move this out. Test it out there.
        # then come back to here. That's my next job. And moving the other
        # stuff out of the CSTLSTM model.

        # here the lens are for inputs
        lens = [t.size()[0] for t in hidden_states]
        start = [sum([lens[j] for j in range(i)]) for i in range(len(lens))]
        end = [start[i] + lens[i] for i in range(len(lens))]

        # we can then go ahead and concatenate for matmul
        prev_h_mat = torch.cat(hidden_states, 0)
        f_hiddens = prev_h_mat.mm(self.U_f)
        # compute the f_jks by expanding the inputs to the same number
        # of rows as there are prev_hs for each, then just do a simple add.
        indices = [i for i in range(len(lens)) for _ in range(lens[i])]
        f_inputs_ready = f_inputs[indices]

        f_jks = F.sigmoid(
            f_inputs_ready + f_hiddens + self.b_f.expand(
                f_hiddens.size()[0], self.hidden_size))

        # cell and hidden state
        fc_mul = f_jks * prev_c_mat
        sum_idx_mtrx = torch.zeros((len(lens), fc_mul.shape[0])).to(fc_mul)
        for i, (b, e) in enumerate(zip(start, end)):
            sum_idx_mtrx[i, b:e] = 1
        fc_term = sum_idx_mtrx.matmul(fc_mul)

        c = F.sigmoid(z_i) * F.tanh(z_u) + fc_term
        h = F.sigmoid(z_o) * F.tanh(c)

        return c, h

class PreviousStates(nn.Module):
    """For getting previous hidden states from lower level given wirings."""

    def __init__(self, hidden_size):
        super(PreviousStates, self).__init__()
        """Create a new PreviousStates.

        Args:
          hidden_size: Integer, number of units in a hidden state vector.
        """
        self.hidden_size = hidden_size
        self.zero_vec = nn.Parameter(torch.zeros(1, hidden_size),
                                     requires_grad=False)

    def __call__(self, level_nodes, level_up_wirings, prev_outputs):
        """Get previous hidden states.

        Args:
          level_nodes: List of nodes on the level to be processed.
          level_up_wirings: List of Lists: the list is of the same length as the
            level_nodes list. Each sublist gives the integer indices of the
            child nodes in the node list on the previous (lower) level. This
            defines how the child nodes wire up to the parent nodes.
          prev_outputs: List of previous hidden state tuples for the level below
            from which we will select from.

        Returns:
          ?
        """
        # Count how many nodes on this level of the forest.
        level_length = len(level_nodes)

        # grab the cell states
        cell_states = self.states(
            level_nodes, level_length, prev_outputs[0], level_up_wirings)

        # grab the hidden states
        hidden_states = self.states(
            level_nodes, level_length, prev_outputs[1], level_up_wirings)

        # mind the order of returning
        return cell_states, hidden_states

    def states(self, level_nodes, level_length, prev_out, child_ixs_level):
        return [(self.zero_vec
                 if (level_nodes[i].is_leaf or len(child_ixs_level[i]) == 0)
                 else prev_out.index_select(0, child_ixs_level[i]))
                for i in range(level_length)]

class ChildSumTreeLSTMEncoder(nn.Module):
    """Child-Sum Tree-LSTM Encoder Module."""

    def __init__(self, edim, hdim, embeddings,
                 p_keep_input, p_keep_rnn, padding_idx):
        """Create a new ChildSumTreeLSTMEncoder.

        Args:
          embed_size: Integer, number of units in word embeddings vectors.
          hidden_size: Integer, number of units in hidden state vectors.
          embeddings: torch.nn.Embedding.
          p_keep_input: Float, the probability of keeping an input unit.
          p_keep_rnn: Float, the probability of keeping an rnn unit.
        """
        super(ChildSumTreeLSTMEncoder, self).__init__()

        self._embeddings = embeddings
        self.sema_dim = hdim
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx

        # Define dropout layer for embedding lookup
        self._drop_input = nn.Dropout(p=1.0 - p_keep_input)

        # Initialize the batch Child-Sum Tree-LSTM cell
        self.cell = BatchChildSumTreeLSTMCell(
            input_size=edim,
            hidden_size=hdim,
            p_dropout=1.0 - p_keep_rnn)

        self.zero_vec = nn.Parameter(torch.zeros(1, hdim),
                                     requires_grad=False)
        # Initialize previous states (to get wirings from nodes on lower level)
        self._prev_states = PreviousStates(hdim)

    def init_hidden(self, level_length):
        cell_states = [self.zero_vec for _ in range(level_length)]
        hidden_states = [self.zero_vec for _ in range(level_length)]
        return cell_states, hidden_states

    def forward(self, forest):
        """Get encoded vectors for each node in the forest.

        Args:
          nodes: Dictionary of structure {Integer (level_index): List (nodes)}
            where each node is represented by a ext.Node object.
          up_wirings: Dictionary of structure
            {Integer (level_index): List of Lists (up wirings)}, where the up
            wirings List is the same length as the number of nodes on the
            current level, and each sublist gives the indices of it's children
            on the lower level's node list, thus defining the upward wiring.

        Returns:
          Dictionary of hidden states for all nodes on all levels, indexed by
            level number, with the list order following that of forest.nodes[l]
            for each level, l.
        """
        outputs = {}
        bsz = len(forest.trees)
        nodes_map = {i:[] for i in range(bsz)}
        # -1 for ROOT
        max_nnodes = max([len(tree.node_list)-1 for tree in forest.trees])
        nodes = torch.zeros(bsz, max_nnodes, self.hdim).to(self.zero_vec)

        # Work backwards through level indices - i.e. bottom up.
        for l in reversed(range(forest.max_level + 1)):

            # Get input word vectors for this level.
            word_indices = torch.cat([n.vocab_ix for n in forest.nodes[l]])
            mask = word_indices.ne(self.padding_idx).float()
            inputs = self._embeddings(word_indices) * mask.unsqueeze(-1)
            inputs = self._drop_input(inputs)

            # Get previous hidden states for this level.
            if l == forest.max_level:
                hidden_states = self.init_hidden(len(forest.nodes[l]))
            else:
                # because the for-cycling is started from the bottom,
                # the pre_ouputs come from depth at (l+1)

                # this branch is used to filter out the children
                # hidden states
                hidden_states = self._prev_states(
                    level_nodes=forest.nodes[l],
                    level_up_wirings=forest.child_ixs[l],
                    prev_outputs=outputs[l+1])

            outputs[l] = self.cell(inputs, hidden_states)
            if l!= 0:
                # record the tree nodes
                for i, n in enumerate(forest.nodes[l]):
                    b = n.tree_ix
                    h_node = inputs[i] if n.is_leaf else outputs[l][1][i]
                    nodes[b, len(nodes_map[b])] = h_node
                    nodes_map[b].append(h_node)

        mask = nodes.sum(-1).ne(0)
        h = outputs[1][1]

        return {'h': h,
                'nodes': nodes,
                'mask':mask}
