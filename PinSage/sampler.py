import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchtext.data.functional import numericalize_tokens_from_iterator


def padding(array, yy, val):
    """
    :param array: torch tensor array
    :param yy: desired width
    :param val: padded value
    :return: padded array
    """
    w = array.shape[0]
    b = 0
    bb = yy - b - w

    return torch.nn.functional.pad(
        array, pad=(b, bb), mode="constant", value=val
    )


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block



class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        # Initialize the instance variables.
        self.g = g  # The user-item graph
        self.user_type = user_type  # The type of user nodes in the graph
        self.item_type = item_type  # The type of item nodes in the graph
        # Extract the type of edges from user to item and item to user from the metagraph of g.
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size  # The size of batches to be generated

    def __iter__(self):
        # This method makes this class an iterable.
        while True:
            # Select random item nodes as the start of the path.
            heads = torch.randint(0, self.g.num_nodes(self.item_type), (self.batch_size,))

            # Perform a two-step random walk following the metapath [Item, User, Item].
            # The first step goes from an item to a user, and the second step goes from the user to another item.
            # This generates the target item nodes for the given start item nodes.
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype],
            )[0][:, 2]

            # Select random item nodes as negative samples.
            neg_tails = torch.randint(0, self.g.num_nodes(self.item_type), (self.batch_size,))

            # Create a mask for valid target nodes (i.e., target nodes that are not -1).
            mask = tails != -1

            # Yield the start nodes, target nodes, and negative nodes that pass the mask.
            yield heads[mask], tails[mask], neg_tails[mask]

            
class NeighborSampler(object):
    def __init__(
        self,
        g,
        user_type,
        item_type,
        random_walk_length,
        random_walk_restart_prob,
        num_random_walks,
        num_neighbors,
        num_layers,
    ):
        # Initialize the instance variables
        self.g = g  # The user-item graph
        self.user_type = user_type  # The type of user nodes in the graph
        self.item_type = item_type  # The type of item nodes in the graph
        # Extract the type of edges from user to item and item to user from the metagraph of g
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]

        # Create a list of PinSAGE samplers for random walk-based neighbor sampling
        self.samplers = [
            dgl.sampling.PinSAGESampler(
                g,
                item_type,
                user_type,
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        # Initialize the list to hold the sampled blocks
        blocks = []

        # Iterate over all samplers
        for sampler in self.samplers:
            # Generate a frontier by sampling neighbors of seeds
            frontier = sampler(seeds)

            # If head, tail, and neg_tails are provided
            if heads is not None:
                # Find the edge IDs between heads and tails/neg_tails
                eids = frontier.edge_ids(
                    torch.cat([heads, heads]),
                    torch.cat([tails, neg_tails]),
                    return_uv=True,
                )[2]
                # If any edges are found
                if len(eids) > 0:
                    # Remove these edges from the frontier
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)

            # Compact the frontier graph and make a block
            block = compact_and_copy(frontier, seeds)

            # The destination nodes of the block will be used as the seeds for next layer
            seeds = block.srcdata[dgl.NID]

            # Add the block to the front of the list
            blocks.insert(0, block)

        # Return the list of blocks
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative connections only.
        pos_graph = dgl.graph(
            (heads, tails), num_nodes=self.g.num_nodes(self.item_type)
        )
        neg_graph = dgl.graph(
            (heads, neg_tails), num_nodes=self.g.num_nodes(self.item_type)
        )

        # Compact the graphs so that they share the same set of nodes
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])

        # Use the nodes of the positive graph as the seeds
        seeds = pos_graph.ndata[dgl.NID]

        # Sample blocks using the seeds, heads, tails, and neg_tails
        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)

        # Return the positive graph, the negative graph, and the list of blocks
        return pos_graph, neg_graph, blocks



def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]


def assign_textual_node_features(ndata, textset, ntype):
    """
    Assigns numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    block : DGLGraph
        First element of the compacted blocks, with "dgl.NID" as the
        corresponding node ID in the original graph, hence the index to the
        text dataset.

        The numericalized tokens (and lengths if available) would be stored
        onto the blocks as new node features.
    textset : torchtext.data.Dataset
        A torchtext dataset whose number of examples is the same as that
        of nodes in the original graph.
    """
    node_ids = ndata[dgl.NID].numpy()

    for field_name, field in textset.items():
        textlist, vocab, pad_var, batch_first = field

        examples = [textlist[i] for i in node_ids]
        ids_iter = numericalize_tokens_from_iterator(vocab, examples)

        maxsize = max([len(textlist[i]) for i in node_ids])
        ids = next(ids_iter)
        x = torch.asarray([num for num in ids])
        lengths = torch.tensor([len(x)])
        tokens = padding(x, maxsize, pad_var)

        for ids in ids_iter:
            x = torch.asarray([num for num in ids])
            l = torch.tensor([len(x)])
            y = padding(x, maxsize, pad_var)
            tokens = torch.vstack((tokens, y))
            lengths = torch.cat((lengths, l))

        if not batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + "__len"] = lengths


def assign_features_to_blocks(blocks, g, textset, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_textual_node_features(blocks[0].srcdata, textset, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    assign_textual_node_features(blocks[-1].dstdata, textset, ntype)

    
class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype, textset):
        # Initialize the instance variables.
        self.sampler = sampler  # The neighborhood sampler
        self.ntype = ntype  # The type of nodes in the graph
        self.g = g  # The user-item graph
        self.textset = textset  # The dataset containing the text features of the nodes

    def collate_train(self, batches):
        # Extract the start nodes (heads), target nodes (tails), and negative nodes (neg_tails) from the batch.
        heads, tails, neg_tails = batches[0]
        
        # Construct multilayer neighborhood using PinSAGE sampler and generate positive and negative graphs, and blocks.
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        
        # Assign features to the nodes in the blocks.
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)
        
        # Return the positive graph, negative graph, and blocks for training.
        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        # Convert the samples into a tensor.
        batch = torch.LongTensor(samples)
        
        # Sample blocks for the nodes in the batch.
        blocks = self.sampler.sample_blocks(batch)
        
        # Assign features to the nodes in the blocks.
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)
        
        # Return the blocks for testing.
        return blocks
