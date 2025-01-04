import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GraphNeuralNetwork:
    """Implements Graph Neural Network for complex relationship modeling."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.model = GCN(input_dim, hidden_dim, output_dim)
        self.graph = nx.Graph()
        self.node_features = {}  # Store node features
        self.edge_index = None
        self.features = None

    def add_relationship(self, source: str, target: str, relationship_type: str):
        """Add relationship to graph."""
        self.graph.add_edge(source, target, type=relationship_type)
        self._update_edge_index()

    def add_node_feature(self, node:str, feature: torch.Tensor):
        self.node_features[node] = feature

    def _update_edge_index(self):
        edge_list = list(self.graph.edges())
        if edge_list:
             # mapping nodes to indices and making edge index from indices
            nodes = list(self.graph.nodes())
            node_to_index = {node: index for index, node in enumerate(nodes)}
            self.edge_index = torch.tensor(
                 [[node_to_index[u], node_to_index[v]] for u,v in edge_list], dtype=torch.long
            ).t().contiguous()
        else:
            self.edge_index = None

    def prepare_data(self):
         """Prepare data for GNN training."""
         try:
             if not self.node_features:
                 logger.warning("No node features provided, cannot prepare data.")
                 return False

             nodes = list(self.graph.nodes())

             # create list of node features
             features_list = [self.node_features[node] for node in nodes]
             # check if all tensors have the same shape
             if len(set([f.shape for f in features_list])) > 1:
                 logger.warning("Node features do not have uniform shapes")
                 return False
             
             self.features = torch.stack(features_list)
             
             # make sure all nodes are used in edges or use all the nodes if there are no edges
             if self.edge_index is None and len(self.graph.nodes()) > 0:
                self.edge_index = torch.empty((2, 0), dtype=torch.long)

             return True

         except Exception as e:
            logger.error(f"Graph data preparation failed: {e}")
            return False

    def train(self, labels: torch.Tensor):
        """Train the GNN model."""
        try:
            if not self.prepare_data():
                logger.warning("Failed to prepare graph data for training.")
                return

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.model.train()
            optimizer.zero_grad()

            # Forward pass
            out = self.model(self.features, self.edge_index)

            loss = F.cross_entropy(out, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            return loss

        except Exception as e:
            logger.error(f"GNN training failed: {e}")
            raise