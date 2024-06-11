from .classification import CFTask
from .graph_classification import GraphCFT, NodeCFT

tasks = {
    'classification': CFTask,
    'graphcft': GraphCFT,
    'nodecft': NodeCFT
}