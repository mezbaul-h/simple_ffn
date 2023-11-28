import pytest
from .networks import Network
from .layers import Layer
from .activations import Sigmoid


@pytest.fixture
def network_232():
    return Network(
        Layer(2, 20, activation=Sigmoid()),
        Layer(20, 20, activation=Sigmoid()),
        Layer(20, 2, activation=Sigmoid()),
        learning_rate=0.05,
    )
