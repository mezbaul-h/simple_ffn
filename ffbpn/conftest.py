import pytest
from .network import Network
from .layer import Layer
from .activation import Sigmoid


@pytest.fixture
def network_232():
    return Network(
        Layer(2, 3, activation=Sigmoid()),
        Layer(3, 2, activation=Sigmoid()),
    )
