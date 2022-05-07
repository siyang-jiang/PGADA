from functools import partial

from src.modules.batch_norm import *

BATCHNORM = ConventionalBatchNorm
# BATCHNORM = TransductiveBatchNorm

from src.modules.backbones import *
from src.modules import *
from src.methods import *

# Parameters of the model (method and feature extractor)

BACKBONE = Conv4
# BACKBONE = Conv6
# BACKBONE = ResNet10
# BACKBONE = Conv4S

H = H_3

TRANSPORTATION_MODULE = OptimalTransport(
    regularization=0.05,
    learn_regularization=False,
    max_iter=1000,
    stopping_criterion=1e-4,
)

MODEL = partial(
    # ProtoNet,
    MatchingNet,
    # TransPropNet,
    # TransFineTune,
    transportation=TRANSPORTATION_MODULE,
)
