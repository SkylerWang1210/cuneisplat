from .loss import Loss
from .loss_cunei_geometry import (
    LossCuneiGeometry,
    LossCuneiGeometryCfgWrapper,
)
from .loss_cunei_surface import (
    LossCuneiSurface,
    LossCuneiSurfaceCfgWrapper,
)
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossCuneiGeometryCfgWrapper: LossCuneiGeometry,
    LossCuneiSurfaceCfgWrapper: LossCuneiSurface,
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = (
    LossCuneiGeometryCfgWrapper
    | LossCuneiSurfaceCfgWrapper
    | LossDepthCfgWrapper
    | LossLpipsCfgWrapper
    | LossMseCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
