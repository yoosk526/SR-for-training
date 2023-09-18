from .rlfn import RLFN
from .rlfn_s import RLFN_S
from .abpn import ABPN

def get_model(args):
    model = args.model
    upscale = args.hr_size // args.lr_size
    feature = args.feature
    
    if model == 'rlfn':
        return RLFN(feature_channels=feature, upscale=upscale)
    
    if model == 'rlfn_s':
        return RLFN_S(feature_channels=feature, upscale=upscale)
    
    if model == 'abpn':
        return ABPN(feature=feature, upscale_ratio=upscale)