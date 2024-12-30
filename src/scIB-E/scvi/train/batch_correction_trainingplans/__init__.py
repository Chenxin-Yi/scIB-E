# Level 1
from .level1_batch_removal._RBP import RBP_TrainingPlan
from .level1_batch_removal._Orthog import Orthog_TrainingPlan
from .level1_batch_removal._HSIC import HSIC_TrainingPlan
from .level1_batch_removal._MIM import MIM_TrainingPlan
from .level1_batch_removal._RCE import RCE_TrainingPlan
from .level1_batch_removal._GAN import GAN_TrainingPlan

# Level 2
from .level2_celltype_incorporation._CellSupcon import CellSupcon_TrainingPlan
from .level2_celltype_incorporation._IRM import IRM_TrainingPlan
from .level2_celltype_incorporation._Domain_meta_learning import Domain_meta_learning_TrainingPlan
 
# Level 3
from .level3_batch_removal_and_celltype_incorporation._RCE_CE import RCE_CE_TrainingPlan
from .level3_batch_removal_and_celltype_incorporation._HSIC_CellSupcon import HSIC_CellSupcon_TrainingPlan
from .level3_batch_removal_and_celltype_incorporation._MIM_CellSupcon import MIM_CellSupcon_TrainingPlan
from .level3_batch_removal_and_celltype_incorporation._Orthog_CellSupcon import Orthog_CellSupcon_TrainingPlan
from .level3_batch_removal_and_celltype_incorporation._RCE_CellSupcon import RCE_CellSupcon_TrainingPlan
from .level3_batch_removal_and_celltype_incorporation._RBP_CellSupcon import RBP_CellSupcon_TrainingPlan
from .level3_batch_removal_and_celltype_incorporation._DomainClassTripletLoss import DomainClassTripletLoss_TrainingPlan

__all__ = [
    # Level 1
    "RBP_TrainingPlan",
    "Orthog_TrainingPlan",
    "HSIC_TrainingPlan",
    "MIM_TrainingPlan",
    "RCE_TrainingPlan",
    "GAN_TrainingPlan",

    # Level 2
    "CellSupcon_TrainingPlan",
    "IRM_TrainingPlan",
    "Domain_meta_learning_TrainingPlan",

    # Level 3
    "RCE_CE_TrainingPlan",
    "HSIC_CellSupcon_TrainingPlan",
    "MIM_CellSupcon_TrainingPlan",
    "Orthog_CellSupcon_TrainingPlan",
    "RCE_CellSupcon_TrainingPlan",
    "RBP_CellSupcon_TrainingPlan",
    "DomainClassTripletLoss_TrainingPlan",
    ]