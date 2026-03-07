"""Activation patching protocols (DISAMB/CF/COH) built on a shared runner."""

from .base import ActivationPatchingProtocol, CaseSkip, PatchingCase, run_activation_patching
from .authority_protocol import AuthorityLanguageGameProtocol
from .coh_protocol import COHConstraintAblationProtocol
from .cf_protocol import CFInterventionSwapProtocol
from .disamb_protocol import DISAMBContextSwapProtocol
