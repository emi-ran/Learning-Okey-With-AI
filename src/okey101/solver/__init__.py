"""Correctness-first, deterministic legal candidate solvers."""

from .attachment_solver import AttachmentCandidate, generate_attachments
from .meld_generator import generate_melds
from .opening_solver import OpeningCandidate, find_legal_openings
from .pair_solver import PairOpeningCandidate, find_pair_openings, generate_pairs

__all__ = [
    "AttachmentCandidate",
    "OpeningCandidate",
    "PairOpeningCandidate",
    "find_legal_openings",
    "find_pair_openings",
    "generate_attachments",
    "generate_melds",
    "generate_pairs",
]
