"""Pytest fixtures and configuration for pysimlr E2E tests."""

import pytest
import torch
import numpy as np
import random


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Ensure determinism across all E2E test runs."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def synthetic_two_views():
    """
    Generate clean 2-view synthetic data with ground-truth latent components.
    N=60 samples, p1=16 features, p2=12 features, k=2 shared dimensions.
    """
    torch.manual_seed(42)
    n, p1, p2, k = 60, 16, 12, 2
    u_true = torch.randn(n, k)
    # Orthogonalize ground truth U
    u_true, _ = torch.linalg.qr(u_true)

    w1 = torch.randn(k, p1)
    w2 = torch.randn(k, p2)

    x1 = u_true @ w1 + 0.05 * torch.randn(n, p1)
    x2 = u_true @ w2 + 0.05 * torch.randn(n, p2)

    return {
        "x1": x1,
        "x2": x2,
        "u_true": u_true,
        "w1": w1,
        "w2": w2,
        "n": n,
        "p1": p1,
        "p2": p2,
        "k": k
    }


@pytest.fixture
def synthetic_three_views():
    """
    Generate 3-view synthetic data with shared consensus.
    N=50 samples, dimensions: p1=14, p2=10, p3=12, k=3 shared latent components.
    """
    torch.manual_seed(101)
    n, p1, p2, p3, k = 50, 14, 10, 12, 3
    u_true = torch.randn(n, k)
    u_true, _ = torch.linalg.qr(u_true)

    w1 = torch.randn(k, p1)
    w2 = torch.randn(k, p2)
    w3 = torch.randn(k, p3)

    x1 = u_true @ w1 + 0.05 * torch.randn(n, p1)
    x2 = u_true @ w2 + 0.05 * torch.randn(n, p2)
    x3 = u_true @ w3 + 0.05 * torch.randn(n, p3)

    return {
        "views": [x1, x2, x3],
        "u_true": u_true,
        "n": n,
        "dims": [p1, p2, p3],
        "k": k
    }


@pytest.fixture
def synthetic_omics_data():
    """
    Simulated multi-omics dataset:
    - View 0: Transcriptomics (RNA-seq normalized logCPM, 80 samples x 40 genes)
    - View 1: Proteomics (LC-MS/MS normalized abundance, 80 samples x 25 proteins)
    Driven by 2 biological pathways + modality-specific noise.
    """
    torch.manual_seed(2024)
    n, p_rna, p_prot, k = 80, 40, 25, 2

    # Latent pathway activity
    pathway_signal = torch.randn(n, k)
    pathway_signal, _ = torch.linalg.qr(pathway_signal)

    # Loadings
    v_rna = torch.randn(k, p_rna)
    v_prot = torch.randn(k, p_prot)

    x_rna = pathway_signal @ v_rna + 0.1 * torch.randn(n, p_rna)
    x_prot = pathway_signal @ v_prot + 0.1 * torch.randn(n, p_prot)

    return {
        "x_rna": x_rna,
        "x_prot": x_prot,
        "pathway_signal": pathway_signal,
        "n": n,
        "p_rna": p_rna,
        "p_prot": p_prot,
        "k": k
    }
