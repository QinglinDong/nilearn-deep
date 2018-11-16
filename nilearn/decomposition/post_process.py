"""
Base class for decomposition estimators, utilities for masking and dimension
reduction of group data
"""
from __future__ import division
from math import ceil
import itertools
import glob
from distutils.version import LooseVersion

import numpy as np

from scipy import linalg
import sklearn
import nilearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, svd_flip
from .._utils.cache_mixin import CacheMixin, cache
from .._utils.niimg import _safe_get_data
from .._utils.niimg_conversions import _resolve_globbing
from .._utils.compat import _basestring
from ..input_data import NiftiMapsMasker
from ..input_data.masker_validation import check_embedded_nifti_masker


def fast_svd(X, n_components, random_state=None):
    """ Automatically switch between randomized and lapack SVD (heuristic
        of scikit-learn).

    Parameters
    ==========
    X: array, shape (n_samples, n_features)
        The data to decompose

    n_components: integer
        The order of the dimensionality of the truncated SVD

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    ========

    U: array, shape (n_samples, n_components)
        The first matrix of the truncated svd

    S: array, shape (n_components)
        The second matric of the truncated svd

    V: array, shape (n_components, n_features)
        The last matric of the truncated svd

    """
