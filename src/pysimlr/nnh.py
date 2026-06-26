import torch
import pandas as pd
import numpy as np
import re
from typing import List, Optional, Union, Dict, Any
from .simlr import simlr, initialize_simlr
from .utils import multigrep
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def _get_names_from_dataframe(
    x: Union[str, List[str]],
    df: pd.DataFrame,
    exclusions: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """
    Extract column names with concatenated search parameters from a dataframe.
    """
    patterns = [x] if isinstance(x, str) else list(x)
    if not patterns:
        return []
    
    cols = df.columns.tolist()
    matched = []
    for col in cols:
        matches_all = True
        for pat in patterns:
            if not re.search(pat, col):
                matches_all = False
                break
        if matches_all:
            matched.append(col)
            
    if exclusions is not None:
        exclude_patterns = [exclusions] if isinstance(exclusions, str) else list(exclusions)
        filtered = []
        for col in matched:
            matches_any_exclude = False
            for pat in exclude_patterns:
                if re.search(pat, col):
                    matches_any_exclude = True
                    break
            if not matches_any_exclude:
                filtered.append(col)
        matched = filtered
        
    return matched

def mapAsymVar(
    mydataframe: pd.DataFrame,
    leftvar: List[str],
    leftname: str = "left",
    rightname: str = "right",
    replacer: str = "Asym"
) -> pd.DataFrame:
    """
    Compute absolute asymmetry between left and right variables.
    """
    df = mydataframe.copy()
    rightvar = [col.replace(leftname, rightname) for col in leftvar]
    cols_set = set(df.columns)
    valid_indices = [i for i, r_col in enumerate(rightvar) if r_col in cols_set]
    
    if not valid_indices:
        return df
        
    left_cols = [leftvar[i] for i in valid_indices]
    right_cols = [rightvar[i] for i in valid_indices]
    
    left_data = df[left_cols].values
    right_data = df[right_cols].values
    
    temp = np.abs(left_data - right_data)
    newnames = [col.replace(leftname, replacer) for col in left_cols]
    
    for i, col_name in enumerate(newnames):
        df[col_name] = temp[:, i]
        
    return df

def mapLRAverageVar(
    mydataframe: pd.DataFrame,
    leftvar: List[str],
    leftname: str = "left",
    rightname: str = "right",
    replacer: str = "LRAVG"
) -> pd.DataFrame:
    """
    Compute average of left and right variables.
    """
    df = mydataframe.copy()
    rightvar = [col.replace(leftname, rightname) for col in leftvar]
    cols_set = set(df.columns)
    valid_indices = [i for i, r_col in enumerate(rightvar) if r_col in cols_set]
    
    if not valid_indices:
        return df
        
    left_cols = [leftvar[i] for i in valid_indices]
    right_cols = [rightvar[i] for i in valid_indices]
    
    left_data = df[left_cols].values
    right_data = df[right_cols].values
    
    temp = (left_data + right_data) * 0.5
    newnames = [col.replace(leftname, replacer) for col in left_cols]
    
    for i, col_name in enumerate(newnames):
        df[col_name] = temp[:, i]
        
    return df

def antspymm_predictors(
    demog: pd.DataFrame,
    doasym: bool = False,
    return_colnames: bool = True
) -> Union[List[str], pd.DataFrame]:
    """
    Identify and extract ANTsPyMM predictor columns from a dataframe,
    optionally computing hemisphere asymmetry and average columns.

    Parameters
    ----------
    demog : pd.DataFrame
        The input DataFrame containing neuroimaging feature columns.
    doasym : bool, default=False
        Whether to calculate asymmetry (Left - Right) variables.
    return_colnames : bool, default=True
        If True, return a list of selected column names.
        If False, return the modified DataFrame with L/R averages and asymmetries.

    Returns
    -------
    Union[List[str], pd.DataFrame]
        List of selected column names or the modified DataFrame.
    """
    # 1. badcaud
    badcaud = _get_names_from_dataframe("bn_str_ca", demog)
    badcaud = [name for name in badcaud if not re.search("deep", name)]
    
    # 2. exclusions list (xcl)
    xcl = [
        "hier_id", "background", "SNR", "evr", "mask", "msk", 
        "smoothing", "minutes", "RandBasis", "templateL1", "upsampl", 
        "paramset", "nc_wm", "nc_csf", "censor", "bandpass", 
        "outlier", "meanBold", "dimensionality", "spc", "org", 
        "andwidth", "unclassified", "cleanup", "slice"
    ] + badcaud + [
        "dimx", "dimy", "dimz", "dimt", "modality"
    ]
    
    if doasym and return_colnames:
        xcl = xcl + ["left", "right", "_l_", "_r_"]
        
    # 3. t1namesbst
    t1namesbst_all = _get_names_from_dataframe(["T1Hier", "brainstem", "vol"], demog, exclusions=["tissues", "lobes"])
    t1namesbst = t1namesbst_all[1:] if len(t1namesbst_all) > 0 else []
    
    # 4. Assemble predictor names (testnames)
    testnames = []
    
    # T1w_
    testnames.extend(_get_names_from_dataframe("T1w_", demog, exclusions=xcl))
    
    # mtl
    testnames.extend(_get_names_from_dataframe("mtl", demog, exclusions=xcl))
    
    # cerebellum
    testnames.extend(_get_names_from_dataframe("cerebellum", demog, exclusions=xcl + ["_cerebell"]))
    
    # T1Hier_
    t1hier_exclusions = [
        "hier_id", r"[.]1", r"[.]2", r"[.]3", "background", "tissue", 
        "dktregions", "T1Hier_resnetGrade", "hemisphere", "lobes", 
        "SNR", "evr", "area"
    ] + xcl
    testnames.extend(_get_names_from_dataframe("T1Hier_", demog, exclusions=t1hier_exclusions))
    
    # t1namesbst
    testnames.extend(t1namesbst)
    
    # rsfMRI_fcnxpro
    rsfmri_exclusions = [
        "hier_id", "background", "thk", "area", "vol", "FD", "dvars", 
        "ssnr", "tsnr", "motion", "SNR", "evr", "_alff", "falff_sd", 
        "falff_mean"
    ] + xcl
    testnames.extend(_get_names_from_dataframe("rsfMRI_fcnxpro", demog, exclusions=rsfmri_exclusions))
    
    # perf_
    perf_exclusions = [
        "hier_id", "background", "thk", "area", "vol", "FD", "dvars", 
        "ssnr", "tsnr", "motion", "SNR", "evr", "_alff", "falff_sd", 
        "falff_mean"
    ] + xcl
    testnames.extend(_get_names_from_dataframe("perf_", demog, exclusions=perf_exclusions))
    
    # DTI_
    dti_exclusions = [
        "hier_id", "background", "thk", "area", "vol", "motion", "FD", 
        "dvars", "ssnr", "tsnr", "SNR", "evr", "cnx", "relcn"
    ] + xcl
    testnames.extend(_get_names_from_dataframe("DTI_", demog, exclusions=dti_exclusions))
    
    # 5. Make unique and intersect with dataframe columns
    seen = set()
    testnames_unique = []
    for name in testnames:
        if name not in seen:
            seen.add(name)
            testnames_unique.append(name)
            
    cols_set = set(demog.columns)
    testnames = [name for name in testnames_unique if name in cols_set]
    
    if return_colnames:
        return testnames
        
    # 6. Modify dataframe
    demog_mod = demog.copy()
    
    # Rename columns to standardized lowercase left/right
    new_cols = []
    for col in demog_mod.columns:
        new_col = col.replace("Right", "right").replace("Left", "left")
        new_cols.append(new_col)
    demog_mod.columns = new_cols
    
    # Update selected testnames to match renamed columns
    testnames = [name.replace("Right", "right").replace("Left", "left") for name in testnames]
    
    # Apply Asymmetry and Average mappings
    l_r_names = [name for name in testnames if "_l_" in name]
    if doasym:
        demog_mod = mapAsymVar(demog_mod, l_r_names, "_l_", "_r_")
    demog_mod = mapLRAverageVar(demog_mod, l_r_names, "_l_", "_r_")
    
    left_right_names = [name for name in testnames if "left" in name]
    if doasym:
        demog_mod = mapAsymVar(demog_mod, left_right_names, "left", "right")
    demog_mod = mapLRAverageVar(demog_mod, left_right_names, "left", "right")
    
    return demog_mod

def nnh_update_residuals(mat: torch.Tensor, 
                         covariate_data: pd.DataFrame, 
                         covariate_cols: Union[str, List[str]]) -> torch.Tensor:
    """
    Residualize a data matrix against one or more covariates.

    Removes the linear influence of covariates (e.g., age, sex, or mean signal) 
    from the modality features using joint linear regression.

    Parameters
    ----------
    mat : torch.Tensor
        The input data matrix to be residualized.
    covariate_data : pd.DataFrame
        DataFrame containing the covariate values.
    covariate_cols : Union[str, List[str]]
        The name of the column, list of columns, or formula string (e.g., 'age + sex') 
        in `covariate_data` to use as regressors. If 'mean', residualizes against the 
        row-wise mean of `mat`.

    Returns
    -------
    torch.Tensor
        The residualized data matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    mat = torch.as_tensor(mat).float()
    if isinstance(covariate_cols, str) and covariate_cols == "mean":
        mymean = torch.mean(mat, dim=1, keepdim=True)
        cov_vals = mymean.numpy()
    else:
        if isinstance(covariate_cols, str):
            if "+" in covariate_cols:
                cols = [c.strip() for c in covariate_cols.split('+')]
            else:
                cols = [covariate_cols]
        else:
            cols = list(covariate_cols)
        cov_vals = covariate_data[cols].values
        
    mat_np = mat.numpy()
    residuals = np.zeros_like(mat_np)
    
    for i in range(mat_np.shape[1]):
        y = mat_np[:, i]
        reg = LinearRegression().fit(cov_vals, y)
        residuals[:, i] = y - reg.predict(cov_vals)
        
    return torch.as_tensor(residuals).float()

def nnh_embed(blaster: pd.DataFrame,
              select_training_boolean: Optional[np.ndarray] = None,
              connect_cog: Optional[List[str]] = None,
              energy: str = "acc",
              nsimlr: int = 5,
              constraint: str = "ortho",
              covariates: List[str] = [],
              do_asym: int = 1,
              resnet_grade_thresh: float = 1.02,
              use_flow: bool = False,
              verbose: bool = False,
              **simlr_kwargs) -> Dict[str, Any]:
    """
    NNHEmbed: Perform SiMLR or Flow-SiMLR Analysis on Multimodal Neuroimaging Data.

    This function automates the preprocessing, modality grouping, and 
    SiMLR/Flow-SiMLR embedding for complex neuroimaging datasets (e.g., ANTsPyMM 
    outputs). It handles asymmetry, covariate correction, and 
    automated feature selection.

    Parameters
    ----------
    blaster : pd.DataFrame
        The input DataFrame containing subjects and multimodal features.
    select_training_boolean : Optional[np.ndarray], default=None
        Boolean mask to select training samples.
    connect_cog : Optional[List[str]], default=None
        List of cognitive/outcome variables to include in the embedding.
    energy : str, default="acc"
        Energy function type for SiMLR.
    nsimlr : int, default=5
        The rank (K) of the latent embedding.
    constraint : str, default="ortho"
        Orthogonality constraint for SiMLR.
    covariates : List[str], default=[]
        Columns to use as covariates for residualization (e.g., 'age', 'sex').
    do_asym : int, default=1
        Whether to compute asymmetry features (left - right).
    resnet_grade_thresh : float, default=1.02
        Quality control threshold for neuroimaging features.
    use_flow : bool, default=False
        If True, run Flow-SiMLR (flow_simr_v) instead of linear SiMLR.
    verbose : bool, default=False
        Whether to print progress messages.
    **simlr_kwargs : Dict[str, Any]
        Additional arguments passed to the underlying SiMLR / Flow-SiMLR function.

    Returns
    -------
    Dict[str, Any]
        The result dictionary, containing 'u', 'v', 'w', and metadata.

    Raises
    ------
    ValueError
        If there are too few rows after filtering or quality control.
    TypeError
        If inputs are of invalid types.
```
# The rest of the docstring and comments ...
```python
    """
    if verbose:
        print("Step 1: Selecting and filtering features...")
        
    idps = antspymm_predictors(blaster)
    
    # Filter nuisance and specific patterns
    exclude_patterns = ["cleanup", "snseg", "_deep_", "peraf", "alff", "LRAVGcit168", "_l_", "_r_"]
    idps = [x for x in idps if not any(p in x for p in exclude_patterns)]
    
    if do_asym == 0:
        idps = [x for x in idps if "Asym" not in x]
    else:
        idps = [x for x in idps if "Asymait168" not in x]
        
    # Assemble modalities
    idplist = {}
    idplist["t1"] = [x for x in idps if "T1Hier" in x]
    idplist["dt"] = [x for x in idps if "DTI" in x or "mean_fa" in x or "mean_md" in x]
    idplist["rsf"] = [x for x in idps if "rsfMRI" in x]
    idplist["perf"] = [x for x in idps if "perf_cbf" in x]
    
    if connect_cog:
        idplist["cg"] = connect_cog
        
    # Clean empty lists
    idplist = {k: v for k, v in idplist.items() if len(v) > 0}
    
    if verbose:
        print(f"Assembled {len(idplist)} feature modalities: {list(idplist.keys())}")

    # Step 2: Quality Control
    if "T1Hier_resnetGrade" in blaster.columns:
        qc_mask = blaster["T1Hier_resnetGrade"] >= resnet_grade_thresh
    else:
        qc_mask = pd.Series([True] * len(blaster))
        
    if select_training_boolean is not None:
        final_mask = qc_mask & pd.Series(select_training_boolean)
    else:
        final_mask = qc_mask
        
    blaster_sub = blaster[final_mask]
    
    if len(blaster_sub) < 5:
        raise ValueError("Too few rows after quality control filtering.")

    # Step 3: Matrix Assembly and Imputation
    mats = {}
    imputer = SimpleImputer(strategy='mean')
    
    for name, cols in idplist.items():
        data = blaster_sub[cols].values
        # Impute missing values
        data_imputed = imputer.fit_transform(data)
        mats[name] = torch.as_tensor(data_imputed).float()
        
    # Step 4: Covariate Adjustment
    if covariates:
        if verbose:
            print(f"Adjusting modalities by covariates: {covariates}")
        for name in mats:
            mats[name] = nnh_update_residuals(mats[name], blaster_sub, covariates)
            
    # Step 5: Run SIMLR or Flow-SIMLR
    mat_list = list(mats.values())
    
    if use_flow:
        from .flows import flow_simr_v
        import inspect
        
        flow_kwargs = simlr_kwargs.copy()
        
        # Map iterations to epochs
        if 'iterations' in flow_kwargs and 'epochs' not in flow_kwargs:
            flow_kwargs['epochs'] = flow_kwargs.pop('iterations')
            
        # Map energy_type
        if 'energy_type' not in flow_kwargs:
            if energy == "acc":
                flow_kwargs['energy_type'] = 'regression'
            else:
                flow_kwargs['energy_type'] = energy
                
        # Map retraction_type from constraint
        if 'retraction_type' not in flow_kwargs:
            if "polar" in constraint:
                flow_kwargs['retraction_type'] = "soft_polar"
            elif "ns" in constraint or constraint in ["ortho", "nsaflow"]:
                flow_kwargs['retraction_type'] = "soft_ns"
            else:
                flow_kwargs['retraction_type'] = "soft_polar"
                
        # Map positivity (defaults to 'either' to align with linear simlr defaults)
        if 'positivity' not in flow_kwargs:
            flow_kwargs['positivity'] = 'either'
            
        # Filter valid parameters for flow_simr_v dynamically
        flow_sig = inspect.signature(flow_simr_v)
        valid_flow_params = set(flow_sig.parameters.keys())
        flow_kwargs = {k: v for k, v in flow_kwargs.items() if k in valid_flow_params}
        
        if verbose:
            print(f"Running Flow-SiMLR with k={nsimlr}...")
            
        result = flow_simr_v(
            data_matrices=mat_list,
            k=nsimlr,
            verbose=verbose,
            **flow_kwargs
        )
        
        # Ensure 'w' matrix is present for output compatibility
        if 'w' not in result:
            u_val = result["u"]
            w_mats = []
            for x in mat_list:
                try:
                    u_pinv = torch.linalg.pinv(u_val)
                    w_mats.append(u_pinv @ x)
                except:
                    w_mats.append(torch.zeros(nsimlr, x.shape[1], dtype=u_val.dtype, device=u_val.device))
            result["w"] = w_mats
            
    else:
        if verbose:
            print(f"Running SIMLR with k={nsimlr}...")
            
        result = simlr(
            data_matrices=mat_list,
            k=nsimlr,
            energy_type=energy,
            verbose=verbose,
            **simlr_kwargs
        )
    
    # Add metadata
    result["modality_names"] = list(mats.keys())
    result["feature_names"] = idplist
    
    return result
