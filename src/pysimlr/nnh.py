import torch
import pandas as pd
import numpy as np
import re
import copy
from typing import List, Optional, Union, Dict, Any, Tuple
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


def _shorten_pymm_names(names: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Shorten ANTsPyMM column names to standardized forms to improve matching.
    """
    is_single = isinstance(names, str)
    name_list = [names] if is_single else list(names)
    
    def shorten_single(nm: str) -> str:
        def shorten_nm_names(n: str) -> str:
            n = n.replace("nm2dmt.nm.", "nm.")
            n = n.replace(".avg.", ".iavg.")
            n = n.replace("intmean", "iavg")
            n = n.replace("intsum", "isum")
            n = n.replace("volume", "vol")
            n = n.replace("substantianigra", "sn")
            return n
            
        n = nm.lower().replace("_", ".")
        while ".." in n:
            n = n.replace("..", ".")
        n = n.replace("sagittal.stratum.include.inferior.longitidinal.fasciculus.and.inferior.fronto.occipital.fasciculus.", "ilf.and.ifo")
        n = n.replace(".cres.stria.terminalis.can.not.be.resolved.with.current.resolution.", "")
        n = n.replace("longitudinal.fasciculus", "l.fasc")
        n = n.replace("corona.radiata", "cor.rad")
        n = n.replace("central", "cent")
        n = n.replace("deep.cit168", "dp.")
        n = n.replace("cit168", "")
        n = n.replace(".include", "")
        n = n.replace("mtg.sn", "")
        n = n.replace("brainstem", ".bst")
        n = n.replace("rsfmri.", "rsf.")
        n = n.replace("dti.mean.fa.", "dti.fa.")
        n = n.replace("perf.cbf.mean.", "cbf.")
        n = n.replace(".jhu.icbm.labels.1mm", "")
        n = n.replace(".include.optic.radiation.", "")
        while ".." in n:
            n = n.replace("..", ".")
        n = n.replace("cerebellar.peduncle", "cereb.ped")
        n = n.replace("anterior.limb.of.internal.capsule", "ant.int.cap")
        n = n.replace("posterior.limb.of.internal.capsule", "post.int.cap")
        n = n.replace("t1hier.", "t1.")
        n = n.replace("anterior", "ant")
        n = n.replace("posterior", "post")
        n = n.replace("inferior", "inf")
        n = n.replace("superior", "sup")
        n = n.replace("dktcortex", ".ctx")
        n = n.replace(".lravg", "")
        n = n.replace("dti.mean.fa", "dti.fa")
        n = n.replace("retrolenticular.part.of.internal", "rent.int.cap")
        n = n.replace("iculus.could.be.a.part.of.ant.internal.capsule", "")
        n = n.replace(".fronto.occipital.", ".frnt.occ.")
        n = n.replace(".longitidinal.fasciculus.", ".long.fasc.")
        n = n.replace(".external.capsule", ".ext.cap")
        n = n.replace("of.internal.capsule", ".int.cap")
        n = n.replace("fornix.cres.stria.terminalis", "fornix.")
        n = n.replace("capsule", "")
        n = n.replace("and.inf.frnt.occ.fasciculus.", "")
        n = n.replace("crossing.tract.a.part.of.mcp.", "")
        n = n.replace("post.thalamic.radiation.optic.radiation", "post.thalamic.radiation")
        n = n.replace("adjusted", "adj")
        while ".." in n:
            n = n.replace("..", ".")
        n = n.replace("t1w.mean", "t1vth")
        n = n.replace("fcnxpro129", "p2")
        n = n.replace("fcnxpro134", "p3")
        n = n.replace("fcnxpro122", "p1")
        n = shorten_nm_names(n)
        return n

    shortened = [shorten_single(nm) for nm in name_list]
    return shortened[0] if is_single else shortened


def apply_simlr_matrices(
    existing_df: pd.DataFrame,
    simlr_v: Union[List[Union[pd.DataFrame, np.ndarray, torch.Tensor]], Dict[str, Union[pd.DataFrame, np.ndarray, torch.Tensor]]],
    feature_names: Optional[Union[List[List[str]], Dict[str, List[str]]]] = None,
    n_limit: Optional[int] = None,
    version_prefix: str = "r",
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Project data using SiMLR projection matrices and append the columns.
    """
    extended_df = existing_df.copy()
    added_names = []
    used_names = set(extended_df.columns)
    
    simlr_v_dict = {}
    if isinstance(simlr_v, dict):
        simlr_v_dict = simlr_v
    else:
        for i, W in enumerate(simlr_v):
            simlr_v_dict[f"block{i}"] = W
            
    feature_names_dict = {}
    if feature_names is not None:
        if isinstance(feature_names, dict):
            feature_names_dict = feature_names
        else:
            for i, names_list in enumerate(feature_names):
                feature_names_dict[f"block{i}"] = names_list
                
    for block_name, W in simlr_v_dict.items():
        if isinstance(W, pd.DataFrame):
            row_names = W.index.tolist()
            col_names = W.columns.tolist()
            W_values = W.values
        else:
            if block_name not in feature_names_dict:
                raise ValueError(f"Feature names list is missing for block '{block_name}'")
            row_names = feature_names_dict[block_name]
            if torch.is_tensor(W):
                W_values = W.detach().cpu().numpy()
            else:
                W_values = np.asarray(W)
            col_names = [f"PC{i+1}" for i in range(W_values.shape[1])]
            
        if n_limit is not None and W_values.shape[1] > n_limit:
            W_values = W_values[:, :n_limit]
            col_names = col_names[:n_limit]
            
        # Find overlap
        overlap = [name for name in row_names if name in used_names]
        if not overlap:
            if verbose:
                print(f"Warning: No overlapping features for block '{block_name}' - skipping")
            continue
            
        overlap_indices = [row_names.index(name) for name in overlap]
        W_subset = W_values[overlap_indices, :]
        
        X = extended_df[overlap].values
        X_clean = np.nan_to_num(X)
        Y = X_clean @ W_subset
        
        final_names = []
        for cname in col_names:
            clean_cname = cname
            if clean_cname.startswith(block_name):
                clean_cname = clean_cname[len(block_name):]
            base_name = f"{block_name}{clean_cname}"
            
            candidate = base_name
            major = 0
            minor = 0
            while candidate in used_names:
                minor += 1
                if minor > 9:
                    minor = 1
                    major += 1
                version_tag = f"{version_prefix}{major:02d}."
                match = re.search(r"(\d+)$", base_name)
                if match:
                    digits = match.group(1)
                    candidate = base_name[:-len(digits)] + version_tag + digits
                else:
                    candidate = base_name + version_tag
            final_names.append(candidate)
            used_names.add(candidate)
            
        for j, fname in enumerate(final_names):
            extended_df[fname] = Y[:, j]
            added_names.append(fname)
            
    return extended_df, added_names


def apply_simlr_matrices_dtfix(
    existing_df: pd.DataFrame,
    matrices_list: Union[List[Union[pd.DataFrame, np.ndarray, torch.Tensor]], Dict[str, Union[pd.DataFrame, np.ndarray, torch.Tensor]]],
    feature_names: Optional[Union[List[List[str]], Dict[str, List[str]]]] = None,
    n_limit: Optional[int] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply SIMLR projection matrices with potential DTI column name shortening fix.
    """
    existing_df_fix = existing_df.copy()
    existing_df_cols = existing_df_fix.columns.tolist()
    dti_cols = [c for c in existing_df_cols if "DTI_" in c]
    
    matrices_list_fix = {}
    if isinstance(matrices_list, dict):
        for k, W in matrices_list.items():
            if isinstance(W, pd.DataFrame):
                matrices_list_fix[k] = W.copy()
            else:
                matrices_list_fix[k] = W
    else:
        for i, W in enumerate(matrices_list):
            matrices_list_fix[f"block{i}"] = W
            
    feature_names_fix = {}
    if feature_names is not None:
        if isinstance(feature_names, dict):
            feature_names_fix = copy.deepcopy(feature_names)
        else:
            for i, names_list in enumerate(feature_names):
                feature_names_fix[f"block{i}"] = list(names_list)
                
    if dti_cols:
        shortened_existing_df_cols = _shorten_pymm_names(dti_cols)
        dt_correspondence = False
        dta_correspondence = False
        
        for block_key in list(matrices_list_fix.keys()):
            is_dt = (block_key == "dt") or (block_key.startswith("block") and feature_names_fix.get(block_key) and any("DTI_" in n for n in feature_names_fix[block_key]))
            if is_dt:
                W_dt = matrices_list_fix[block_key]
                if isinstance(W_dt, pd.DataFrame):
                    dt_row_names = W_dt.index.tolist()
                    shortened_dt_row_names = _shorten_pymm_names(dt_row_names)
                    orig_match = sum(1 for c in dti_cols if c in dt_row_names)
                    short_match = sum(1 for c in shortened_existing_df_cols if c in shortened_dt_row_names)
                    if short_match > orig_match:
                        dt_correspondence = True
                        W_dt.index = shortened_dt_row_names
                else:
                    if block_key in feature_names_fix:
                        dt_row_names = feature_names_fix[block_key]
                        shortened_dt_row_names = _shorten_pymm_names(dt_row_names)
                        orig_match = sum(1 for c in dti_cols if c in dt_row_names)
                        short_match = sum(1 for c in shortened_existing_df_cols if c in shortened_dt_row_names)
                        if short_match > orig_match:
                            dt_correspondence = True
                            feature_names_fix[block_key] = shortened_dt_row_names
                            
        for block_key in list(matrices_list_fix.keys()):
            is_dta = (block_key == "dta")
            if is_dta:
                W_dta = matrices_list_fix[block_key]
                if isinstance(W_dta, pd.DataFrame):
                    dta_row_names = W_dta.index.tolist()
                    shortened_dta_row_names = _shorten_pymm_names(dta_row_names)
                    orig_match = sum(1 for c in dti_cols if c in dta_row_names)
                    short_match = sum(1 for c in shortened_existing_df_cols if c in shortened_dta_row_names)
                    if short_match > orig_match:
                        dta_correspondence = True
                        W_dta.index = shortened_dta_row_names
                else:
                    if block_key in feature_names_fix:
                        dta_row_names = feature_names_fix[block_key]
                        shortened_dta_row_names = _shorten_pymm_names(dta_row_names)
                        orig_match = sum(1 for c in dti_cols if c in dta_row_names)
                        short_match = sum(1 for c in shortened_existing_df_cols if c in shortened_dta_row_names)
                        if short_match > orig_match:
                            dta_correspondence = True
                            feature_names_fix[block_key] = shortened_dta_row_names
                            
        if dt_correspondence or dta_correspondence:
            if verbose:
                print("Shortened names improve DTI correspondence. Applying shortened names to DataFrame...")
            col_map = {orig: short for orig, short in zip(dti_cols, shortened_existing_df_cols)}
            existing_df_fix = existing_df_fix.rename(columns=col_map)
            
    new_df, added_names = apply_simlr_matrices(
        existing_df_fix,
        matrices_list_fix,
        feature_names=feature_names_fix,
        n_limit=n_limit,
        verbose=verbose
    )
    
    if dti_cols and (dt_correspondence or dta_correspondence):
        col_map_restore = {short: orig for orig, short in zip(dti_cols, shortened_existing_df_cols)}
        new_df = new_df.rename(columns=col_map_restore)
        
    return new_df, added_names


def _preprocess_matrix(M: np.ndarray, block_name: str, preprocess_policy: str = "drop", verbose: bool = False) -> Dict[str, Any]:
    """
    Remove invalid features, center, scale, and replace NaNs with zero.
    """
    n_samples, n_features = M.shape
    all_na = np.all(np.isnan(M), axis=0)
    
    zero_var = np.zeros(n_features, dtype=bool)
    for col in range(n_features):
        col_vals = M[:, col]
        valid_vals = col_vals[np.isfinite(col_vals)]
        if len(valid_vals) <= 1 or np.std(valid_vals) == 0:
            zero_var[col] = True
            
    invalid_cols = all_na | zero_var
    dropped = []
    zeroed = []
    
    if np.any(invalid_cols):
        bad_indices = np.where(invalid_cols)[0]
        if preprocess_policy == "drop":
            if verbose:
                print(f"Dropping {len(bad_indices)} invalid columns from block '{block_name}'")
            M = np.delete(M, bad_indices, axis=1)
            dropped = list(bad_indices)
        elif preprocess_policy == "zero":
            if verbose:
                print(f"Zeroing {len(bad_indices)} invalid columns in block '{block_name}'")
            M = M.copy()
            M[:, invalid_cols] = 0.0
            zeroed = list(bad_indices)
            
    if M.shape[1] < 1:
        raise ValueError(f"Block '{block_name}' has no usable columns after preprocessing.")
        
    mean = np.mean(M, axis=0)
    std = np.std(M, axis=0, ddof=1)
    std[std == 0] = 1.0
    
    M_scaled = (M - mean) / std
    M_scaled[~np.isfinite(M_scaled)] = 0.0
    
    return {"matrix": M_scaled, "dropped": dropped, "zeroed": zeroed}


def _build_feature_correlation_adjacency(M: np.ndarray, threshold: float = 0.8, use_abs: bool = True) -> np.ndarray:
    """
    Build feature correlation adjacency matrix.
    """
    if M.shape[1] == 1:
        return np.ones((1, 1))
        
    df_temp = pd.DataFrame(M)
    C = df_temp.corr().values.copy()
    C[~np.isfinite(C)] = 0.0
    
    if use_abs:
        A = np.abs(C)
    else:
        A = C.copy()
        
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 1.0)
    return A


def _compute_recommended_k(M: np.ndarray, method: str = "cumulative", cumulative_thresh: float = 0.9, min_k: int = 2) -> int:
    """
    Compute recommended rank k using cumulative variance or elbow method.
    """
    max_k = max(1, min(M.shape[0] - 1, M.shape[1]))
    min_k_eff = min(max_k, min_k)
    
    if M.shape[1] <= 1 or M.shape[0] <= 2:
        return min_k_eff
        
    try:
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        sdsq = S**2
    except:
        return min_k_eff
        
    if len(sdsq) < 1 or np.sum(sdsq) <= 0:
        return min_k_eff
        
    if method == "cumulative":
        prop = sdsq / np.sum(sdsq)
        cum_prop = np.cumsum(prop)
        indices = np.where(cum_prop >= cumulative_thresh)[0]
        if len(indices) > 0:
            k = indices[0] + 1
        else:
            k = len(prop)
    else:
        if len(sdsq) < 2:
            k = min_k_eff
        else:
            eps = np.finfo(float).eps
            ratios = sdsq[:-1] / np.maximum(sdsq[1:], eps)
            k = int(np.argmax(ratios)) + 1
            
    return int(max(min_k_eff, min(k, max_k)))


def _resolve_k_new(k_new, new_modality_names, blocks, k_method, cumvar_threshold, min_k):
    out = {}
    if k_new is None:
        for nm in new_modality_names:
            out[nm] = _compute_recommended_k(blocks[nm], method=k_method, cumulative_thresh=cumvar_threshold, min_k=min_k)
        return out
        
    if isinstance(k_new, (int, float, np.integer)):
        for nm in new_modality_names:
            out[nm] = int(k_new)
        return out
        
    for nm in new_modality_names:
        if isinstance(k_new, dict) and nm in k_new and k_new[nm] is not None:
            out[nm] = int(k_new[nm])
        else:
            out[nm] = _compute_recommended_k(blocks[nm], method=k_method, cumulative_thresh=cumvar_threshold, min_k=min_k)
    return out


def _combine_joint_k(k_dict, policy="max", min_k=2):
    vals = [v for v in k_dict.values() if np.isfinite(v) and v >= 1]
    if not vals:
        return min_k
    if policy == "max":
        out = max(vals)
    elif policy == "median":
        out = int(np.median(vals))
    elif policy == "min":
        out = min(vals)
    else:
        out = max(vals)
    return int(max(out, min_k))


def _detect_prefixes(simnames: List[str]) -> List[str]:
    prefixes = []
    for name in simnames:
        match = re.match(r"^(.+?)(?:PC|C)\d+$", name)
        if match:
            prefixes.append(match.group(1))
    return list(set(prefixes))


def extend_simlr_embedding_with_new_modalities(
    pymm: pd.DataFrame,
    simlr_result: Dict[str, Any],
    new_modalities: Dict[str, List[str]],
    mode: str = "concatenate",
    split_prefixes: List[str] = ["t1", "dt", "rsf"],
    adjacency_type: str = "feature_correlation",
    cor_threshold: float = 0.8,
    use_abs_correlation: bool = True,
    k_new: Optional[Union[int, Dict[str, int]]] = None,
    k_method: str = "cumulative",
    cumvar_threshold: float = 0.9,
    min_k: int = 2,
    joint_k_policy: str = "max",
    preprocess: str = "drop",
    method: str = "simlr",
    feature_names: Optional[Union[List[List[str]], Dict[str, List[str]]]] = None,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Extend an existing SiMLR embedding with new modalities.

    Parameters
    ----------
    pymm : pd.DataFrame
        DataFrame containing subject demographic/imaging variables.
    simlr_result : Dict[str, Any]
        The result dictionary from an existing SiMLR run.
    new_modalities : Dict[str, List[str]]
        Dictionary of new modality names and their column names.
    mode : str, default="concatenate"
        How to handle existing projections ("concatenate", "split", or "auto").
    split_prefixes : List[str], default=["t1", "dt", "rsf"]
        Prefixes of existing projections to split on.
    adjacency_type : str, default="feature_correlation"
        Regularization matrix strategy.
    cor_threshold : float, default=0.8
        Threshold for adjacency matrix correlation.
    use_abs_correlation : bool, default=True
        Whether to use absolute correlation values.
    k_new : Union[int, Dict[str, int]], optional
        Rank/dimensionality of new modalities.
    k_method : str, default="cumulative"
        Rank determination strategy ("cumulative" or "elbow").
    cumvar_threshold : float, default=0.9
        PCA cumulative variance threshold.
    min_k : int, default=2
        Minimum modality k rank.
    joint_k_policy : str, default="max"
        How to combine modality ranks ("max", "median", "min").
    preprocess : str, default="drop"
        How to handle invalid variables ("drop" or "zero").
    method : str, default="simlr"
        pysimlr method to use ("simlr", "flow_simr_v", or "lend_simr").
    feature_names : Union[List[List[str]], Dict[str, List[str]]], optional
        Feature names of the original simlr run.
    verbose : bool, default=False
        Whether to print diagnostic messages.
    **kwargs : dict
        Additional parameters passed to the underlying pysimlr method.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the updated SiMLR result, blocks, and diagnostics.
    """
    if mode not in ["concatenate", "split", "auto"]:
        raise ValueError("mode must be one of 'concatenate', 'split', 'auto'")
    if adjacency_type not in ["feature_correlation"]:
        raise ValueError("adjacency_type must be 'feature_correlation'")
    if k_method not in ["cumulative", "elbow"]:
        raise ValueError("k_method must be one of 'cumulative', 'elbow'")
    if joint_k_policy not in ["max", "median", "min"]:
        raise ValueError("joint_k_policy must be one of 'max', 'median', 'min'")
    if preprocess not in ["drop", "zero"]:
        raise ValueError("preprocess must be one of 'drop', 'zero'")
    if not isinstance(new_modalities, dict) or not new_modalities:
        raise ValueError("new_modalities must be a non-empty dictionary mapping names to column lists")
    for nm, cols in new_modalities.items():
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError(f"new_modalities['{nm}'] must be a list of strings")
            
    if "v" not in simlr_result:
        raise ValueError("simlr_result must contain 'v' (projection matrices)")
        
    feat_names_arg = feature_names
    if feat_names_arg is None:
        if "feature_names" in simlr_result:
            feat_names_arg = simlr_result["feature_names"]
        elif "modality_names" in simlr_result and isinstance(simlr_result.get("v"), dict):
            feat_names_arg = {k: v.index.tolist() for k, v in simlr_result["v"].items() if isinstance(v, pd.DataFrame)}
            
    pymm_proj, simnames = apply_simlr_matrices_dtfix(
        pymm,
        simlr_result["v"],
        feature_names=feat_names_arg,
        verbose=verbose
    )
    
    if mode == "auto":
        detected = _detect_prefixes(simnames)
        if not detected:
            raise ValueError("Could not auto-detect prefixes from simnames; use 'concatenate' or provide 'split_prefixes'.")
        split_prefixes = detected
        if verbose:
            print(f"Auto-detected split prefixes: {split_prefixes}")
            
    new_cols_flat = []
    for cols in new_modalities.values():
        new_cols_flat.extend(cols)
        
    overlap_existing = set(new_cols_flat).intersection(set(simnames))
    if overlap_existing:
        raise ValueError(f"New modality columns overlap existing SIMLR projection columns: {overlap_existing}")
        
    missing_new = set(new_cols_flat).difference(set(pymm_proj.columns))
    if missing_new:
        raise ValueError(f"new_modalities contains columns not found in projected data: {missing_new}")
        
    blocks_raw = {}
    block_features = {}
    
    if mode == "concatenate":
        blocks_raw["sim"] = pymm_proj[simnames].values
        block_features["sim"] = list(simnames)
    else:
        for pref in split_prefixes:
            keep = [name for name in simnames if name.startswith(pref)]
            if keep:
                blocks_raw[pref] = pymm_proj[keep].values
                block_features[pref] = keep
            elif verbose:
                print(f"Prefix '{pref}' not found among projected SIMLR names; skipping.")
                
        if not blocks_raw:
            raise ValueError("No existing SIMLR blocks were constructed. Check 'split_prefixes' or use 'concatenate'.")
            
    for mod, cols in new_modalities.items():
        blocks_raw[mod] = pymm_proj[cols].values
        block_features[mod] = list(cols)
        
    n_subjects = len(pymm_proj)
    for nm, mat in blocks_raw.items():
        if mat.shape[0] != n_subjects:
            raise ValueError(f"Block '{nm}' has {mat.shape[0]} rows; expected {n_subjects}")
            
    blocks = {}
    block_features_preprocessed = {}
    preprocessing_diagnostics = {}
    
    for nm, mat in blocks_raw.items():
        prep = _preprocess_matrix(mat, nm, preprocess_policy=preprocess, verbose=verbose)
        blocks[nm] = prep["matrix"]
        
        original_cols = block_features[nm]
        if preprocess == "drop":
            keep_indices = [i for i in range(len(original_cols)) if i not in prep["dropped"]]
            preprocessed_cols = [original_cols[i] for i in keep_indices]
        else:
            preprocessed_cols = list(original_cols)
            
        block_features_preprocessed[nm] = preprocessed_cols
        preprocessing_diagnostics[nm] = {
            "dropped_columns": [original_cols[i] for i in prep["dropped"]],
            "zeroed_columns": [original_cols[i] for i in prep["zeroed"]],
            "n_rows": prep["matrix"].shape[0],
            "n_cols": prep["matrix"].shape[1],
            "colnames": preprocessed_cols
        }
        
    adjacency = {}
    for nm, mat in blocks.items():
        adjacency[nm] = _build_feature_correlation_adjacency(
            mat,
            threshold=cor_threshold,
            use_abs=use_abs_correlation
        )
        
    new_modality_names = list(new_modalities.keys())
    k_new_used = _resolve_k_new(
        k_new=k_new,
        new_modality_names=new_modality_names,
        blocks=blocks,
        k_method=k_method,
        cumvar_threshold=cumvar_threshold,
        min_k=min_k
    )
    
    joint_k = _combine_joint_k(k_new_used, policy=joint_k_policy, min_k=min_k)
    if verbose:
        k_used_str = "; ".join(f"{k}={v}" for k, v in k_new_used.items())
        print(f"k_new used: {k_used_str} | joint_k={joint_k}")
        
    block_names_ordered = list(blocks.keys())
    mat_list = [blocks[name] for name in block_names_ordered]
    adj_list = [adjacency[name] for name in block_names_ordered]
    
    method_kwargs = copy.deepcopy(kwargs)
    if method in ["flow_simr_v", "lend_simr"]:
        if "iterations" in method_kwargs:
            method_kwargs["epochs"] = method_kwargs.pop("iterations")
    elif method == "simlr":
        if "epochs" in method_kwargs:
            method_kwargs["iterations"] = method_kwargs.pop("epochs")

    if method == "simlr":
        simlr_fit = simlr(
            data_matrices=mat_list,
            k=joint_k,
            smoothing_matrices=[torch.tensor(adj).float() for adj in adj_list],
            verbose=verbose,
            **method_kwargs
        )
    elif method == "flow_simr_v":
        from .flows import flow_simr_v
        simlr_fit = flow_simr_v(
            data_matrices=mat_list,
            k=joint_k,
            verbose=verbose,
            **method_kwargs
        )
    elif method == "lend_simr":
        from .deep import lend_simr
        simlr_fit = lend_simr(
            data_matrices=mat_list,
            k=joint_k,
            verbose=verbose,
            **method_kwargs
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose from 'simlr', 'flow_simr_v', 'lend_simr'.")
        
    new_v_dict = {}
    for idx, name in enumerate(block_names_ordered):
        v_mat = simlr_fit["v"][idx]
        if torch.is_tensor(v_mat):
            v_mat = v_mat.detach().cpu().numpy()
        features = block_features_preprocessed[name]
        df_v = pd.DataFrame(
            v_mat,
            index=features,
            columns=[f"PC{i+1}" for i in range(v_mat.shape[1])]
        )
        new_v_dict[name] = df_v
    simlr_fit["v_dict"] = new_v_dict
    
    updated_simlr_result = copy.deepcopy(simlr_result)
    
    if isinstance(simlr_result["v"], dict):
        if "v" not in updated_simlr_result:
            updated_simlr_result["v"] = {}
        for nm in new_modality_names:
            if nm in new_v_dict:
                updated_simlr_result["v"][nm] = new_v_dict[nm]
    else:
        if not isinstance(updated_simlr_result["v"], list):
            updated_simlr_result["v"] = list(updated_simlr_result["v"])
        for nm in new_modality_names:
            if nm in new_v_dict:
                v_new_val = new_v_dict[nm].values
                if len(simlr_result["v"]) > 0 and torch.is_tensor(simlr_result["v"][0]):
                    v_new_val = torch.tensor(v_new_val).float()
                updated_simlr_result["v"].append(v_new_val)
                
    if "modality_names" in updated_simlr_result:
        if not isinstance(updated_simlr_result["modality_names"], list):
            updated_simlr_result["modality_names"] = list(updated_simlr_result["modality_names"])
        for nm in new_modality_names:
            if nm not in updated_simlr_result["modality_names"]:
                updated_simlr_result["modality_names"].append(nm)
                
    if "feature_names" in updated_simlr_result:
        if not isinstance(updated_simlr_result["feature_names"], dict):
            if isinstance(updated_simlr_result["feature_names"], list):
                updated_simlr_result["feature_names"] = list(updated_simlr_result["feature_names"])
                for nm in new_modality_names:
                    updated_simlr_result["feature_names"].append(new_modalities[nm])
        else:
            for nm in new_modality_names:
                updated_simlr_result["feature_names"][nm] = new_modalities[nm]
                
    diagnostics = {
        "mode_used": mode,
        "split_prefixes_used": split_prefixes,
        "adjacency_type": adjacency_type,
        "cor_threshold": cor_threshold,
        "use_abs_correlation": use_abs_correlation,
        "preprocess": preprocess,
        "n_subjects": n_subjects,
        "block_names": block_names_ordered,
        "block_dims": {nm: mat.shape for nm, mat in blocks.items()},
        "preprocessing": preprocessing_diagnostics,
        "k_method": k_method,
        "cumvar_threshold": cumvar_threshold,
        "min_k": min_k,
        "joint_k_policy": joint_k_policy
    }
    
    return {
        "updated_simlr_result": updated_simlr_result,
        "simlr_fit": simlr_fit,
        "blocks": blocks,
        "adjacency": adjacency,
        "projected_data": pymm_proj,
        "k_new_used": k_new_used,
        "joint_k": joint_k,
        "diagnostics": diagnostics
    }
