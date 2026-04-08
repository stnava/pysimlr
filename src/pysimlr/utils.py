import torch
import pandas as pd
import re
import time
from typing import List, Optional, Union

def set_seed_based_on_time() -> int:
    """
    Set random seed based on current time.
    
    Returns:
        The numeric value used as the seed.
    """
    seed_value = int(time.time() * 1000000) % (2**32 - 1)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value

def multigrep(patterns: List[str], desc: List[str], intersect: bool = False) -> torch.Tensor:
    """
    Grep entries with a list of search patterns.
    
    Args:
        patterns: List of search patterns (regex).
        desc: Target list of items to be searched.
        intersect: Whether to use intersection or union.
        
    Returns:
        Indices of desc that match patterns as a torch.Tensor.
    """
    indices_set = None
    for pattern in patterns:
        matches = {i for i, item in enumerate(desc) if re.search(pattern, item)}
        if indices_set is None or not intersect:
            if indices_set is None:
                indices_set = matches
            else:
                indices_set.update(matches)
        else:
            indices_set.intersection_update(matches)
    
    if indices_set is None:
        return torch.tensor([], dtype=torch.long)
    
    sorted_indices = sorted(list(indices_set))
    return torch.tensor(sorted_indices, dtype=torch.long)

def get_names_from_dataframe(patterns: List[str], df: pd.DataFrame, exclusions: Optional[List[str]] = None) -> List[str]:
    """
    Extract column names with search and exclusion parameters.
    
    Args:
        patterns: List of strings to search for in column names.
        df: The dataframe with column names to search.
        exclusions: List of strings to exclude from matching names.
        
    Returns:
        List of matching column names.
    """
    outnames = df.columns.tolist()
    
    for pattern in patterns:
        outnames = [name for name in outnames if re.search(pattern, name)]
        
    if exclusions:
        to_exclude = set()
        for excl in exclusions:
            to_exclude.update([name for name in outnames if re.search(excl, name)])
        
        outnames = [name for name in outnames if name not in to_exclude]
        
    return outnames

def map_asym_var(df: pd.DataFrame, left_vars: List[str], left_name: str = 'left', 
                right_name: str = 'right', replacer: str = 'Asym') -> pd.DataFrame:
    """
    Convert left/right variables to a measure of asymmetry.
    """
    df = df.copy()
    
    for left_var in left_vars:
        right_var = left_var.replace(left_name, right_name)
        if right_var in df.columns:
            new_name = left_var.replace(left_name, replacer)
            # Use torch for calculation
            l_val = torch.from_numpy(df[left_var].values)
            r_val = torch.from_numpy(df[right_var].values)
            diff = l_val - r_val
            asym = diff * torch.sign(diff)
            df[new_name] = asym.numpy()
            
    return df

def map_lr_average_var(df: pd.DataFrame, left_vars: List[str], left_name: str = 'left', 
                       right_name: str = 'right', replacer: str = 'LRAVG') -> pd.DataFrame:
    """
    Convert left/right variables to an average measurement.
    """
    df = df.copy()
    
    for left_var in left_vars:
        right_var = left_var.replace(left_name, right_name)
        if right_var in df.columns:
            new_name = left_var.replace(left_name, replacer)
            l_val = torch.from_numpy(df[left_var].values)
            r_val = torch.from_numpy(df[right_var].values)
            avg = 0.5 * (l_val + r_val)
            df[new_name] = avg.numpy()
            
    return df
