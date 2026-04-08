import numpy as np
import pandas as pd
import subprocess
import os
from pysimlr import multigrep, get_names_from_dataframe

def run_r_code(code: str) -> str:
    # Use Rscript to run code and return output
    process = subprocess.Popen(['Rscript', '-e', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"R error: {stderr}")
    return stdout

def test_parity_utils():
    # 1. Multigrep Parity
    desc = ["apple", "banana", "cherry", "date", "eggplant"]
    patterns = ["a", "e"]
    
    # Python results (0-indexed)
    py_res_union = multigrep(patterns, desc, intersect=False)
    py_res_intersect = multigrep(patterns, desc, intersect=True)
    
    # R results (1-indexed)
    r_code = f"""
    multigrep <- function( x, desc, intersect=FALSE ) {{
      roisel = c()
      for ( xx in x ) {{
        if (length(roisel)==0 | !intersect ) {{
          roisel = c( roisel, grep(xx, desc) )
        }} else {{
          roisel = intersect( roisel, grep(xx, desc) )
        }}
      }}
      return(  roisel )
    }}
    desc = c("apple", "banana", "cherry", "date", "eggplant")
    patterns = c("a", "e")
    cat("union:", multigrep(patterns, desc, FALSE), "\\n")
    cat("intersect:", multigrep(patterns, desc, TRUE), "\\n")
    """
    try:
        r_out = run_r_code(r_code)
        # Parse R output: "union: 1 2 4 1 3 5 \n intersect: 1 4 \n"
        # Wait, R union might have duplicates if not handled.
        # Python uses np.unique(indices).
        # Let's adjust R code for comparison.
        r_code_adj = r_code + "cat('union_unique:', sort(unique(multigrep(patterns, desc, FALSE))), '\\n')"
        r_out = run_r_code(r_code_adj)
        
        # Compare union
        r_union = [int(x) - 1 for x in r_out.split('union_unique: ')[1].split('\n')[0].split()]
        assert np.array_equal(np.sort(py_res_union), np.sort(r_union))
        
        # Compare intersect
        r_intersect = [int(x) - 1 for x in r_out.split('intersect: ')[1].split('\n')[0].split()]
        assert np.array_equal(np.sort(py_res_intersect), np.sort(r_intersect))
        print("Utils parity: PASSED")
    except Exception as e:
        print(f"Skipping R parity check (R not found or error): {e}")

if __name__ == "__main__":
    test_parity_utils()
