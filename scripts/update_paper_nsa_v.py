import os

file_path = "/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/code/pysimlr/paper/03_experiments.qmd"

with open(file_path, "r") as f:
    content = f.read()

insert_text = """

### Interpretability of the NSA Layer (Learned V Matrices)

To explicitly demonstrate the mechanistic transparency preserved by the First-Layer Contract, we visualize the learned weights ($V$) extracted directly from the NSA layer of the LEND architecture across both the Heart Disease and Diabetes clinical datasets. 

![**Interpretability of Learned NSA Layer Weights.** Heatmaps of the learned $V$ matrices for Heart Disease and Diabetes datasets. Using a high-contrast, color-blind friendly diverging colormap, the visualization instantly reveals the magnitude and direction of each clinical feature's contribution to the latent dimensions (Latent 1 and Latent 2). This low data-to-ink ratio focuses strictly on the extracted diagnostic weights, fulfilling the First-Layer Contract by offering an auditable linear entry point to the deep network.](figures/nsa_v_interpretability.png){#fig-nsa-v-interpretability width="100%"}

**Educational Exposition on Latent Interpretability:** 
The $V$ matrices effectively act as a diagnostic lens. In the Heart Disease dataset (Row 1), we observe how demographics (View 1, e.g., age, sex, chest pain type) and exercise/ECG metrics (View 2, e.g., maximum heart rate, ST depression) are uniquely weighted to form the latent risk profiles. For Diabetes (Row 2), the model clearly delineates the contributions of basic vitals (View 1, e.g., BMI, blood pressure) against complex blood serums (View 2). By strictly constraining the first layer to be linear and enforcing NSA orthogonality, we ensure that deep consensus mechanisms ($U$) do not obscure the foundational clinical drivers.

## Systematic Evaluation Matrix"""

if "### Interpretability of the NSA Layer" not in content:
    content = content.replace("## Systematic Evaluation Matrix", insert_text)
    with open(file_path, "w") as f:
        f.write(content)
    print("Paper updated successfully.")
else:
    print("Content already exists.")
