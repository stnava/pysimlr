import re
import base64
import requests
import os
import subprocess
import time

def generate_mermaid_png(mermaid_code, output_path, retries=3):
    # Strip Quarto comments (#|) before encoding
    clean_code = "\n".join([line for line in mermaid_code.split("\n") if not line.strip().startswith("#|")])
    
    # Encode to base64 for mermaid.ink
    graphbytes = clean_code.encode("utf-8")
    base64_string = base64.b64encode(graphbytes).decode("ascii")
    url = f"https://mermaid.ink/img/{base64_string}"
    
    print(f"Generating: {output_path} (retries={retries})...")
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"Successfully saved {output_path}")
                return True
            elif response.status_code == 503:
                print(f"503 Service Unavailable for {output_path}, retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"Error generating {output_path}: {response.status_code}")
                if response.status_code < 500:
                    return False
                time.sleep(2)
        except Exception as e:
            print(f"Exception generating {output_path}: {e}")
            time.sleep(2)
            
    print(f"Failed to generate {output_path} after {retries} attempts.")
    return False

def extract_and_build():
    fig_dir = "paper/figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    mapping = {
        "fig-interpretability-gap": "paper/01_intro_background.qmd",
        "fig-simlr": "paper/02_methods.qmd",
        "fig-lend": "paper/02_methods.qmd",
        "fig-ned": "paper/02_methods.qmd",
        "fig-nedpp": "paper/02_methods.qmd"
    }

    all_success = True
    for fig_id, filepath in mapping.items():
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
        
        pattern = rf"::: \{{#{fig_id}\}}[\s\S]*?```{{mermaid}}([\s\S]*?)```"
        match = re.search(pattern, content)
        
        if match:
            mermaid_code = match.group(1).strip()
            output_path = os.path.join(fig_dir, f"{fig_id}.png")
            success = generate_mermaid_png(mermaid_code, output_path)
            if not success:
                all_success = False
        else:
            print(f"Could not find mermaid block for {fig_id} in {filepath}")
            all_success = False

    if all_success:
        print("\nAll Mermaid diagrams generated successfully.")
        
        print("Starting Quarto HTML Render...")
        try:
            subprocess.run(["quarto", "render", "paper", "--to", "html"], check=True)
            print("HTML version rendered successfully.")
        except subprocess.CalledProcessError as e:
            print(f"HTML render failed: {e}")

        print("\nStarting Quarto PDF Render...")
        try:
            subprocess.run(["quarto", "render", "paper", "--to", "pdf"], check=True)
            print("PDF version rendered successfully.")
        except subprocess.CalledProcessError as e:
            print(f"PDF render failed: {e}")
    else:
        print("\nSkipping Quarto render due to Mermaid generation errors.")

if __name__ == "__main__":
    extract_and_build()
