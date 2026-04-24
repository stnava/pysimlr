import re
import base64
import requests
import os
import subprocess
import time
import shutil

def generate_mermaid_png(mermaid_code, output_path, retries=3):
    clean_code = "\n".join([line for line in mermaid_code.split("\n") if not line.strip().startswith("#|")])
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
        
        print("\n--- Phase 1: Rendering HTML ---")
        try:
            # Render to HTML
            subprocess.run(["quarto", "render", "paper", "--to", "html"], check=True)
            
            # Identify output directory
            source_dir = os.path.abspath("paper/_book")
            target_dir = os.path.abspath("docs/manuscript")
            
            # Check if index.html exists
            html_path = os.path.join(source_dir, "index.html")
            if os.path.exists(html_path):
                print(f"\nSUCCESS: HTML version rendered to: {html_path}")
                
                # Copy to docs/manuscript for GitHub Pages
                print(f"Publishing to: {target_dir}...")
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(source_dir, target_dir)
                print(f"Done! Book is now available at {target_dir}")
                print(f"To view: open '{os.path.join(target_dir, 'index.html')}'")
            else:
                print(f"\nWARNING: HTML render finished but {html_path} was not found.")
        except subprocess.CalledProcessError as e:
            print(f"HTML render failed with error code {e.returncode}")

        print("\n--- Phase 2: Rendering PDF ---")
        try:
            subprocess.run(["quarto", "render", "paper", "--to", "pdf"], check=True)
            pdf_path = os.path.abspath("paper/_book/paper.pdf")
            if os.path.exists(pdf_path):
                print(f"\nSUCCESS: PDF version rendered to: {pdf_path}")
                # Also copy PDF to docs/manuscript
                shutil.copy(pdf_path, os.path.join(target_dir, "Deep-SiMLR-Manuscript.pdf"))
                print(f"PDF copied to {target_dir}/Deep-SiMLR-Manuscript.pdf")
            else:
                print(f"\nWARNING: PDF render finished but {pdf_path} was not found.")
        except subprocess.CalledProcessError as e:
            print(f"PDF render failed with error code {e.returncode}")
    else:
        print("\nSkipping Quarto render due to Mermaid generation errors.")

if __name__ == "__main__":
    extract_and_build()
