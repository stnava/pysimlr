import re
import base64
import requests
import os
import subprocess
import time
import shutil
import sys
import glob

# Ensure real-time terminal feedback without buffering behind subprocesses
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

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

def clean_latex_auxiliary_files(paper_dir="paper", verbose=True):
    """
    Safely purge residual or corrupted LaTeX auxiliary files in paper directory
    prior to PDF compilation and upon compilation failure.
    
    Prevents cascading LuaLaTeX crashes caused by truncated auxiliary files
    (e.g., 'File ended while scanning use of \\@writefile').
    """
    explicit_files = [
        "index.aux", "index.toc", "index.log", "index.out", "index.lot", "index.lof", "index.tex", "index.fls",
        "paper.aux", "paper.toc", "paper.log", "paper.out", "paper.lot", "paper.lof", "paper.fls"
    ]
    
    aux_extensions = [
        "*.aux", "*.toc", "*.log", "*.out", "*.lot", "*.lof",
        "*.bbl", "*.blg", "*.fls", "*.fdb_latexmk", "*.synctex.gz"
    ]
    
    files_to_remove = set()
    for f in explicit_files:
        p = os.path.join(paper_dir, f)
        if os.path.exists(p):
            files_to_remove.add(p)
            
    for ext in aux_extensions:
        for p in glob.glob(os.path.join(paper_dir, ext)):
            files_to_remove.add(p)
            
    purged = []
    for p in sorted(files_to_remove):
        try:
            os.remove(p)
            purged.append(os.path.basename(p))
        except OSError as e:
            if verbose:
                print(f"Warning: could not remove auxiliary file {p}: {e}")
                
    if verbose and purged:
        print(f"Purged {len(purged)} stale LaTeX auxiliary file(s): {', '.join(purged)}")
    return len(purged)

def atomic_publish_file(source_file, target_file):
    """
    Atomically publish source_file to target_file using a temporary file
    and os.replace to prevent partial reads or transient missing states.
    """
    target_dir = os.path.dirname(os.path.abspath(target_file))
    os.makedirs(target_dir, exist_ok=True)
    tmp_target = os.path.join(target_dir, f".{os.path.basename(target_file)}.tmp")
    shutil.copy2(source_file, tmp_target)
    os.replace(tmp_target, target_file)

def publish_directory(source_dir, target_dir, preserve_files=("Deep-SiMLR-Manuscript.pdf",)):
    """
    Synchronize contents from source_dir into target_dir without destroying
    the destination directory or wiping preserved artifacts like the manuscript PDF.
    Prunes stale files that no longer exist in source_dir, except those in preserve_files.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Non-destructive recursive copy / update
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    
    # Prune stale files and directories in target_dir that are not in source_dir
    source_entries = set(os.listdir(source_dir))
    for item in os.listdir(target_dir):
        if item in preserve_files:
            continue
        if item not in source_entries:
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                try:
                    os.remove(item_path)
                except OSError:
                    pass

def sync_pdf_figures():
    """
    Synchronize missing or outdated PDF figures from PNGs in figure directories
    prior to PDF compilation, ensuring zero-error LuaLaTeX compilation under freeze: auto.
    
    Checks all potential source locations (paper/_book, paper/_freeze, paper)
    and synchronizes to all required destination directories (paper/.../figure-pdf
    and paper/_freeze/.../figure-pdf).
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL/Pillow not available, skipping automatic PNG to PDF synchronization.")
        return 0

    chapters = {
        "01_intro_background",
        "02_methods",
        "03_experiments",
        "04_discussion",
        "appendix_brca_clinical_validation",
        "appendix_flow_simr_v",
        "appendix_operational",
        "index"
    }
    
    # Dynamically discover any additional chapters from directory structures
    for path in glob.glob("paper/*_files") + glob.glob("paper/_book/*_files"):
        stem = os.path.basename(path).replace("_files", "")
        chapters.add(stem)
    for path in glob.glob("paper/_freeze/*"):
        if os.path.isdir(path):
            chapters.add(os.path.basename(path))

    synced_count = 0

    for chapter in sorted(chapters):
        candidate_source_dirs = [
            os.path.join("paper/_book", f"{chapter}_files", "figure-html"),
            os.path.join("paper/_freeze", chapter, "figure-html"),
            os.path.join("paper", f"{chapter}_files", "figure-html"),
            os.path.join("paper/_book", f"{chapter}_files", "figure-latex"),
            os.path.join("paper/_freeze", chapter, "figure-latex"),
            os.path.join("paper", f"{chapter}_files", "figure-latex"),
        ]

        target_dest_dirs = [
            os.path.join("paper", f"{chapter}_files", "figure-pdf"),
            os.path.join("paper/_freeze", chapter, "figure-pdf"),
            os.path.join("paper/_book", f"{chapter}_files", "figure-pdf"),
        ]

        available_pngs = {}
        for src_dir in candidate_source_dirs:
            if os.path.exists(src_dir):
                for fname in sorted(os.listdir(src_dir)):
                    if fname.lower().endswith(".png"):
                        fpath = os.path.join(src_dir, fname)
                        if os.path.getsize(fpath) > 0:
                            if fname not in available_pngs or os.path.getmtime(fpath) > os.path.getmtime(available_pngs[fname]):
                                available_pngs[fname] = fpath

        # Also inspect existing valid PDFs across all destination directories
        existing_pdfs = {}
        for dest_dir in target_dest_dirs:
            if os.path.exists(dest_dir):
                for fname in sorted(os.listdir(dest_dir)):
                    if fname.lower().endswith(".pdf"):
                        fpath = os.path.join(dest_dir, fname)
                        if os.path.getsize(fpath) > 0:
                            if fname not in existing_pdfs or os.path.getmtime(fpath) > os.path.getmtime(existing_pdfs[fname]):
                                existing_pdfs[fname] = fpath

        # Cross-synchronize existing valid PDFs across destinations
        for pdf_filename, valid_pdf_path in existing_pdfs.items():
            for dest_dir in target_dest_dirs:
                dest_pdf_path = os.path.join(dest_dir, pdf_filename)
                if not os.path.exists(dest_pdf_path) or os.path.getsize(dest_pdf_path) == 0:
                    try:
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy2(valid_pdf_path, dest_pdf_path)
                        print(f"Restored figure PDF across destinations: {dest_pdf_path} from {valid_pdf_path}")
                        synced_count += 1
                    except Exception as e:
                        print(f"Error copying {valid_pdf_path} to {dest_pdf_path}: {e}")

        if not available_pngs:
            continue

        for png_filename, png_path in sorted(available_pngs.items()):
            pdf_filename = os.path.splitext(png_filename)[0] + ".pdf"
            png_mtime = os.path.getmtime(png_path)

            for dest_dir in target_dest_dirs:
                os.makedirs(dest_dir, exist_ok=True)
                dest_pdf_path = os.path.join(dest_dir, pdf_filename)

                needs_sync = False
                if not os.path.exists(dest_pdf_path) or os.path.getsize(dest_pdf_path) == 0:
                    needs_sync = True
                elif png_mtime > os.path.getmtime(dest_pdf_path):
                    needs_sync = True

                if needs_sync:
                    # Check if another destination already has an up-to-date valid PDF
                    if (
                        pdf_filename in existing_pdfs
                        and os.path.getsize(existing_pdfs[pdf_filename]) > 0
                        and os.path.getmtime(existing_pdfs[pdf_filename]) >= png_mtime
                    ):
                        try:
                            shutil.copy2(existing_pdfs[pdf_filename], dest_pdf_path)
                            print(f"Restored figure PDF from existing: {dest_pdf_path} from {existing_pdfs[pdf_filename]}")
                            synced_count += 1
                            continue
                        except Exception as e:
                            print(f"Error copying {existing_pdfs[pdf_filename]} to {dest_pdf_path}: {e}")

                    try:
                        img = Image.open(png_path)
                        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                            rgba = img.convert("RGBA")
                            rgb_img = Image.new("RGB", rgba.size, (255, 255, 255))
                            rgb_img.paste(rgba, mask=rgba.split()[-1])
                        else:
                            rgb_img = img.convert("RGB")
                        rgb_img.save(dest_pdf_path, "PDF", resolution=300.0)
                        print(f"Synchronized figure PDF: {dest_pdf_path} from {png_path}")
                        synced_count += 1
                        existing_pdfs[pdf_filename] = dest_pdf_path
                    except Exception as e:
                        print(f"Error converting {png_path} to {dest_pdf_path}: {e}")

    print(f"Figure synchronization complete: {synced_count} figures synchronized.")
    return synced_count


def extract_and_build():
    fig_dir = "paper/figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    mapping = {
        "fig-interpretability-gap": "paper/01_intro_background.qmd",
        "fig-simlr": "paper/02_methods.qmd",
        "fig-lend": "paper/02_methods.qmd",
        "fig-ned": "paper/02_methods.qmd",
        "fig-nedpp": "paper/02_methods.qmd",
        "fig-flow-simr": "paper/02_methods.qmd"
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

    if not all_success:
        print("\nSkipping Quarto render due to Mermaid generation errors.")
        sys.exit(1)

    print("\nAll Mermaid diagrams generated successfully.")
    
    # Prepare environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    if "QUARTO_PYTHON" not in env:
        env["QUARTO_PYTHON"] = sys.executable

    source_dir = os.path.abspath("paper/_book")
    target_dir = os.path.abspath("docs/manuscript")

    print("\n--- Phase 1: Rendering HTML ---")
    try:
        subprocess.run(["quarto", "render", "paper", "--to", "html"], env=env, check=True)
        
        html_path = os.path.join(source_dir, "index.html")
        if os.path.exists(html_path):
            print(f"\nSUCCESS: HTML version rendered to: {html_path}")
            # Non-destructive publish: preserves Deep-SiMLR-Manuscript.pdf if present
            publish_directory(source_dir, target_dir, preserve_files=("Deep-SiMLR-Manuscript.pdf",))
            print(f"Published to: {target_dir}")
        else:
            print(f"\nWARNING: HTML render finished but {html_path} was not found.")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"HTML render failed with error code {e.returncode}")
        sys.exit(e.returncode)

    print("\n--- Phase 2: Rendering PDF ---")
    # Step 1: Pre-compilation cleanup of stale LaTeX auxiliary files
    clean_latex_auxiliary_files(paper_dir="paper", verbose=True)

    # Step 2: Synchronize Figure Assets for PDF Compilation
    print("--- Synchronizing Figure Assets for PDF Compilation ---")
    sync_pdf_figures()

    # Step 3: Compile PDF via Quarto / LuaLaTeX
    try:
        subprocess.run(["quarto", "render", "paper", "--to", "pdf"], env=env, check=True)
        pdf_path = os.path.abspath("paper/_book/paper.pdf")
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            print(f"\nSUCCESS: PDF version rendered to: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
            target_pdf = os.path.join(target_dir, "Deep-SiMLR-Manuscript.pdf")
            atomic_publish_file(pdf_path, target_pdf)
            print(f"Atomically copied PDF to: {target_pdf}")
            
            # Ensure both HTML and PDF coexist in paper/_book as well as docs/manuscript
            book_dest_pdf = os.path.join(source_dir, "Deep-SiMLR-Manuscript.pdf")
            if book_dest_pdf != pdf_path:
                atomic_publish_file(pdf_path, book_dest_pdf)
            for item in os.listdir(target_dir):
                s = os.path.join(target_dir, item)
                d = os.path.join(source_dir, item)
                if not os.path.exists(d):
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
        else:
            print(f"\nWARNING: PDF render finished but valid {pdf_path} was not found.")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"PDF render failed with error code {e.returncode}")
        # Crash recovery: purge corrupt auxiliary files so subsequent runs don't crash on \@writefile
        clean_latex_auxiliary_files(paper_dir="paper", verbose=True)
        sys.exit(e.returncode)

if __name__ == "__main__":
    extract_and_build()
