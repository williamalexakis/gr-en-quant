import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)
for code, out in [("ell_Grek","data/flores_el.txt"), ("eng_Latn","data/flores_en.txt")]:
    dataset = load_dataset("openlanguagedata/flores_plus", code)
    rows = list(dataset["dev"]) + list(dataset["devtest"])
    
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(r["text"] for r in rows) + "\n")
        
    print(f"Wrote {len(rows)} lines to '{out}'.")
