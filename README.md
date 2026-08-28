# Greek versus English Degradation Under Consumer GGUF Quantization

These are the results and scripts for an empirical research paper on how GGUF k-quantization damages Greek more than English on sub-4B Large Language Models.

**Paper: 10.5281/zenodo.22145680**

Greek diverges further from the ```Q8_0``` reference than English at every level of the quantization ladder on both Llama 3.2 3B and 1B, with a mean Greek-English KL divergence ratio that never falls below 1.35. The damage migrates from a tail of severely distorted tokens at high precision to typical tokens at low precision, with a median KLD ratio of 1.05 at ```Q6_K``` on 3B rising to 2.00 at ```Q2_K```. The penalty is larger on the smaller model, with a practical safe floor of ```Q4_K_M``` on 3B and ```Q5_K_M``` on 1B, and ```Q2_K``` unusable on either model.

## Requirements
- llama.cpp build 10090, commit ```7347430f4```, ```llama-perplexity``` on ```PATH```
- Python 3.10+
- Hugging Face account with [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) dataset gate accepted for ```scripts/prep_data.py``` to work

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: The original results in the paper were produced on an Apple M2 (```darwin-arm64```) with Metal backend, AppleClang 21.0.0.

## Reproducing
(Run all commands from repo root)

### 1. Build Corpus
Run this script to write the dev and devtest splits of FLORES+ to ```data/``` as one sentence per line:
```bash
python3 scripts/prep_data.py
```

### 2. Get Models
The full GGUF k-quant ladder across both models is sourced from a single uploader, so there's no cross-uploader variation. ```Q8_0``` is the reference, and the five evaluated levels are ```Q6_K```, ```Q5_K_M```, ```Q4_K_M```, ```Q3_K_M```, and ```Q2_K_M```.

Run these to set the models up:
```bash
mkdir -p models_3b models_1b

hf download QuantFactory/Llama-3.2-3B-GGUF \
    Llama-3.2-3B.Q8_0.gguf Llama-3.2-3B.Q6_K.gguf Llama-3.2-3B.Q5_K_M.gguf \
    Llama-3.2-3B.Q4_K_M.gguf Llama-3.2-3B.Q3_K_M.gguf Llama-3.2-3B.Q2_K.gguf \
    --local-dir models_3b

hf download QuantFactory/Llama-3.2-1B-GGUF \
    Llama-3.2-1B.Q8_0.gguf Llama-3.2-1B.Q6_K.gguf Llama-3.2-1B.Q5_K_M.gguf \
    Llama-3.2-1B.Q4_K_M.gguf Llama-3.2-1B.Q3_K_M.gguf Llama-3.2-1B.Q2_K.gguf \
    --local-dir models_1b
```

### 3. Run Evaluation
Write the ```Q8_0``` reference logits once per model and language before scoring each quant against the reference, using these scripts:
```bash
./scripts/runeval.sh 3b
./scripts/runeval.sh 1b
```

### 4. Parse & Graph Data
Run this script to recreate the summary CSVs and figures:
```bash
python3 scripts/analyze.py
```
