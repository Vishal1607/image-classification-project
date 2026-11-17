# Image Captioning: CLIP + GPT-2 on Flickr30k

Phase 1 builds a curated Flickr30k subset on the standard Karpathy split (5,000 images, about 25k captions), cleans it, embeds every image with CLIP ViT-B/32, and tokenizes every caption with GPT-2.

| Split | Images | Notes |
|---|---|---|
| train | 2,986 | random sample of the Karpathy train split (seed 42) |
| val | 1,014 | the full Karpathy val split |
| test | 1,000 | the full Karpathy test split (comparable with published results) |

## Setup (local, conda)
```bash
conda create -n clipcap python=3.11 -y
conda activate clipcap
pip install -r requirements.txt
python -m ipykernel install --user --name clipcap --display-name "Python (clipcap)"
```
Open `Image_Captioning_Baseline.ipynb` and pick the **Python (clipcap)** kernel. The notebook uses the Apple-silicon GPU (`mps`), a CUDA GPU, or the CPU, whichever is available.

## Setup (Google Colab)
Upload the notebook, choose *Runtime → Change runtime type → T4 GPU*, and run all cells. Outputs are saved to `MyDrive/image-captioning` on Google Drive.

## First run
The first run streams the 4.4 GB Flickr30k image set once and saves only the 5,000 needed images (about 0.5 GB). This takes a while, depending on your connection. Later runs reuse the images and embeddings already on disk.

## Outputs (generated, git-ignored)
- `data/flickr30k_subset/images/`: the 5,000 images
- `data/flickr30k_subset/captions_clean.csv`: raw and cleaned captions, image id, Karpathy split
- `outputs/clip_image_embeddings.pt`: one L2-normalized 512-d CLIP embedding per image
- `outputs/gpt2_tokens.pt`: GPT-2 `input_ids` and `attention_mask` per caption, linked to its image

## Data sources
- Images: [`lmms-lab-encoder/flickr30k`](https://huggingface.co/datasets/lmms-lab-encoder/flickr30k)
- Karpathy split and raw captions: [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k) (`flickr_annotations_30k.csv`)
- Flickr30k (Young et al., 2014) may be used for non-commercial research and education only.
