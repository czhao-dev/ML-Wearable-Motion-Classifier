# Applied Machine Learning Projects

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A monorepo of nine end-to-end machine learning projects spanning computer vision, large language models, graph learning, time-series forecasting, and production inference serving. Each project lives in its own subdirectory with independent dependencies, tests, documented results, and a full README.

## Projects at a Glance

| Project | Area | Key Technologies | Standout |
| --- | --- | --- | --- |
| [ml-satellite-image-classifier](#ml-satellite-image-classifier) | Computer Vision | PyTorch, Keras/TF, FastAPI, Docker | 99.83% accuracy; FastAPI server serving all four models |
| [ml-llm-alignment-fine-tuning](#ml-llm-alignment-fine-tuning) | LLM Alignment | PyTorch, TRL, HuggingFace, LoRA | Full SFT → RM → PPO RLHF → DPO pipeline, all trained locally |
| [ml-tiny-llm-gpt](#ml-tiny-llm-gpt) | Language Modeling | PyTorch | GPT decoder-only Transformer built from scratch |
| [ml-gcp-vertex-rag-chatbot](#ml-gcp-vertex-rag-chatbot) | RAG / GenAI | LangChain, Vertex AI, Chroma, Cloud Run | Document Q&A app deployed to GCP Cloud Run |
| [ml-movie-recommender](#ml-movie-recommender) | Graph ML | PyTorch Geometric, igraph | Heterogeneous GNN over IMDb graphs; top-N recommendation on MovieLens |
| [ml-social-network-predictor](#ml-social-network-predictor) | Graph ML | igraph, PyTorch | DeepWalk embeddings reach 0.986 ROC-AUC on 4,039-node Facebook graph |
| [ml-wearable-motion-classifier](#ml-wearable-motion-classifier) | Classical ML | scikit-learn, NumPy, SciPy | IMU → trajectory → ensemble classifier for clinical rehabilitation |
| [ml-recyclable-material-classifier-vgg16](#ml-recyclable-material-classifier-vgg16) | Computer Vision | Keras/TF, VGG16 | Transfer learning; caught and fixed training-on-test bug from source notebook |
| [ml-boston-climate-modeler](#ml-boston-climate-modeler) | Time-Series | Python (no ML deps) | Ridge regression from scratch; 22 unit tests; zero third-party runtime deps |

---

## Project Details

### ml-satellite-image-classifier

Binary classification of 64×64 satellite image tiles as agricultural vs. non-agricultural land.

- **Models:** Six-block CNN (32→1024 channels, 5×5 kernels) and CNN–Vision Transformer hybrid (CNN feature map tokenized and fed through multi-head self-attention blocks), each implemented independently in both Keras/TensorFlow and PyTorch — four models total, trained and evaluated in parallel across frameworks.
- **Results:** PyTorch CNN 99.83%, Keras CNN 99.33%, PyTorch CNN-ViT 99.67%, Keras CNN-ViT 99.42% — all on a 1,200-image held-out split never seen during training.
- **Inference server:** FastAPI app (`serve/`) loads all four models at startup and exposes `/health`, `/models`, and `POST /predict?model=` endpoints. Model backend is selectable per request. Deployed with Docker Compose; model weights mounted read-only at runtime to keep the image small.
- **Notable:** Caught and fixed a data-leakage bug in the original evaluation methodology that scored models against the full training set rather than a held-out split.

**Stack:** Python · PyTorch · Keras/TensorFlow · FastAPI · Uvicorn · Docker Compose

---

### ml-llm-alignment-fine-tuning

Four LLM alignment techniques implemented end-to-end, each with its own training objective and evaluation metric — all locally runnable on a laptop CPU.

- **Supervised fine-tuning (SFT):** LoRA-adapts `facebook/opt-350m` on CodeAlpaca-20k with TRL's `SFTTrainer`; evaluated with SacreBLEU before and after fine-tuning.
- **Reward modeling:** GPT-2 + LoRA trained on chosen/rejected response pairs with `RewardTrainer`; reaches **0.96 pairwise ranking accuracy** on a held-out preference set.
- **PPO RLHF:** `gpt2-imdb` steered toward positive and negative sentiment with `PPOTrainer` against a sentiment-classifier reward; mean reward improves from 0.24 → 1.27 (positive policy) and −0.32 → 0.56 (negative policy).
- **DPO:** GPT-2 + LoRA fine-tuned directly on preference pairs with `DPOTrainer` (no separate reward model or RL loop); reaches **0.70 reward accuracy** on held-out pairs.
- Every script replaces commented-out or pre-downloaded training from the source notebooks with real, locally-runnable training — every number above comes from an actual training run.

**Stack:** Python · PyTorch · HuggingFace Transformers · TRL · LoRA (PEFT)

---

### ml-tiny-llm-gpt

A from-scratch GPT-style language model covering the complete pipeline from raw text to generated output.

- **Architecture:** Decoder-only Transformer with configurable depth, heads, and embedding dimension; causal self-attention, learned positional embeddings, and layer normalization — no HuggingFace model code in the loop.
- **Pipeline:** Custom BPE tokenizer training → dataset preprocessing and sequence packing → training loop with gradient clipping and validation checkpointing → top-k / top-p text generation → perplexity evaluation → throughput and memory benchmarking.
- Intentionally sized to run on consumer hardware while demonstrating every component of a modern LLM training stack.

**Stack:** Python · PyTorch

---

### ml-gcp-vertex-rag-chatbot

A deployed RAG document Q&A app: upload a PDF, TXT, Markdown, CSV, or DOCX file and ask questions grounded in its content.

- **RAG pipeline:** LangChain document loaders → `RecursiveCharacterTextSplitter` → Vertex AI `text-embedding-004` embeddings → Chroma vector store → `RetrievalQA` chain → Gemini 2.5 Flash answer with source grounding.
- **Deployment:** Containerized with Docker and deployed to GCP Cloud Run with scale-to-zero cost controls; credentials handled via Application Default Credentials for local development.
- **Interface:** Gradio web UI; standalone annotated scripts for each RAG concept (loading, splitting, embedding, retrieval) as reference implementations.

**Stack:** Python · LangChain · Google Vertex AI (Gemini + text-embedding-004) · Chroma · Gradio · Docker · Cloud Run

---

### ml-movie-recommender

Graph feature engineering pipeline extended with a heterogeneous GNN, evaluated on two separate tasks.

- **Feature engineering (igraph):** Actor/movie networks built from IMDb data; actors ranked by PageRank, movie communities detected with Fast Greedy Newman, Jaccard movie-movie similarity computed — all kept exactly as in the original coursework pipeline.
- **Heterogeneous GNN (PyTorch Geometric):** `HeteroConv` encoder with `GraphConv` for weighted relations and `SAGEConv` for unweighted bipartite edges; graph-derived features (PageRank, community ID, Jaccard similarity) used as node features.
- **IMDb track:** Movie rating prediction benchmarked against neighborhood averaging, linear regression, and bipartite graph averaging baselines.
- **MovieLens track:** Genuine personalized top-N recommendation on `ml-latest-small` (943 users, ~9,700 movies), evaluated with Precision/Recall/NDCG@{5,10,20} against the full catalog — not sampled negatives.

**Stack:** Python · PyTorch · PyTorch Geometric · igraph · scikit-learn

---

### ml-social-network-predictor

Structural analysis of large social graphs extended into a link-prediction task comparing hand-engineered heuristics against learned node embeddings.

- **Graph analysis (igraph):** Degree distribution, ego-network extraction, Fast-Greedy / Edge-Betweenness / Infomap / Walktrap community detection, embeddedness and dispersion scoring on 4,039-node Facebook and Google+ graphs.
- **Node embeddings:** Hand-rolled DeepWalk-style skip-gram model (PyTorch) trained on random walks from the training graph only — test edges withheld before any graph operation.
- **Link prediction results** (full Facebook graph, 1,000 held-out test edges): heuristics-only **0.974 ROC-AUC**, embeddings-only **0.986 ROC-AUC**, combined 0.953 ROC-AUC (threshold sensitivity; see case study writeup).
- Proper holdout methodology documented: test edges excluded before all graph analysis and embedding training to prevent leakage.

**Stack:** Python · PyTorch · igraph · scikit-learn

---

### ml-wearable-motion-classifier

A signal-processing and ML pipeline classifying upper-body movements from wrist-worn IMU data in a clinical rehabilitation context (Wolf Motor Function Test).

- **Preprocessing:** Sensor-frame alignment, gravity subtraction, zero-velocity updates (ZUPT), and wrist trajectory reconstruction from raw accelerometer/gyroscope/quaternion captures of a wrist-worn MPU-9150.
- **Features:** Vertical power, azimuth rotation, peak counts, path length, variance, and acceleration statistics extracted from reconstructed trajectories.
- **Models:** Deterministic rule-based baseline + four scikit-learn classifiers (SVM, random forest, histogram gradient boosting, soft-voting ensemble). Data augmentation pipeline with source-trial-aware grouped cross-validation to prevent leakage from augmented variants back into the test fold.
- **Results:** ~74% accuracy (grouped CV, 15 of 17 classes, real + augmented data) — see README for an honest discussion of dataset limitations.
- **Testing:** 6-module test suite covering the CLI, sensor parser, preprocessing, quaternion math, feature extraction, and the rule-based classifier. Not approved for clinical use.

**Stack:** Python · scikit-learn · NumPy · SciPy · pandas

---

### ml-recyclable-material-classifier-vgg16

Binary classification of organic vs. recyclable material images using VGG16 transfer learning, with two configurations compared head-to-head.

- **Configurations:** Frozen feature-extraction head vs. fine-tuning from `block5_conv3` onward; both trained on 800 images with augmentation, evaluated on a 200-image held-out test set.
- **Results:** Both reach 86.5% test accuracy; fine-tuning improves ROC-AUC (0.9534 vs. 0.9332) and produces more balanced precision/recall across classes.
- **Notable:** Caught and fixed a data-handling bug in the source notebook — a mislabeled generator meant every original training run was training on test-directory images, not training-directory images.

**Stack:** Python · Keras/TensorFlow · VGG16

---

### ml-boston-climate-modeler

Daily weather forecasting for Reading, MA (Boston suburb) from NOAA station data, with zero third-party ML library dependencies.

- **Pipeline:** NOAA CSV parsing → missing-value handling → seasonal, lag, and rolling-window feature construction → Ridge regression implemented from scratch (gradient descent + feature standardization) → model serialization to JSON for reuse without retraining.
- **Results:** Temperature RMSE drops from 10.66 (seasonal climatology baseline) to 8.17 (Ridge), explaining ~75% of held-out 2016 temperature variance (R² = 0.747). Precipitation and snowfall improve RMSE over the baseline but remain limited by event sparsity.
- **Testing:** 22 unit tests covering modeling, metrics, calendar-based train/test splitting, feature construction, and serialization.
- No NumPy, scikit-learn, or any third-party ML library in the runtime path.

**Stack:** Python (standard library only at runtime)

---

## License

This repository is licensed under the [MIT License](LICENSE).
