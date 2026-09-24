# The Complete AI/ML Engineering Stack: Zero to Today

A to-the-subtopic roadmap for **Python → Math → ML → DL → CV → NLP/LLMs → Generative AI → System One models → AI apps → Backend → Data Engineering → MLOps → Cloud → System Design → Scaling**.

Use the checkboxes (`- [ ]`) to track progress. The order within each part is roughly the order to learn it.

---

## Table of Contents

**Foundations**
0. [Computer & Dev Fundamentals](#0-computer--dev-fundamentals)
1. [Python (Beginner → Expert)](#1-python-beginner--expert)
2. [Data Structures & Algorithms](#2-data-structures--algorithms)
3. [Databases & SQL](#3-databases--sql)
4. [Mathematics for ML](#4-mathematics-for-ml)

**Core ML**
5. [Data Analysis & Feature Engineering](#5-data-analysis--feature-engineering)
6. [Classical Machine Learning](#6-classical-machine-learning)
7. [Deep Learning](#7-deep-learning)
8. [Computer Vision](#8-computer-vision)
9. [NLP (Classic → Transformers)](#9-nlp-classic--transformers)
10. [Large Language Models (Internals & Training)](#10-large-language-models-internals--training)
11. [Post-Training & Alignment (SFT → RLHF → DPO → RLVR → OPD → RLCD)](#11-post-training--alignment)
12. [System One / Decision Models (Jev, Laya)](#12-system-one--decision-models-new--sept-2026)

**Specialized Areas**
13. [Generative AI (Images, Video, Audio, 3D)](#13-generative-ai-images-video-audio-3d)
14. [Speech & Audio AI](#14-speech--audio-ai)
15. [Reinforcement Learning](#15-reinforcement-learning)
16. [Other ML Areas (RecSys, Time Series, Graphs, Tabular DL)](#16-other-ml-areas)

**Building Applications**
17. [LLM Application Engineering (Prompting, RAG, Agents, Evals)](#17-llm-application-engineering)
18. [Backend Engineering for AI (FastAPI, Redis, Queues, Auth)](#18-backend-engineering-for-ai)
19. [Vector Databases & Search](#19-vector-databases--search)
20. [Frontend for AI Apps](#20-frontend-for-ai-apps)

**Production & Operations**
21. [Data Engineering](#21-data-engineering)
22. [MLOps & LLMOps](#22-mlops--llmops)
23. [Model Serving & Inference Optimization](#23-model-serving--inference-optimization)
24. [DevOps: Docker, Kubernetes, CI/CD, IaC](#24-devops-docker-kubernetes-cicd-iac)
25. [Cloud Platforms](#25-cloud-platforms)
26. [GPU, CUDA & Hardware](#26-gpu-cuda--hardware)
27. [Monitoring & Observability](#27-monitoring--observability)
28. [Security, Safety & Responsible AI](#28-security-safety--responsible-ai)

**Design, Scaling & Career**
29. [System Design (General)](#29-system-design-general)
30. [ML System Design](#30-ml-system-design)
31. [LLM / GenAI System Design](#31-llm--genai-system-design)
32. [Scaling (Training & Serving)](#32-scaling-training--serving)
33. [Edge & On-Device AI](#33-edge--on-device-ai)
34. [Research, Projects & Career](#34-research-projects--career)
35. [Suggested Learning Order & Projects](#35-suggested-learning-order--projects)

---

## 0. Computer & Dev Fundamentals

- [ ] **How computers work:** CPU, RAM, cache, disk, bits/bytes, binary/hex, instruction cycle
- [ ] **Operating systems:**
  - Processes vs threads, scheduling, context switching
  - Memory (virtual memory, paging, stack vs heap)
  - File systems, I/O, system calls, signals, deadlocks
- [ ] **Linux & shell:**
  - Navigation and files: `ls cd pwd cp mv rm mkdir find grep sed awk cat less head tail`
  - Permissions and ownership: `chmod chown`
  - Processes and resources: `ps top htop kill`, `df du free`
  - Networking commands: `curl wget ssh scp rsync`
  - Archives and schedulers: `tar zip`, `cron`, `systemd`, `tmux/screen`
  - Pipes and redirection, environment variables, `.bashrc/.zshrc`, bash scripting
- [ ] **Networking:**
  - OSI/TCP-IP model, IP, DNS, ports, TCP vs UDP
  - HTTP/1.1 vs HTTP/2 vs HTTP/3, HTTPS/TLS
  - Status codes, headers, cookies, CORS, WebSockets, SSE, gRPC basics
  - Proxies, NAT, firewalls, load balancing basics
- [ ] **Git & GitHub:**
  - Basics: init, clone, add, commit, status, log, diff
  - Branching and history: branch, checkout/switch, merge, rebase, cherry-pick, stash, reset vs revert, tags
  - Collaboration: remotes, PRs, code review, merge conflicts, .gitignore, Git LFS
  - Workflows: GitFlow, trunk-based
- [ ] **Dev environment:** VS Code/Cursor, Jupyter/JupyterLab, Google Colab, Kaggle notebooks, WSL, Homebrew, dotfiles
- [ ] **AI coding tools:** Copilot-style assistants, agentic coding CLIs, prompt-driven development, reviewing AI-generated code
- [ ] **Software engineering practices:**
  - Clean code, SOLID, DRY, KISS, YAGNI
  - Design patterns: factory, singleton, strategy, observer, adapter, dependency injection
  - Code review, documentation, semantic versioning

---

## 1. Python (Beginner → Expert)

### 1.1 Basics
- [ ] Installing Python, REPL, running scripts, `__main__`
- [ ] **Variables and types:** dynamic typing, `int float complex bool str None`, type conversion
- [ ] **Operators:** arithmetic, comparison, logical, bitwise, identity (`is`), membership (`in`), walrus (`:=`)
- [ ] **Strings:** indexing, slicing, methods (`split join strip replace find upper lower format`), f-strings, escape chars, raw strings, encoding/Unicode
- [ ] **Input/output:** `print`, `input`, formatting
- [ ] **Control flow:** `if/elif/else`, `match/case` (structural pattern matching)
- [ ] **Loops:** `for`, `while`, `break`, `continue`, `else` on loops, `range`, `enumerate`, `zip`

### 1.2 Data Structures
- [ ] **Lists:** methods, slicing, sorting (`sort` vs `sorted`, `key`), nested lists, copying (shallow vs deep)
- [ ] **Tuples:** immutability, packing/unpacking, namedtuple
- [ ] **Sets:** operations (union, intersection, difference), frozenset
- [ ] **Dictionaries:** methods, iteration, dict comprehension, `get`, `setdefault`, merging (`|`)
- [ ] **collections:** `Counter`, `defaultdict`, `OrderedDict`, `deque`, `ChainMap`
- [ ] **Comprehensions:** list, dict, set, generator expressions
- [ ] Arrays module vs lists, mutability vs immutability, hashability

### 1.3 Functions
- [ ] Defining functions, return values, default args, keyword args
- [ ] `*args`, `**kwargs`, positional-only (`/`) and keyword-only (`*`) parameters
- [ ] Scope (LEGB), `global`, `nonlocal`
- [ ] Lambda, `map`, `filter`, `reduce`, `sorted` with key
- [ ] Recursion, closures, first-class functions, higher-order functions
- [ ] Docstrings, `functools` (`partial`, `lru_cache`, `cache`, `wraps`)

### 1.4 OOP
- [ ] Classes, objects, `__init__`, `self`, instance vs class attributes
- [ ] **Four pillars:** encapsulation, inheritance, polymorphism, abstraction
- [ ] Multiple inheritance, MRO, `super()`
- [ ] `@staticmethod`, `@classmethod`, `@property` (getters/setters)
- [ ] **Dunder/magic methods:**
  - Representation: `__str__ __repr__`
  - Containers: `__len__ __getitem__ __setitem__ __iter__ __next__`
  - Comparison and operators: `__eq__ __lt__ __add__`
  - Callables and context managers: `__call__ __enter__ __exit__`
  - Hashing: `__hash__`
- [ ] Abstract base classes (`abc`), interfaces, Protocols (structural typing)
- [ ] `dataclasses`, `__slots__`, enums
- [ ] Composition vs inheritance, mixins, metaclasses (advanced)

### 1.5 Error Handling & Files
- [ ] `try/except/else/finally`, raising exceptions, custom exceptions, exception groups
- [ ] **File I/O:** `open`, modes, `with`, reading/writing text and binary
- [ ] `pathlib`, `os`, `shutil`, `glob`
- [ ] **Formats:** CSV, JSON, YAML, TOML, pickle (and its security risk), Parquet
- [ ] Logging (`logging` module, levels, handlers, structured logs)

### 1.6 Modules, Packages & Environments
- [ ] `import` system, `__init__.py`, relative vs absolute imports
- [ ] **Package managers:** pip, `requirements.txt`, `pyproject.toml`, Poetry, **uv**, conda/mamba
- [ ] Virtual environments: venv, conda envs
- [ ] Building/publishing packages (PyPI), entry points, CLI tools (`argparse`, `click`, `typer`)

### 1.7 Advanced Python
- [ ] Iterators and iterables, generators, `yield`, `yield from`, lazy evaluation
- [ ] Decorators (function and class decorators, with args), context managers (`contextlib`)
- [ ] **Type hints:**
  - Basics: `typing`, generics, `TypeVar`, `Optional`, `Union`, `Literal`
  - Advanced: `TypedDict`, `Annotated`, `ParamSpec`
  - Checkers: mypy/pyright
- [ ] **Concurrency:**
  - `threading`, `multiprocessing`, `concurrent.futures`
  - The GIL (and free-threaded Python 3.13+)
- [ ] **Async:** `asyncio`, `async/await`, event loop, tasks, `gather`, semaphores, `aiohttp`/`httpx`
- [ ] **Memory:** reference counting, garbage collection, `sys.getsizeof`, weakrefs
- [ ] **Performance:**
  - Profiling: `cProfile`, `line_profiler`, `timeit`
  - Speedups: vectorization, Cython, Numba, PyPy, Rust extensions (PyO3)
- [ ] Regular expressions (`re`), `datetime`/`zoneinfo`, `itertools`, `operator`
- [ ] Serialization: `json`, `msgpack`, `orjson`, Pydantic models

### 1.8 Code Quality & Testing
- [ ] PEP 8, formatting and linting (**ruff**, black, isort), pre-commit hooks
- [ ] **Testing:** `pytest` (fixtures, parametrize, mocking, `monkeypatch`), `unittest`, coverage
- [ ] TDD, property-based testing (`hypothesis`), integration tests
- [ ] Debugging: `pdb`, `breakpoint()`, IDE debuggers

### 1.9 Other Languages (Good to Know)
- [ ] **SQL:** mandatory (see §3)
- [ ] **Bash:** scripting for automation
- [ ] **JavaScript/TypeScript:** for AI app frontends and SDKs
- [ ] **C++:** performance, CUDA kernels, llama.cpp-style runtimes
- [ ] **Rust/Go:** high-performance inference servers and tooling (optional)

---

## 2. Data Structures & Algorithms

- [ ] **Complexity:** Big-O, Big-Θ, Big-Ω, time vs space, amortized analysis
- [ ] **Arrays & strings:** two pointers, sliding window, prefix sums, Kadane's algorithm
- [ ] **Hashing:** hash maps/sets, collision handling, frequency counting
- [ ] **Linked lists:** singly and doubly linked, fast/slow pointers, reversal, cycle detection
- [ ] **Stacks & queues:** monotonic stack, deque, priority queue
- [ ] **Recursion & backtracking:** permutations, subsets, N-Queens
- [ ] **Sorting:** bubble, selection, insertion, merge, quick, heap, counting, radix
- [ ] **Searching:** binary search (and on the answer), ternary search
- [ ] **Trees:** binary trees, traversals (in/pre/post/level), BST, AVL/Red-Black (concepts), segment tree, Fenwick tree
- [ ] **Heaps:** min/max heap, top-K, median of stream
- [ ] **Tries:** prefix search, autocomplete
- [ ] **Graphs:**
  - Representations, BFS, DFS, topological sort, union-find
  - Shortest paths: Dijkstra, Bellman-Ford, Floyd-Warshall
  - MST: Prim, Kruskal
- [ ] **Dynamic programming:** memoization vs tabulation, knapsack, LIS, LCS, edit distance, DP on grids/trees
- [ ] **Greedy:** interval scheduling, Huffman coding
- [ ] **Bit manipulation:** masks, XOR tricks
- [ ] **ML-relevant algorithms:** approximate nearest neighbor (HNSW, IVF, LSH), bloom filters, consistent hashing, reservoir sampling

---

## 3. Databases & SQL

### 3.1 SQL
- [ ] **Queries:** `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, aliases
- [ ] **Aggregation:** `GROUP BY`, `HAVING`, `COUNT SUM AVG MIN MAX`
- [ ] **Joins:** inner, left, right, full, cross, self
- [ ] **Advanced queries:** subqueries, correlated subqueries, CTEs (`WITH`), recursive CTEs
- [ ] **Window functions:** `ROW_NUMBER RANK DENSE_RANK LAG LEAD`, `PARTITION BY`, running totals
- [ ] **Modifying data:** `INSERT UPDATE DELETE UPSERT`, `CASE WHEN`, `COALESCE`, date functions
- [ ] **Schema design:** DDL (`CREATE ALTER DROP`), constraints, primary/foreign keys
- [ ] **Normalization:** 1NF–3NF, BCNF, denormalization
- [ ] **Indexes:** B-tree, hash, composite, covering, `EXPLAIN ANALYZE`, query optimization
- [ ] **Transactions:** ACID, isolation levels, locks, MVCC
- [ ] Views, materialized views, stored procedures, triggers

### 3.2 Relational Databases
- [ ] **PostgreSQL (deep):**
  - JSONB, full-text search, extensions
  - **pgvector** for embeddings
  - Partitioning, replication
- [ ] MySQL, SQLite (local and edge apps), DuckDB (analytics, in-process)
- [ ] **ORMs:** SQLAlchemy (core + ORM), SQLModel, Alembic migrations, connection pooling (PgBouncer)

### 3.3 NoSQL
- [ ] **Key-value:** Redis (see §18.6), DynamoDB
- [ ] **Document:** MongoDB (documents, aggregation pipeline, indexes)
- [ ] **Wide-column:** Cassandra, ScyllaDB, HBase
- [ ] **Graph:** Neo4j, Cypher (used in GraphRAG / knowledge graphs)
- [ ] **Search engines:** Elasticsearch/OpenSearch (inverted index, BM25, analyzers)
- [ ] **Time-series:** InfluxDB, TimescaleDB
- [ ] **Theory:** CAP theorem, BASE vs ACID, when to use SQL vs NoSQL

---

## 4. Mathematics for ML

### 4.1 Linear Algebra
- [ ] Scalars, vectors, matrices, tensors
- [ ] Vector operations, dot product, cross product, norms (L1, L2, L∞, Frobenius)
- [ ] Matrix multiplication, transpose, identity, inverse, determinant, trace
- [ ] Linear independence, span, basis, rank, null space, orthogonality
- [ ] Linear transformations, projections, Gram-Schmidt
- [ ] Eigenvalues/eigenvectors, eigendecomposition, diagonalization
- [ ] **SVD** (used in PCA, LoRA intuition, compression), low-rank approximation
- [ ] Positive definite matrices, matrix calculus, cosine similarity

### 4.2 Calculus
- [ ] Limits, derivatives, rules (product, quotient, chain)
- [ ] Partial derivatives, gradients, directional derivatives
- [ ] Jacobian, Hessian, Taylor series
- [ ] Integrals (for probability), multivariable calculus
- [ ] **Backpropagation** as the chain rule on computational graphs, automatic differentiation (forward vs reverse mode)

### 4.3 Probability
- [ ] Sample spaces, events, axioms, conditional probability, independence
- [ ] **Bayes' theorem**, law of total probability
- [ ] Random variables (discrete/continuous), PMF, PDF, CDF
- [ ] Expectation, variance, covariance, correlation
- [ ] **Distributions:** Bernoulli, binomial, Poisson, geometric, uniform, normal/Gaussian, exponential, beta, gamma, Dirichlet, categorical, multivariate Gaussian
- [ ] Joint, marginal, and conditional distributions
- [ ] Law of large numbers, Central Limit Theorem
- [ ] **Estimation:** MLE, MAP, Bayesian inference, conjugate priors
- [ ] **Sampling:** Monte Carlo, importance sampling, MCMC (Metropolis-Hastings, Gibbs)

### 4.4 Statistics
- [ ] Descriptive stats: mean, median, mode, std, percentiles, skewness, kurtosis
- [ ] Sampling methods, bias, standard error
- [ ] **Hypothesis testing:** null/alternative, p-values, Type I/II errors, power
- [ ] **Tests:** t-test, chi-square, ANOVA, Mann-Whitney
- [ ] Confidence intervals, bootstrapping
- [ ] **A/B testing:** sample size, significance, multiple testing (Bonferroni), sequential testing
- [ ] Correlation vs causation, causal inference basics (DAGs, propensity scores, diff-in-diff)
- [ ] Regression analysis, residuals, multicollinearity

### 4.5 Optimization
- [ ] Convex vs non-convex functions, local vs global minima, saddle points
- [ ] Gradient descent: batch, stochastic, mini-batch
- [ ] Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW, Lion, Muon, Shampoo
- [ ] Learning rate schedules, second-order methods (Newton, L-BFGS)
- [ ] Constrained optimization, Lagrange multipliers, KKT conditions (for SVM)

### 4.6 Information Theory
- [ ] Entropy, cross-entropy, KL divergence (forward vs **reverse KL**, which OPD uses)
- [ ] Mutual information, perplexity, bits-per-byte
- [ ] **Proper scoring rules:** log loss, Brier score (the basis of RLCD, §11–12)

---

## 5. Data Analysis & Feature Engineering

### 5.1 NumPy
- [ ] ndarray, dtypes, shape, reshape, flatten, ravel
- [ ] Indexing, slicing, boolean masks, fancy indexing
- [ ] **Broadcasting**, vectorization, ufuncs
- [ ] Aggregations, axis operations, `np.linalg`, `np.random`
- [ ] Stacking, splitting, views vs copies, memory layout

### 5.2 Pandas
- [ ] Series, DataFrame, reading CSV/Excel/JSON/Parquet/SQL
- [ ] Selection: `loc`, `iloc`, boolean filtering, `query`
- [ ] **Cleaning:** missing values (`isna fillna dropna`), duplicates, type casting
- [ ] **Transforming:** `apply`, `map`, vectorized string (`.str`) and datetime (`.dt`) methods
- [ ] **Grouping and reshaping:** `groupby`, `agg`, `transform`, pivot tables, `melt`, `stack/unstack`
- [ ] **Combining:** `merge`, `join`, `concat`
- [ ] **Time series:** resampling, rolling windows, shifting
- [ ] Performance: categoricals, chunking
- [ ] **Alternatives:** Polars (fast, lazy), DuckDB, Dask/Modin for big data

### 5.3 Visualization
- [ ] Matplotlib (figure/axes, subplots, customization)
- [ ] Seaborn (distplots, heatmaps, pairplots)
- [ ] Plotly (interactive), Altair
- [ ] Dashboards: Streamlit, Dash, Tableau/Power BI basics

### 5.4 EDA
- [ ] Univariate/bivariate/multivariate analysis
- [ ] Distributions, outliers (IQR, z-score), correlation matrices
- [ ] Data quality checks, profiling (ydata-profiling)
- [ ] Target leakage detection

### 5.5 Feature Engineering
- [ ] **Encoding:** one-hot, label, ordinal, target/mean, frequency, hashing, embeddings
- [ ] **Scaling:** standardization, min-max, robust scaling, log/Box-Cox/Yeo-Johnson transforms
- [ ] Binning, polynomial features, interaction features
- [ ] Date/time features, text features (TF-IDF, embeddings), geospatial features
- [ ] **Missing value imputation:** mean/median, KNN, iterative/MICE
- [ ] **Imbalanced data:** oversampling, undersampling, SMOTE/ADASYN, class weights
- [ ] **Feature selection:** filter (correlation, chi-square, mutual info), wrapper (RFE), embedded (L1, tree importance)
- [ ] Dimensionality reduction (see §6.4)
- [ ] Data sources: web scraping (requests, BeautifulSoup, Playwright), REST APIs, public datasets (Kaggle, HF Datasets, UCI)

### 5.6 Data Labeling & Synthetic Data
- [ ] **Annotation tools:** Label Studio, CVAT, Labelbox
- [ ] Labeling guidelines, inter-annotator agreement
- [ ] Active learning, weak supervision (Snorkel)
- [ ] **Synthetic data generation** with LLMs, data augmentation, privacy-preserving synthetic data

---

## 6. Classical Machine Learning

### 6.1 Core Concepts
- [ ] **Learning types:** supervised, unsupervised, semi-supervised, self-supervised, reinforcement
- [ ] Training/validation/test splits, stratification, time-based splits
- [ ] **Bias-variance tradeoff**, overfitting, underfitting
- [ ] **Cross-validation:** k-fold, stratified, group, time-series CV
- [ ] Data leakage, baseline models
- [ ] Regularization: L1, L2, elastic net
- [ ] No free lunch theorem, curse of dimensionality

### 6.2 Regression
- [ ] Linear regression (OLS, normal equation, gradient descent), assumptions
- [ ] Polynomial regression, Ridge, Lasso, Elastic Net
- [ ] Quantile regression, Huber regression, Poisson/GLMs

### 6.3 Classification
- [ ] Logistic regression (sigmoid, log loss, softmax/multinomial)
- [ ] K-Nearest Neighbors, distance metrics
- [ ] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [ ] **Support Vector Machines:** margins, soft margin, kernels (linear, RBF, polynomial), kernel trick
- [ ] Decision trees (Gini, entropy, information gain, pruning)
- [ ] Multi-class strategies (OvR, OvO), multi-label classification

### 6.4 Ensembles
- [ ] Bagging, Random Forest, Extra Trees
- [ ] **Boosting:** AdaBoost, Gradient Boosting, **XGBoost, LightGBM, CatBoost**
- [ ] Stacking, blending, voting

### 6.5 Unsupervised Learning
- [ ] **Clustering:** K-Means (k-means++, elbow, silhouette), K-Medoids, hierarchical, DBSCAN, HDBSCAN, Gaussian Mixture Models (EM algorithm), spectral clustering
- [ ] **Dimensionality reduction:** PCA, kernel PCA, LDA, ICA, NMF, t-SNE, UMAP
- [ ] **Anomaly detection:** Isolation Forest, One-Class SVM, LOF, autoencoders
- [ ] Association rules (Apriori, FP-Growth)

### 6.6 Evaluation Metrics
- [ ] **Classification:**
  - Accuracy, precision, recall, F1, F-beta
  - ROC-AUC, PR-AUC, confusion matrix
  - Log loss, MCC, Cohen's kappa
- [ ] **Calibration:** reliability diagrams, ECE, Platt scaling, isotonic regression (matters for §12)
- [ ] **Regression:** MSE, RMSE, MAE, MAPE, R², adjusted R²
- [ ] **Ranking:** MAP, MRR, NDCG, Hit@K, Recall@K
- [ ] **Clustering:** silhouette, Davies-Bouldin, ARI, NMI
- [ ] Threshold tuning, cost-sensitive evaluation

### 6.7 Tuning, Tools & Explainability
- [ ] **Hyperparameter tuning:** grid search, random search, Bayesian optimization (**Optuna**, Hyperopt), early stopping
- [ ] **scikit-learn:** estimators API, `Pipeline`, `ColumnTransformer`, custom transformers, model persistence (joblib)
- [ ] **Explainability:** feature importance, permutation importance, **SHAP**, LIME, partial dependence plots, ICE
- [ ] AutoML: AutoGluon, H2O, FLAML

---

## 7. Deep Learning

### 7.1 Neural Network Fundamentals
- [ ] Perceptron, multi-layer perceptron, universal approximation theorem
- [ ] **Activations:** sigmoid, tanh, ReLU, Leaky ReLU, ELU, GELU, SiLU/Swish, **SwiGLU**, softmax
- [ ] Forward propagation, **backpropagation**, computational graphs
- [ ] **Loss functions:**
  - MSE, MAE, Huber, BCE, categorical cross-entropy
  - Hinge, focal, contrastive/InfoNCE, triplet, KL divergence
- [ ] **Initialization:** Xavier/Glorot, He/Kaiming

### 7.2 Training Deep Networks
- [ ] **Optimizers:** SGD, momentum, Adam, **AdamW**, Adafactor, Lion, Muon
- [ ] **LR schedules:** step, cosine, warmup, one-cycle, WSD (warmup-stable-decay)
- [ ] **Regularization:** dropout, weight decay, data augmentation, early stopping, label smoothing, mixup/cutmix
- [ ] **Normalization:** BatchNorm, LayerNorm, GroupNorm, **RMSNorm**, pre-norm vs post-norm
- [ ] **Problems and fixes:**
  - Vanishing/exploding gradients, dead ReLUs
  - Gradient clipping, residual connections
- [ ] **Mixed precision:** FP32, FP16, **BF16**, FP8, FP4, loss scaling
- [ ] Gradient accumulation, gradient checkpointing
- [ ] Batch size effects, learning rate finder, reproducibility (seeds)
- [ ] Debugging training: overfit a single batch, loss curves, NaN hunting

### 7.3 PyTorch (Primary Framework)
- [ ] Tensors, dtypes, devices (`cpu`/`cuda`/`mps`), broadcasting
- [ ] Autograd, `requires_grad`, `backward()`, `no_grad`, `detach`
- [ ] `nn.Module`, layers (`Linear Conv2d Embedding LSTM MultiheadAttention`), `nn.Sequential`
- [ ] `Dataset`, `DataLoader`, samplers, collate functions, workers
- [ ] **Training loop:** zero_grad → forward → loss → backward → step; eval mode
- [ ] Saving/loading (`state_dict`), checkpointing, `safetensors`
- [ ] `torch.compile`, AMP (`autocast`, `GradScaler`), custom autograd functions
- [ ] **Distributed:** DDP, FSDP/FSDP2 (see §32)
- [ ] **Ecosystem:** PyTorch Lightning, Hugging Face Accelerate, torchmetrics, TorchVision, TorchAudio
- [ ] **Also know:** TensorFlow/Keras 3 (functional API, `tf.data`), **JAX** (jit, grad, vmap, pmap, Flax)

### 7.4 Sequence Models (Pre-Transformer)
- [ ] RNN, backprop through time, LSTM (gates), GRU, bidirectional RNNs
- [ ] Seq2seq encoder-decoder, teacher forcing
- [ ] Attention (Bahdanau, Luong) as the bridge to Transformers

### 7.5 Other Architectures & Concepts
- [ ] Autoencoders (vanilla, denoising, sparse, variational: see §13)
- [ ] Embeddings and representation learning, metric learning, Siamese networks
- [ ] **Transfer learning:** feature extraction vs fine-tuning, freezing layers
- [ ] **Self-supervised learning:** contrastive (SimCLR, MoCo), masked modeling (MAE, BERT), JEPA
- [ ] Multi-task learning, meta-learning, few-shot learning
- [ ] **Mixture of Experts (MoE):** gating, top-k routing, load balancing
- [ ] **State Space Models:** S4, **Mamba**/Mamba-2, hybrid Transformer-SSM models
- [ ] Neural ODEs, KANs (Kolmogorov-Arnold Networks), capsule nets (awareness)
- [ ] **Mechanistic interpretability:** probing, attention visualization, sparse autoencoders, circuits, feature steering

---

## 8. Computer Vision

### 8.1 Image Fundamentals
- [ ] Pixels, channels, color spaces (RGB, BGR, HSV, LAB, grayscale), image formats
- [ ] **OpenCV:**
  - I/O, resizing, cropping, rotation, affine/perspective transforms
  - Filtering (blur, Gaussian, median), thresholding, morphology
  - Edges (Sobel, Canny), contours, histograms, histogram equalization
- [ ] Classical features: Harris corners, SIFT, SURF, ORB, HOG, feature matching, homography
- [ ] Camera models, calibration, stereo vision, optical flow

### 8.2 CNNs
- [ ] Convolution, kernels, stride, padding, dilation, receptive field
- [ ] Pooling (max, average, global), 1×1 convs, depthwise separable convs
- [ ] **Architectures:** LeNet, AlexNet, VGG, GoogLeNet/Inception, **ResNet**, DenseNet, MobileNet (v1–v4), EfficientNet, RegNet, ConvNeXt

### 8.3 Vision Transformers
- [ ] **ViT:** patches, CLS token, position embeddings
- [ ] Variants: DeiT, Swin, BEiT, MAE
- [ ] **Self-supervised:** DINO, **DINOv2/DINOv3**, iBOT
- [ ] Hybrid CNN-Transformer models

### 8.4 Tasks
- [ ] **Image classification:** augmentation (Albumentations, RandAugment), fine-tuning with `timm`
- [ ] **Object detection:**
  - Two-stage: R-CNN, Fast/Faster R-CNN, FPN
  - One-stage: YOLO (v1 → latest, Ultralytics), SSD, RetinaNet
  - Transformer-based: DETR, RT-DETR, Grounding DINO
  - Concepts: anchors vs anchor-free, NMS, IoU/GIoU, mAP@0.5:0.95
- [ ] **Segmentation:**
  - Semantic: FCN, U-Net, DeepLab
  - Instance and panoptic: Mask R-CNN, Mask2Former
  - Promptable: **SAM / SAM 2 / SAM 3** (images and video)
- [ ] **Keypoints & pose:** OpenPose, HRNet, YOLO-pose, MediaPipe
- [ ] **Face:** detection, recognition (ArcFace), landmarks, anti-spoofing
- [ ] **OCR & documents:** Tesseract, PaddleOCR, TrOCR, layout analysis, document AI (Donut, VLM-based OCR)
- [ ] **Tracking:** SORT, DeepSORT, ByteTrack, multi-object tracking
- [ ] **Video:** action recognition (3D CNNs, SlowFast, TimeSformer, VideoMAE), video understanding with VLMs
- [ ] **3D vision:** depth estimation (Depth Anything), point clouds (PointNet), NeRF, **3D Gaussian Splatting**, SLAM
- [ ] Image retrieval, super-resolution, denoising, image captioning, visual question answering (VQA)

### 8.5 Multimodal Vision
- [ ] **CLIP, SigLIP:** contrastive image-text, zero-shot classification
- [ ] **Vision-Language Models (VLMs):** LLaVA-style architecture (vision encoder + projector + LLM), Qwen-VL, InternVL, Gemma/Llama vision variants
- [ ] Open-vocabulary detection/segmentation, grounding
- [ ] **Vision-Language-Action (VLA) models** for robotics (awareness)

### 8.6 Tools
- [ ] torchvision, `timm`, Ultralytics, Detectron2, MMDetection, Supervision, Roboflow, FiftyOne, Kornia

---

## 9. NLP (Classic → Transformers)

### 9.1 Text Preprocessing
- [ ] Tokenization (word, sentence), normalization, lowercasing, stopwords
- [ ] Stemming (Porter), lemmatization, regex cleaning, Unicode handling
- [ ] **Libraries:** NLTK, spaCy, Gensim

### 9.2 Classic NLP
- [ ] Bag of Words, TF-IDF, n-grams, language models (n-gram, smoothing)
- [ ] POS tagging, NER, dependency parsing, chunking
- [ ] Topic modeling (LDA, NMF, BERTopic), sentiment analysis, text classification
- [ ] Hidden Markov Models, CRFs

### 9.3 Word Embeddings
- [ ] Word2Vec (CBOW, Skip-gram, negative sampling), GloVe, FastText
- [ ] Contextual embeddings (ELMo), sentence embeddings (Sentence-BERT)

### 9.4 The Transformer (Deep Understanding)
- [ ] "Attention Is All You Need": encoder-decoder architecture
- [ ] **Scaled dot-product attention:** Q, K, V, softmax, scaling by √d
- [ ] Multi-head attention, self vs cross-attention, causal masking, padding masks
- [ ] Feed-forward networks, residual connections, LayerNorm
- [ ] **Positional encoding:** sinusoidal, learned, **RoPE**, ALiBi, YaRN/NTK scaling for long context
- [ ] **Build a Transformer from scratch** in PyTorch (e.g., nanoGPT-style)

### 9.5 Transformer Families
- [ ] **Encoder-only:** BERT (MLM, NSP), RoBERTa, DistilBERT, ALBERT, DeBERTa, **ModernBERT** (Laya is built on it), mmBERT
- [ ] **Decoder-only:** GPT-1/2/3 lineage, Llama, Mistral, Qwen, DeepSeek, Gemma, Phi, gpt-oss
- [ ] **Encoder-decoder:** T5, BART, mT5, Flan-T5
- [ ] **Tasks:** classification, NER, QA (extractive/generative), summarization, translation, paraphrase, NLI

### 9.6 Tokenizers
- [ ] **Subword algorithms:** BPE, byte-level BPE, WordPiece, Unigram, SentencePiece
- [ ] `tiktoken`, HF `tokenizers`, vocab size tradeoffs, special tokens, chat templates
- [ ] Multilingual tokenization issues, tokenizer-free/byte-level models (awareness)

---

## 10. Large Language Models (Internals & Training)

### 10.1 LLM Architecture (Modern)
- [ ] Decoder-only Transformer stack, embedding and unembedding (tied weights)
- [ ] **Attention variants:** MHA, **MQA**, **GQA**, **MLA** (DeepSeek), sliding window, sparse attention (e.g., DeepSeek sparse attention)
- [ ] **FlashAttention** (v1–v3+), PagedAttention, linear attention
- [ ] **Mixture of Experts:** DeepSeek-MoE, shared experts, expert parallelism
- [ ] RMSNorm, SwiGLU, RoPE, QK-norm
- [ ] **Long context:** 128K–1M+ tokens, context extension, "lost in the middle"
- [ ] **Diffusion LLMs** (e.g., LLaDA) as a non-autoregressive alternative (awareness)
- [ ] **Multimodal LLMs:** vision/audio encoders, any-to-any models, native image generation

### 10.2 Pretraining
- [ ] Next-token prediction objective, causal language modeling
- [ ] **Data:**
  - Sources: Common Crawl, deduplication (MinHash), quality filtering
  - Datasets: FineWeb/FineWeb-Edu-style data, code data, data mixtures
  - Synthetic pretraining data
- [ ] **Scaling laws:** Kaplan, **Chinchilla** (compute-optimal), over-training for inference efficiency
- [ ] Training stability: loss spikes, µP (maximal update parametrization), warmup, z-loss
- [ ] Mid-training / continued pretraining, annealing, long-context stages
- [ ] Compute budgets: FLOPs estimation (6ND), MFU, GPU-hours

### 10.3 Fine-Tuning
- [ ] Full fine-tuning vs **PEFT**
- [ ] **LoRA** (rank, alpha, target modules), **QLoRA** (4-bit NF4), DoRA, adapters, prefix/prompt tuning, IA³
- [ ] Instruction tuning datasets, chat templates, packing, loss masking
- [ ] Domain adaptation, catastrophic forgetting, model merging (TIES, DARE, SLERP)
- [ ] **Tools:** HF Transformers, **PEFT**, **TRL**, **Unsloth**, Axolotl, LLaMA-Factory, torchtune

### 10.4 Decoding & Inference Behavior
- [ ] Greedy, beam search, sampling, temperature, top-k, top-p (nucleus), min-p
- [ ] Repetition/frequency/presence penalties, stop sequences
- [ ] **Structured decoding:** JSON mode, grammar/constrained decoding (Outlines, XGrammar)
- [ ] **Speculative decoding:** draft models, Medusa, EAGLE, DFlash
- [ ] KV cache, prefix caching, logprobs

### 10.5 Reasoning Models
- [ ] Chain-of-thought, self-consistency, tree/graph of thoughts
- [ ] **Test-time compute scaling:** long thinking, budget control, "thinking" vs "non-thinking" modes
- [ ] Process reward models vs outcome reward models
- [ ] RL for reasoning (DeepSeek-R1 style, see §11), verifiers, best-of-N, self-verification

### 10.6 Model Compression
- [ ] **Quantization:**
  - Post-training: INT8/INT4/FP8/FP4
  - Methods and formats: **GPTQ, AWQ**, SmoothQuant, **GGUF** (llama.cpp), bitsandbytes, HQQ, EXL2
  - Quantization-aware training
- [ ] **Knowledge distillation:** logit distillation, sequence-level distillation, **on-policy distillation** (§11)
- [ ] Pruning (structured/unstructured), sparsity (2:4), small language models (SLMs)

### 10.7 LLM Evaluation & Benchmarks
- [ ] Perplexity, held-out loss
- [ ] **Knowledge and reasoning:** MMLU/MMLU-Pro, GPQA, Humanity's Last Exam
- [ ] **Math:** GSM8K, MATH, AIME
- [ ] **Code:** HumanEval, **SWE-bench**, LiveCodeBench
- [ ] **Agents:** Terminal-Bench, τ-bench, GAIA
- [ ] **Chat and instruction following:** Chatbot Arena/LMArena, IFEval, long-context (RULER, needle-in-a-haystack)
- [ ] Contamination, benchmark saturation, **lm-evaluation-harness**

---

## 11. Post-Training & Alignment

The full lineage, from oldest to newest:

- [ ] **Supervised Fine-Tuning (SFT):** demonstrations, instruction following, rejection sampling
- [ ] **RLHF (Reinforcement Learning from Human Feedback):**
  - Preference data collection (pairwise comparisons)
  - Reward model training (Bradley-Terry)
  - **PPO** with KL penalty to a reference model
  - Issues: reward hacking, sycophancy, overconfidence, mode collapse, cost
- [ ] **Constitutional AI / RLAIF (RL from AI Feedback):** AI-generated preferences, critiques, and revisions against a set of principles
- [ ] **Direct preference methods (no reward model):**
  - **DPO**
  - Variants: IPO, KTO, ORPO, SimPO
  - Online DPO, iterative DPO
- [ ] **RLVR (RL from Verifiable Rewards):** deterministic rewards (correct answer, unit tests pass), no learned reward model
- [ ] **GRPO** (Group Relative Policy Optimization, DeepSeekMath/R1) and variants (DAPO, Dr. GRPO, GSPO), REINFORCE/RLOO
- [ ] **On-Policy Distillation (OPD), the 2026 standard:**
  - Student samples its own trajectories
  - Teacher grades every token (per-token reverse KL)
  - Dense signal, far cheaper than RL
  - Used to fuse separately-RL'd specialist models
- [ ] **OPD variants:** self-distillation, Distilled RL, cross-stage distillation, black-box OPD
- [ ] **RLCD (Reinforcement Learning for Calibrated Decisions):**
  - Optimizes probabilities against outcomes using proper scoring rules, instead of human preference
  - Used by System One models (§12)
- [ ] **RL environments:** long-horizon agent tasks, verifiers, environment hubs (e.g., Prime Intellect)
- [ ] **Agentic RL:** multi-turn tool use, coding/computer-use environments
- [ ] **Safety training:** refusals, harmlessness vs helpfulness, red-teaming data, deliberative alignment
- [ ] **Frameworks:** TRL, **verl**, OpenRLHF, NeMo-RL, Miles, slime, SkyRL

---

## 12. System One / Decision Models (New: Sept 2026)

A new model class that returns **typed, calibrated decisions instead of generated text**. The name comes from Kahneman's System 1: fast and intuitive, as opposed to System 2's slow reasoning.

- [ ] **Concepts:**
  - System 1 vs System 2 thinking
  - Non-autoregressive, single-pass parallel inference
  - Typed outputs cannot hallucinate outside the defined schema
- [ ] **Interface:** give it a *state* (messy text, JSON, a ticket, an agent trace) plus typed *questions*, and it returns answers with probabilities
- [ ] **Question primitives:**
  - **Choice:** pick from enumerated options
  - **Score:** rate against a rubric
  - **Noul:** probability that the answer is yes (0–1)
- [ ] **Training:** RLCD, with proper scoring rules (Brier score, log loss) and calibration
- [ ] **TypeSafe Jev** (Sept 15, 2026):
  - Hosted API from TypeSafe AI, founded by Diogo Almeida (co-creator of RLHF and InstructGPT)
  - Claims 70–500 ms latency, 40–200× faster and 40–400× cheaper than frontier LLMs on decision tasks
  - Its benchmarks are vendor-run, so test on your own data
  - Tooling: LangChain integration, MCP server, Vercel AI Gateway
- [ ] **Convai Laya** (Sept 18, 2026):
  - Open weights (Apache-2.0); a 421M ModernBERT-large checkpoint, plus a 322M multilingual one (100+ languages)
  - About 33 ms per question on a T4, about 7 ms batched
  - Runtimes: `pip install laya`, ONNX, MLX, CoreML ports
  - Can be fine-tuned locally
- [ ] **Use cases:**
  - Routing, moderation, intent detection, triage, relevance scoring
  - Safety checks, escalation decisions, and choosing when to call a big LLM or a human
  - Real-time control (games, robots, simulations)
- [ ] **Hybrid architecture:** a decision model as a cheap first layer, an LLM only for generation and deep reasoning, plus confidence thresholds for fallback
- [ ] **Not for:** chat, code generation, or anything that needs a written explanation

---

## 13. Generative AI (Images, Video, Audio, 3D)

### 13.1 Foundations
- [ ] Generative vs discriminative models, likelihood-based vs implicit models
- [ ] Latent spaces, sampling, evaluation (FID, IS, CLIP score, human eval)

### 13.2 VAEs
- [ ] Encoder/decoder, ELBO, reparameterization trick, KL term
- [ ] β-VAE, VQ-VAE, VQ-GAN (discrete latents)

### 13.3 GANs
- [ ] Generator vs discriminator, minimax loss, training instability, mode collapse
- [ ] DCGAN, WGAN/WGAN-GP, conditional GAN, Pix2Pix, CycleGAN, StyleGAN (1–3), BigGAN

### 13.4 Autoregressive & Flow Models
- [ ] PixelCNN, image GPT, autoregressive image tokens
- [ ] Normalizing flows (RealNVP, Glow)

### 13.5 Diffusion Models
- [ ] Forward/reverse process, noise schedules, **DDPM**, score matching, SDE view
- [ ] Samplers: DDIM, DPM-Solver, Euler; guidance: classifier and **classifier-free guidance**
- [ ] **Latent diffusion** (Stable Diffusion), VAE latent space, U-Net backbone
- [ ] **Diffusion Transformers (DiT)**, MMDiT, **flow matching / rectified flow** (SD3, FLUX-style models)
- [ ] Consistency models, distillation for few-step generation (LCM, turbo/lightning variants)
- [ ] **Conditioning and control:** text encoders (CLIP, T5), **ControlNet**, IP-Adapter, T2I-Adapter
- [ ] **Customization:** **LoRA** for diffusion, DreamBooth, textual inversion
- [ ] **Editing:** img2img, inpainting, outpainting, instruction-based editing
- [ ] **Tools:** HF **Diffusers**, **ComfyUI**, Automatic1111/Forge, SGLang Diffusion

### 13.6 Video Generation
- [ ] Text-to-video and image-to-video, spatiotemporal DiTs, video VAEs
- [ ] Open models (Wan, HunyuanVideo, LTX-style) vs closed APIs
- [ ] World models (interactive, game-like generation)

### 13.7 3D Generation
- [ ] NeRF, 3D Gaussian Splatting, text-to-3D, image-to-3D, score distillation

### 13.8 Multimodal / Native Generation
- [ ] Native image generation inside LLMs, unified understanding and generation models, any-to-any models

---

## 14. Speech & Audio AI

- [ ] **Audio basics:** sampling rate, waveforms, spectrograms, **mel spectrograms**, MFCCs, `librosa`, `torchaudio`
- [ ] **Speech-to-text (ASR):** CTC, RNN-T, **Whisper** (and faster-whisper, whisper.cpp), wav2vec 2.0, streaming ASR
- [ ] **Text-to-speech (TTS):** Tacotron, FastSpeech, VITS, neural codec TTS, voice cloning, open TTS models
- [ ] Neural audio codecs (EnCodec, SoundStream), audio tokens
- [ ] **Speech-to-speech / voice agents:** full-duplex models, realtime APIs, latency budgets, turn-taking, VAD (Silero)
- [ ] Speaker diarization (pyannote), speaker verification
- [ ] Music generation, sound effects, audio classification
- [ ] **Voice AI stack:** WebRTC, LiveKit/Pipecat, telephony (Twilio)

---

## 15. Reinforcement Learning

- [ ] **Basics:** agent, environment, state, action, reward, policy, return, discount factor
- [ ] **MDPs**, Bellman equations, value functions (V, Q)
- [ ] **Planning:** dynamic programming, policy iteration, value iteration
- [ ] **Model-free methods:** Monte Carlo, TD learning, SARSA, **Q-learning**
- [ ] **Exploration:** ε-greedy, UCB, Thompson sampling, **multi-armed bandits**, contextual bandits
- [ ] **Deep RL:** **DQN** (replay buffer, target network), Double/Dueling DQN
- [ ] **Policy gradients:** REINFORCE, baselines, **advantage**, Actor-Critic, A2C/A3C
- [ ] **PPO** (clipping, GAE), TRPO, SAC, TD3, DDPG
- [ ] Model-based RL (MuZero, Dreamer), offline RL, imitation learning (behavior cloning, DAgger), inverse RL
- [ ] Multi-agent RL, reward shaping, sim-to-real
- [ ] **Tools:** Gymnasium, Stable-Baselines3, CleanRL, RLlib
- [ ] **Link to LLMs:** RLHF/GRPO/RLVR/RLCD all build on these foundations (§11)

---

## 16. Other ML Areas

### 16.1 Recommender Systems
- [ ] Content-based, collaborative filtering (user/item-based)
- [ ] Matrix factorization (SVD, ALS)
- [ ] Implicit feedback, cold start
- [ ] **Two-tower models**, candidate generation → ranking → re-ranking pipeline
- [ ] Deep recsys: Wide & Deep, DeepFM, DLRM, sequential recommenders (SASRec), generative recommenders
- [ ] **Evaluation:** offline (NDCG, Recall@K) vs online (CTR, A/B tests)

### 16.2 Time Series
- [ ] Stationarity, decomposition, autocorrelation (ACF/PACF)
- [ ] **Statistical models:** AR, MA, ARIMA/SARIMA, exponential smoothing, Prophet
- [ ] ML-based forecasting (lag features, gradient boosting)
- [ ] **Deep models:** LSTM, TCN, N-BEATS, Temporal Fusion Transformer, PatchTST
- [ ] **Foundation models:** Chronos, TimesFM, Moirai
- [ ] Anomaly detection in time series, backtesting

### 16.3 Graph ML
- [ ] Graph basics, node/edge/graph-level tasks
- [ ] Node2Vec, **GCN**, GraphSAGE, GAT
- [ ] Knowledge graphs, link prediction
- [ ] **Tools:** PyTorch Geometric, DGL

### 16.4 Tabular Deep Learning & Other
- [ ] TabNet, FT-Transformer, **TabPFN** (tabular foundation models)
- [ ] Survival analysis, causal ML (uplift modeling, EconML, DoWhy)
- [ ] Federated learning, differential privacy, continual learning, active learning
- [ ] AI for science (protein structure models, weather models), robotics, autonomous driving (awareness)

---

## 17. LLM Application Engineering

### 17.1 Working with LLM APIs
- [ ] **Providers:** Anthropic (Claude), OpenAI, Google (Gemini), Mistral, xAI, DeepSeek, open-model hosts (Together, Fireworks, Groq, OpenRouter)
- [ ] **Chat API concepts:** system/user/assistant messages, multi-turn state, parameters
- [ ] **Streaming** (SSE), token counting, context window management, rate limits and retries (exponential backoff)
- [ ] **Tool / function calling:** JSON schemas, parallel tool calls, tool results
- [ ] **Structured outputs:** JSON schema, Pydantic, Instructor
- [ ] Vision/audio/PDF inputs, embeddings APIs, batch APIs
- [ ] Prompt caching, cost tracking
- [ ] Gateways: LiteLLM, OpenRouter, Vercel AI Gateway

### 17.2 Prompt & Context Engineering
- [ ] Zero-shot, few-shot, role/system prompts, delimiters/XML tags
- [ ] Chain-of-thought, ReAct, self-critique, decomposition, prompt chaining
- [ ] **Context engineering:** what goes in the window, summarization, compaction, memory injection
- [ ] Prompt templates and versioning, prompt optimization (**DSPy**, automatic prompt tuning)
- [ ] **Prompt injection** (direct and indirect) and defenses, jailbreaks

### 17.3 Embeddings
- [ ] Embedding models (open and API), dimensions, **Matryoshka embeddings**, multilingual and multimodal embeddings
- [ ] Similarity metrics: cosine, dot product, Euclidean
- [ ] Fine-tuning embeddings (sentence-transformers), MTEB leaderboard

### 17.4 RAG (Retrieval-Augmented Generation)
- [ ] **Ingestion:**
  - Document loaders and PDF parsing (Unstructured, Docling, LlamaParse, VLM-based parsing)
  - Tables and images, metadata extraction
- [ ] **Chunking:** fixed-size, recursive, semantic, document-structure-aware, late chunking, overlap
- [ ] Indexing into a vector DB (§19), metadata filtering
- [ ] **Retrieval:** dense, sparse (**BM25**, SPLADE), **hybrid search** with fusion (RRF)
- [ ] **Reranking:** cross-encoders, rerank APIs, ColBERT (late interaction)
- [ ] **Query transformation:** rewriting, HyDE, multi-query, decomposition, step-back prompting
- [ ] **Advanced patterns:**
  - Parent-document and small-to-big retrieval, contextual retrieval
  - **GraphRAG**, knowledge graphs
  - **Agentic RAG**, corrective/self-RAG
  - Multimodal RAG (ColPali-style page retrieval)
- [ ] Citations and grounding, handling "no answer"
- [ ] **RAG evaluation:** context precision/recall, faithfulness, answer relevance (RAGAS, TruLens)
- [ ] **Tradeoffs:** RAG vs long context vs fine-tuning

### 17.5 AI Agents
- [ ] **What is an agent:** an LLM plus tools in a loop, with autonomy levels
- [ ] **Patterns:**
  - ReAct, plan-and-execute, reflection
  - Routing, parallelization, orchestrator-workers, evaluator-optimizer
- [ ] **Tool design:** clear schemas, error messages, idempotency, permissions
- [ ] **Memory:** short-term (conversation), long-term (vector/DB), episodic/semantic, memory frameworks (Mem0, Letta)
- [ ] **Multi-agent systems:** supervisor, hierarchical, handoffs, subagents, debate
- [ ] **Protocols:**
  - **MCP (Model Context Protocol):**
    - Primitives: tools, resources, prompts
    - Transports: stdio, Streamable HTTP
    - Building MCP servers and clients, auth (OAuth), tool poisoning risks
  - **A2A (Agent-to-Agent):** Agent Cards, task lifecycle (ACP merged into A2A); both are now governed under the Linux Foundation's Agentic AI Foundation
- [ ] **Computer use and browser agents:** Playwright, browser-use style frameworks, screenshots-and-actions loops
- [ ] **Coding agents:** repo navigation, sandboxed execution, SWE-bench-style tasks
- [ ] Sandboxing (E2B, Docker, Firecracker), code execution safety
- [ ] Human-in-the-loop, approvals, interrupts, durable execution (Temporal, LangGraph checkpoints)
- [ ] **Frameworks:** Claude Agent SDK, OpenAI Agents SDK, Google ADK, **LangGraph**, LangChain, LlamaIndex, CrewAI, Pydantic AI, smolagents, Microsoft Agent Framework (AutoGen + Semantic Kernel), Mastra (TypeScript), Vercel AI SDK
- [ ] **Hybrid agents:** System One decision models (§12) for routing and gating, LLMs for generation

### 17.6 Evaluation (Most Underrated Skill)
- [ ] Golden datasets, test cases, regression suites, error analysis
- [ ] **LLM-as-judge:** rubrics, pairwise comparison, bias mitigation, judge calibration
- [ ] Code-based evals (exact match, regex, schema validation, unit tests)
- [ ] **Agent evals:** trajectory evaluation, tool-call accuracy, task success rate, cost and latency per task
- [ ] Hallucination detection, groundedness checks
- [ ] Online evals, user feedback, A/B testing prompts and models
- [ ] **Tools:** promptfoo, DeepEval, RAGAS, Braintrust, LangSmith, Langfuse, Arize Phoenix, Inspect, OpenAI Evals

### 17.7 Guardrails & Safety in Apps
- [ ] Input/output filtering, PII detection/redaction (Presidio)
- [ ] Moderation models (Llama Guard-style, System One classifiers)
- [ ] Topic restriction, jailbreak detection, output validation (Guardrails AI, NeMo Guardrails)
- [ ] Rate limiting and abuse prevention

### 17.8 Fine-Tune or Not?
- [ ] Decision framework: prompting → RAG → fine-tuning → distillation
- [ ] Distilling a large model into a small one for cost; fine-tuning a System One model for decisions

---

## 18. Backend Engineering for AI

### 18.1 API Fundamentals
- [ ] **REST principles:** resources, HTTP methods (GET, POST, PUT, PATCH, DELETE), idempotency, status codes
- [ ] **API design:** versioning (`/v1`), pagination (offset vs cursor), filtering, sorting, error formats (RFC 7807)
- [ ] OpenAPI/Swagger specs, API documentation
- [ ] **Alternatives:** **GraphQL** (schemas, resolvers), **gRPC** (protobuf, streaming), WebSockets, **Server-Sent Events** (for LLM token streaming), webhooks
- [ ] Sync vs async request handling, long-running jobs (submit, poll, or callback)

### 18.2 FastAPI (Deep)
- [ ] **Routing:** app, path operations, `APIRouter`, tags
- [ ] Path params, query params, request body, headers, cookies, form data, file uploads (`UploadFile`)
- [ ] **Pydantic v2:** models, validation, field constraints, validators, `model_config`, response models
- [ ] **Dependency injection:** `Depends`, sub-dependencies, yield dependencies (DB sessions)
- [ ] **async vs sync endpoints:** when to use each, avoiding blocking the event loop, `run_in_threadpool`
- [ ] **Streaming responses:** `StreamingResponse`, SSE for LLM tokens, WebSocket endpoints
- [ ] Middleware, CORS, exception handlers, custom error responses
- [ ] Background tasks, lifespan events (loading models once at startup)
- [ ] **Security:** OAuth2 password flow, JWT, API keys, scopes
- [ ] **Databases:** SQLAlchemy/SQLModel async sessions, Alembic migrations
- [ ] **Testing:** `TestClient`, `httpx.AsyncClient`, pytest fixtures
- [ ] **Serving:** Uvicorn, Gunicorn with Uvicorn workers, number of workers, Granian
- [ ] Settings management (`pydantic-settings`, `.env`), project structure
- [ ] **Serving ML models in FastAPI:**
  - Load on startup, batch requests, keep GPU work off the event loop
  - Health and readiness endpoints

### 18.3 Other Python Backends
- [ ] **Flask:** blueprints, extensions
- [ ] **Django + Django REST Framework:** ORM, admin, auth
- [ ] Litestar, Starlette (FastAPI's base)
- [ ] **Node.js/TypeScript backends:** Express, NestJS, Hono (for AI apps built on the Vercel AI SDK)

### 18.4 Authentication & Authorization
- [ ] Sessions vs tokens, **JWT** (structure, signing, refresh tokens, expiry)
- [ ] **OAuth 2.0** flows (authorization code + PKCE, client credentials), **OpenID Connect**
- [ ] API keys and hashing, password hashing (bcrypt, argon2)
- [ ] **Access control:** RBAC, ABAC, multi-tenancy isolation
- [ ] **Auth providers:** Auth0, Clerk, Supabase Auth, Firebase Auth, Cognito, Keycloak
- [ ] SSO, SAML (enterprise), MFA

### 18.5 Background Jobs & Messaging
- [ ] **Task queues:** **Celery** (with Redis/RabbitMQ), RQ, Dramatiq, ARQ (async), Huey
- [ ] **Message brokers:**
  - **RabbitMQ:** exchanges, queues, routing, acks
  - **Kafka:** topics, partitions, consumer groups, offsets, retention
  - Redis Streams, AWS SQS/SNS, Google Pub/Sub, NATS
- [ ] **Patterns:** pub/sub, work queues, fan-out, dead-letter queues, retries, idempotency keys, exactly-once vs at-least-once delivery
- [ ] **Workflow engines:** Temporal (durable agents), Inngest, Prefect/Airflow for batch jobs
- [ ] Scheduled jobs (cron, Celery beat)

### 18.6 Redis (Deep)
- [ ] **Data types:** strings, hashes, lists, sets, sorted sets, streams, bitmaps, HyperLogLog, geospatial, JSON
- [ ] **Caching patterns:**
  - Cache-aside, write-through, write-behind
  - TTLs, eviction policies (LRU/LFU), cache invalidation, cache stampede prevention
- [ ] **Uses in AI apps:**
  - **Semantic caching** of LLM responses
  - Session and chat history storage, rate limiting (token bucket, sliding window)
  - Distributed locks, pub/sub, job queues
  - Redis as a **vector DB** (vector search)
- [ ] **Operations:** persistence (RDB, AOF), replication, Sentinel, Redis Cluster, managed Redis (ElastiCache, Upstash), Valkey
- [ ] Python clients: `redis-py` (sync/async)

### 18.7 Caching & Performance (General)
- [ ] Caching layers: browser, CDN, reverse proxy, application, database
- [ ] HTTP caching headers (`ETag`, `Cache-Control`)
- [ ] Connection pooling, async I/O, batching, compression (gzip, brotli)
- [ ] **Profiling and load testing:** Locust, k6, wrk
- [ ] Reverse proxies and web servers: **Nginx**, Caddy, Traefik, Envoy

### 18.8 Storage
- [ ] **Object storage:** **S3**, GCS, Azure Blob, MinIO (presigned URLs, multipart uploads, lifecycle rules)
- [ ] File storage for models and datasets, CDN delivery

### 18.9 Backend-as-a-Service (Fast Prototyping)
- [ ] Supabase (Postgres + pgvector + auth + storage), Firebase, Convex, Appwrite

---

## 19. Vector Databases & Search

- [ ] **Concepts:** embeddings, vector similarity, exact vs **approximate nearest neighbor (ANN)**
- [ ] **Index types:**
  - Flat (brute force)
  - **HNSW** (graph-based; M and ef parameters)
  - **IVF** (inverted file; nlist and nprobe), **PQ** (product quantization), IVF-PQ
  - ScaNN, DiskANN, binary and scalar quantization
- [ ] **Distance metrics:** cosine, dot product, L2, and the importance of normalization
- [ ] **Tradeoffs:** recall vs latency vs memory, index build time
- [ ] Metadata filtering (pre- vs post-filtering), namespaces/collections, multi-tenancy
- [ ] **Hybrid search:** dense + sparse (BM25/SPLADE), reciprocal rank fusion
- [ ] **CRUD on vectors:** upserts, deletes, re-embedding when models change
- [ ] Sharding, replication, and scaling to billions of vectors
- [ ] **Libraries:** **FAISS**, hnswlib, Annoy, ScaNN, USearch
- [ ] **Vector databases:**
  - Dedicated: **Pinecone**, **Qdrant**, **Weaviate**, **Milvus/Zilliz**, **Chroma**, LanceDB, Turbopuffer, Vespa
  - Built into existing stores: **pgvector** (+ pgvectorscale), Redis, Elasticsearch/OpenSearch, MongoDB Atlas Vector Search
- [ ] **Choosing one:** managed vs self-hosted, scale, filtering needs, cost
- [ ] **Search engineering:** inverted indexes, BM25 tuning, analyzers, learning-to-rank, search relevance evaluation

---

## 20. Frontend for AI Apps

- [ ] **Quick ML demo UIs:** **Streamlit**, **Gradio**, Chainlit (chat UIs), Panel
- [ ] **Web basics:** HTML, CSS (Flexbox, Grid), JavaScript (ES6+, promises, async/await, fetch)
- [ ] **TypeScript:** types, interfaces, generics
- [ ] **React:** components, props, state, hooks (`useState`, `useEffect`, `useRef`), context
- [ ] **Next.js:** App Router, server components, API routes, server actions, deployment on Vercel
- [ ] **Styling:** Tailwind CSS, shadcn/ui
- [ ] **AI UI patterns:**
  - Streaming token rendering, chat interfaces, markdown/code rendering
  - Loading and thinking states, tool-call visualization, citations display
  - File upload, voice input
- [ ] **Vercel AI SDK:** `useChat`, streaming, tool calls, generative UI
- [ ] Realtime: WebSockets/SSE on the client, WebRTC for voice
- [ ] Mobile basics (React Native/Expo) for AI apps (optional)

---

## 21. Data Engineering

- [ ] **Data architecture:** OLTP vs OLAP, data warehouse vs **data lake** vs **lakehouse**
- [ ] **ETL vs ELT**, batch vs streaming pipelines
- [ ] **File formats:** CSV, JSON, **Parquet**, Avro, ORC, Arrow
- [ ] **Table formats:** **Delta Lake**, **Apache Iceberg**, Hudi
- [ ] **Warehouses:** Snowflake, BigQuery, Redshift, Databricks SQL, ClickHouse
- [ ] **Big data processing:**
  - **Apache Spark / PySpark:** RDDs, DataFrames, Spark SQL, partitioning, shuffles
  - Ray Data, Dask, Polars for single-node work
- [ ] **Streaming:** **Kafka**, Kafka Connect, Flink, Spark Structured Streaming, Kinesis, change data capture (Debezium)
- [ ] **Orchestration:** **Airflow** (DAGs, operators, sensors), Dagster, Prefect, Mage
- [ ] **Transformation:** **dbt** (models, tests, lineage)
- [ ] **Data quality:** Great Expectations, Soda, Pandera, data contracts
- [ ] Data governance, catalogs (Unity Catalog, DataHub), lineage, PII handling
- [ ] **Feature stores:** **Feast**, Tecton, Hopsworks (online vs offline store, point-in-time correctness)
- [ ] **LLM data pipelines:**
  - Large-scale text processing: dedup, filtering, tokenization at scale (datatrove, NeMo Curator)
  - Synthetic data pipelines

---

## 22. MLOps & LLMOps

- [ ] **ML lifecycle:** problem → data → training → evaluation → deployment → monitoring → retraining
- [ ] **MLOps maturity levels:** manual → pipeline automation → CI/CD/CT (continuous training)
- [ ] **Experiment tracking:** **MLflow**, **Weights & Biases**, Neptune, Comet, TensorBoard
- [ ] **Versioning:**
  - Code (Git), data (**DVC**, lakeFS), models
  - **Model registry** (MLflow, W&B, HF Hub, SageMaker)
- [ ] **Reproducibility:** environment pinning, seeds, config management (**Hydra**, OmegaConf)
- [ ] **ML pipelines:** Kubeflow Pipelines, ZenML, Metaflow, Flyte, SageMaker Pipelines, Vertex AI Pipelines
- [ ] **Testing ML systems:** data tests, model tests (invariance, directional), behavioral testing, performance thresholds
- [ ] CI/CD for ML: automated training, evaluation gates, model promotion
- [ ] **Deployment strategies:** blue-green, **canary**, **shadow deployment**, A/B tests, rollback
- [ ] Model packaging: pickle/joblib, **ONNX**, TorchScript, `torch.export`, safetensors, **Hugging Face Hub**
- [ ] Batch vs online vs streaming inference
- [ ] **LLMOps specifics:**
  - Prompt versioning and management, eval-driven development
  - Tracing (spans for LLM calls, tools, retrieval), token and cost tracking per feature and user
  - Model routing and fallbacks, guardrails in the pipeline
  - Feedback collection, fine-tuning data flywheels
- [ ] **LLMOps tools:** **Langfuse**, **LangSmith**, Arize Phoenix, Helicone, Braintrust, W&B Weave, Opik, PromptLayer

---

## 23. Model Serving & Inference Optimization

### 23.1 General Model Serving
- [ ] **Serving patterns:** REST/gRPC endpoints, batch jobs, streaming, serverless
- [ ] **Frameworks:** FastAPI (custom), **BentoML**, **NVIDIA Triton Inference Server**, TorchServe, **Ray Serve**, KServe, Seldon Core
- [ ] **Runtimes:** **ONNX Runtime**, **TensorRT**, OpenVINO (Intel CPUs), Core ML, TFLite/LiteRT
- [ ] Dynamic batching, model warmup, concurrency, autoscaling
- [ ] Multi-model serving, GPU sharing (MIG, MPS, time slicing)

### 23.2 LLM Serving Engines
- [ ] **vLLM:** PagedAttention, continuous batching, OpenAI-compatible server, broad hardware support
- [ ] **SGLang:** RadixAttention (prefix caching), strong for multi-turn, RAG, and agents; structured outputs
- [ ] **TensorRT-LLM** (max NVIDIA performance), NVIDIA Dynamo, NIM
- [ ] LMDeploy, TGI (Hugging Face), MLC-LLM
- [ ] **Local serving:** **Ollama**, **llama.cpp** (GGUF), LM Studio, MLX (Apple Silicon), ExLlamaV2
- [ ] Choosing an engine: diverse prompts vs shared prefixes vs single-model max throughput

### 23.3 Inference Optimization Techniques
- [ ] **Metrics:** TTFT (time to first token), TPOT/ITL (per-token latency), throughput (tokens/s), goodput, p50/p95/p99 latency
- [ ] **Prefill vs decode phases:** compute-bound vs memory-bound
- [ ] **KV cache:** paging, quantization (FP8 KV), offloading, prefix/prompt caching, cache-aware routing
- [ ] **Continuous (in-flight) batching**, chunked prefill
- [ ] **Prefill-decode disaggregation**
- [ ] **Parallelism for inference:** tensor, pipeline, expert (MoE), data parallelism
- [ ] **Speculative decoding:** draft model, EAGLE, Medusa, n-gram lookup, DFlash
- [ ] **Quantization for serving:** FP8, INT4 (AWQ/GPTQ), FP4 (Blackwell)
- [ ] Kernel optimization: FlashAttention, FlashInfer, CUDA graphs, `torch.compile`
- [ ] **Multi-LoRA serving:** many adapters on one base model
- [ ] Structured output acceleration (XGrammar), long-context serving
- [ ] **Cost modeling:** $/1M tokens, GPU utilization, self-hosting vs API breakeven

### 23.4 Hosted Inference & GPU Clouds
- [ ] **Serverless GPUs:** **Modal**, RunPod, Replicate, Baseten, Beam
- [ ] **Inference providers:** Together, Fireworks, Groq, Cerebras, DeepInfra, OpenRouter
- [ ] **GPU clouds:** Lambda, CoreWeave, Nebius, Vast.ai, Paperspace
- [ ] **Hyperscaler managed services:** Bedrock, Vertex AI, Azure AI Foundry, SageMaker endpoints

---

## 24. DevOps: Docker, Kubernetes, CI/CD, IaC

### 24.1 Docker
- [ ] **Concepts:** containers vs VMs, images, layers, registries (Docker Hub, ECR, GCR, GHCR)
- [ ] **Dockerfile:** `FROM RUN COPY WORKDIR ENV EXPOSE CMD ENTRYPOINT`, `.dockerignore`
- [ ] **Multi-stage builds**, slim/distroless images, layer caching, image size optimization
- [ ] Volumes, bind mounts, networks, port mapping, environment variables, secrets
- [ ] **Docker Compose** for multi-service apps (API + Redis + Postgres + vector DB + worker)
- [ ] **GPU containers:** NVIDIA Container Toolkit, CUDA base images, `--gpus`
- [ ] Security: non-root users, image scanning (Trivy)

### 24.2 Kubernetes
- [ ] **Architecture:** control plane, nodes, kubelet, etcd, API server, scheduler
- [ ] **Core objects:** Pods, ReplicaSets, **Deployments**, **Services** (ClusterIP, NodePort, LoadBalancer), **Ingress**/Gateway API
- [ ] ConfigMaps, Secrets, Namespaces, labels/selectors
- [ ] StatefulSets, DaemonSets, Jobs, CronJobs
- [ ] **Storage:** PersistentVolumes, PVCs, StorageClasses
- [ ] **Scaling:**
  - **HPA** (Horizontal Pod Autoscaler), VPA, Cluster Autoscaler, Karpenter
  - **KEDA** (event and queue-based autoscaling)
- [ ] Resource requests/limits, liveness/readiness/startup probes, rolling updates, rollbacks
- [ ] **GPU on K8s:** NVIDIA GPU Operator, device plugins, node pools, taints/tolerations, MIG
- [ ] **Tools:** `kubectl`, **Helm** charts, Kustomize, k9s, minikube/kind
- [ ] **Managed K8s:** EKS, GKE, AKS
- [ ] **ML on K8s:** KServe, Kubeflow, Ray on K8s (KubeRay), LLM serving operators (vLLM production stack, llm-d)
- [ ] Service mesh (Istio, Linkerd) basics, **GitOps** (Argo CD, Flux)

### 24.3 CI/CD
- [ ] **GitHub Actions:** workflows, jobs, runners, secrets, matrix builds, caching
- [ ] GitLab CI, Jenkins, CircleCI (awareness)
- [ ] **Pipelines:** lint → test → build image → push → deploy, plus eval gates for ML/LLM changes
- [ ] Environments (dev, staging, prod), feature flags

### 24.4 Infrastructure as Code
- [ ] **Terraform/OpenTofu:** providers, resources, state, modules
- [ ] Pulumi, AWS CDK, CloudFormation (awareness)
- [ ] Configuration management: Ansible
- [ ] Secrets management: Vault, AWS Secrets Manager, SOPS

---

## 25. Cloud Platforms

### 25.1 Cloud Fundamentals
- [ ] IaaS vs PaaS vs SaaS vs serverless, regions and availability zones
- [ ] **Networking:** VPC, subnets, security groups, NAT, load balancers
- [ ] **IAM:** users, roles, policies, least privilege
- [ ] Cost management: reserved vs on-demand vs **spot** instances, budgets, tagging

### 25.2 AWS
- [ ] **Compute:** **EC2** (GPU instances: G/P families, Trainium/Inferentia), **Lambda**, ECS/Fargate, EKS
- [ ] **Storage:** **S3**, EBS, EFS
- [ ] **Databases:** RDS/Aurora (pgvector), DynamoDB, ElastiCache, OpenSearch
- [ ] **ML:** **SageMaker** (training, endpoints, pipelines), **Bedrock** (managed LLMs, knowledge bases, agents, guardrails)
- [ ] **Integration:** API Gateway, SQS, SNS, EventBridge, Step Functions
- [ ] **Operations:** CloudWatch, IAM, Secrets Manager, VPC

### 25.3 Google Cloud
- [ ] Compute Engine (GPUs/TPUs), **Cloud Run**, GKE, Cloud Functions
- [ ] Cloud Storage, BigQuery, Cloud SQL, Firestore, Pub/Sub, Dataflow
- [ ] **Vertex AI:** training, endpoints, Model Garden, Gemini API, vector search, Agent Engine
- [ ] TPUs and JAX ecosystem

### 25.4 Azure
- [ ] Azure VMs, AKS, Functions, Container Apps, Blob Storage, Cosmos DB
- [ ] **Azure AI Foundry / Azure OpenAI**, Azure ML, Azure AI Search

### 25.5 Others
- [ ] Vercel/Netlify (frontends), Cloudflare (Workers, Workers AI, R2, Vectorize)
- [ ] Fly.io, Railway, Render (easy deploys)
- [ ] Hugging Face Spaces and Inference Endpoints

---

## 26. GPU, CUDA & Hardware

- [ ] **Why GPUs:** parallelism, SIMT, throughput vs latency
- [ ] **GPU architecture:** SMs, CUDA cores, **Tensor Cores**, warps, memory hierarchy (registers, shared memory, L2, **HBM**)
- [ ] **Memory:** VRAM math for models (parameters × bytes + KV cache + activations), memory bandwidth as the decode bottleneck
- [ ] **NVIDIA generations:** Ampere (A100), Hopper (H100/H200), **Blackwell** (B200/GB200/GB300, FP4), NVL72 racks
- [ ] **Interconnects:** NVLink, NVSwitch, PCIe, InfiniBand, RoCE
- [ ] **Alternatives:** AMD (MI300X/MI350, ROCm), Google TPUs, AWS Trainium/Inferentia, Apple Silicon (MLX), Groq/Cerebras (inference ASICs)
- [ ] **CUDA programming basics:** kernels, threads, blocks, grids, memory coalescing, shared memory
- [ ] **Triton** (OpenAI's GPU kernel language), CUTLASS, writing fused kernels
- [ ] **Profiling:** `nvidia-smi`, Nsight Systems/Compute, PyTorch profiler
- [ ] **Roofline model:** compute-bound vs memory-bound, arithmetic intensity, MFU

---

## 27. Monitoring & Observability

- [ ] **Three pillars:** **logs, metrics, traces**
- [ ] **Logging:** structured JSON logs, log levels, correlation IDs, ELK/EFK stack, Loki
- [ ] **Metrics:** **Prometheus** (counters, gauges, histograms), **Grafana** dashboards, alerting (Alertmanager, PagerDuty)
- [ ] **Tracing:** **OpenTelemetry**, Jaeger, Tempo, distributed tracing
- [ ] **APM platforms:** Datadog, New Relic, Sentry (errors)
- [ ] **SRE concepts:** SLIs, SLOs, SLAs, error budgets, incident response, postmortems
- [ ] **ML monitoring:**
  - **Data drift** (PSI, KS test), **concept drift**, prediction drift, feature distribution monitoring
  - Performance decay, delayed labels, retraining triggers
  - Tools: Evidently, WhyLabs, Arize, Fiddler, NannyML
- [ ] **LLM monitoring:**
  - Latency (TTFT), tokens and cost, error rates, refusal rates
  - Hallucination/groundedness scores, user feedback, trace inspection
- [ ] **GPU monitoring:** utilization, memory, temperature (DCGM exporter)

---

## 28. Security, Safety & Responsible AI

### 28.1 Application & Infrastructure Security
- [ ] **OWASP Top 10:** injection, broken auth, XSS, CSRF, SSRF
- [ ] **OWASP Top 10 for LLM Apps:**
  - Prompt injection, insecure output handling, training data poisoning
  - Model DoS, supply chain risks, sensitive information disclosure
  - Excessive agency, overreliance
- [ ] Secrets management, HTTPS/TLS everywhere, network isolation, least privilege
- [ ] **Supply chain security:** pinned dependencies, pickle risks (use safetensors), model provenance
- [ ] **Agent security:** tool permissions, sandboxing, MCP tool poisoning, data exfiltration via tools, confirmation for destructive actions

### 28.2 AI Safety & Robustness
- [ ] Adversarial examples (FGSM, PGD), robustness, data poisoning, backdoors
- [ ] Jailbreaks, red teaming (manual and automated), model evaluations for dangerous capabilities
- [ ] **Privacy:** membership inference, differential privacy, PII in training data, machine unlearning
- [ ] Watermarking and AI content detection, deepfakes

### 28.3 Responsible AI
- [ ] **Fairness:** bias sources, metrics (demographic parity, equalized odds), mitigation
- [ ] Interpretability/explainability, model cards, datasheets for datasets
- [ ] Human oversight, accountability, transparency
- [ ] **Regulation:** **EU AI Act** (risk tiers, obligations), GDPR, US state and federal AI rules, ISO/IEC 42001, NIST AI RMF
- [ ] Copyright and licensing (model licenses: Apache, MIT, Llama-style community licenses, OpenRAIL)

---

## 29. System Design (General)

### 29.1 Fundamentals
- [ ] Requirements gathering: functional vs non-functional (latency, throughput, availability, consistency, cost)
- [ ] **Back-of-envelope estimation:** QPS, storage, bandwidth, memory, GPU count
- [ ] **Latency numbers** every engineer should know
- [ ] **CAP theorem**, PACELC, consistency models (strong, eventual, causal)

### 29.2 Building Blocks
- [ ] **Load balancers:** L4 vs L7, round robin, least connections, consistent hashing
- [ ] **Caching:** CDN, Redis/Memcached, cache strategies, invalidation
- [ ] **Databases:**
  - SQL vs NoSQL, **replication** (leader-follower, multi-leader), read replicas
  - **Sharding/partitioning** (hash, range), rebalancing
  - Indexes, denormalization
- [ ] **Message queues and event streaming:** Kafka, RabbitMQ, SQS; async processing, backpressure
- [ ] Object storage and blob stores, search indexes
- [ ] **API gateways**, reverse proxies, **rate limiting** (token bucket, leaky bucket, sliding window)
- [ ] Service discovery, configuration management

### 29.3 Architecture Styles
- [ ] Monolith vs **microservices** vs modular monolith, serverless
- [ ] **Event-driven architecture**, CQRS, event sourcing, sagas
- [ ] Idempotency, retries with backoff, **circuit breakers**, bulkheads, timeouts
- [ ] Distributed transactions (2PC), consensus (Raft, Paxos basics), leader election
- [ ] Unique ID generation (Snowflake IDs, UUIDv7)

### 29.4 Reliability & Availability
- [ ] Redundancy, failover, multi-AZ and multi-region, disaster recovery (RPO/RTO)
- [ ] Health checks, graceful degradation, chaos engineering

### 29.5 Classic Design Problems
- [ ] URL shortener, rate limiter, key-value store, chat system (WhatsApp)
- [ ] News feed, notification system, web crawler, search autocomplete
- [ ] YouTube/Netflix, Uber, payment system, distributed cache

---

## 30. ML System Design

- [ ] **Framework:**
  1. Clarify the problem and business metric
  2. Frame it as an ML task
  3. Data (sources, labels, features)
  4. Model (baseline → complex)
  5. Offline evaluation
  6. Serving (batch or real-time)
  7. Online evaluation (A/B tests)
  8. Monitoring and iteration
- [ ] **Data:**
  - Collection, labeling strategy, sampling, class imbalance
  - Train/serve skew, feature stores, point-in-time correctness
- [ ] **Serving:**
  - Online vs batch vs streaming features and predictions
  - Latency budgets, model compression, caching predictions
- [ ] **Architecture patterns:** two-stage (candidate generation → ranking), cascades, ensembles, embeddings plus ANN retrieval
- [ ] **Feedback loops:** implicit/explicit feedback, position bias, exploration vs exploitation
- [ ] **Experimentation:** A/B testing, interleaving, multi-armed bandits, guardrail metrics
- [ ] **Deployment:** shadow mode, canary, rollback, continual learning, retraining cadence
- [ ] Privacy, fairness, and interpretability requirements
- [ ] **Case studies:**
  - Recommendation system (YouTube, Netflix, TikTok feed), search ranking, ad click-through prediction
  - Fraud detection, spam/content moderation, ETA prediction
  - People You May Know, visual search, autocomplete, dynamic pricing, churn prediction

---

## 31. LLM / GenAI System Design

- [ ] **Framework:**
  1. Use case and success metrics
  2. Model selection (API vs open, size, latency, cost)
  3. Prompt/context strategy
  4. Retrieval
  5. Tools and agents
  6. Evals
  7. Guardrails
  8. Serving and scaling
  9. Observability and feedback loop
- [ ] **Model selection and routing:**
  - Small vs large models, cascade routing (cheap model first), fallback providers
  - **System One models (Jev/Laya) as a router or gate** in front of LLMs
- [ ] **RAG system design at scale:**
  - Ingestion pipelines, incremental indexing, embedding model upgrades, multi-tenant indexes
  - Access control on documents (permission-aware retrieval), freshness
- [ ] **Agent system design:**
  - Orchestration, state management, durable execution
  - Tool registries (MCP), agent-to-agent (A2A), sandboxing, human approval steps
  - Long-running tasks, cost and step limits
- [ ] **Caching:** exact-match cache, **semantic cache**, provider **prompt caching**, KV/prefix caching on self-hosted models
- [ ] **Cost optimization:**
  - Token budgets, prompt compression, batching, batch APIs
  - Distillation into smaller models, quantized self-hosting
- [ ] **Latency optimization:** streaming, speculative decoding, parallel tool calls, smaller models for sub-steps, edge inference
- [ ] **Reliability:** rate limits, retries, timeouts, provider failover, output validation, circuit breakers
- [ ] **Self-hosted LLM platform:**
  - GPU capacity planning (VRAM math, throughput per GPU)
  - Autoscaling on queue depth, multi-LoRA serving, prefill/decode disaggregation, cache-aware load balancing
- [ ] **Safety layer:** input/output moderation, PII redaction, jailbreak detection, audit logging
- [ ] **Evaluation infrastructure:** offline eval suites, online sampling plus LLM-as-judge, feedback-to-dataset flywheel
- [ ] **Case studies:**
  - ChatGPT-like chat service, enterprise knowledge assistant (RAG over millions of docs), customer-support agent
  - Coding assistant, AI search engine (Perplexity-like), voice agent, document-processing pipeline
  - Text-to-image service, LLM gateway for a company, high-volume triage/moderation with decision models

---

## 32. Scaling (Training & Serving)

### 32.1 Distributed Training
- [ ] **Data parallelism:** DDP, gradient all-reduce, ring all-reduce
- [ ] **Sharded data parallelism:** **ZeRO** stages 1/2/3 (DeepSpeed), **FSDP/FSDP2**
- [ ] **Model parallelism:**
  - **Tensor parallelism** (Megatron-LM)
  - **Pipeline parallelism** (GPipe, 1F1B, bubbles)
  - **Sequence/context parallelism** (ring attention), **expert parallelism** (MoE)
- [ ] **3D/4D/5D parallelism** combinations
- [ ] Collective ops: all-reduce, all-gather, reduce-scatter; NCCL
- [ ] **Frameworks:** PyTorch Distributed, **DeepSpeed**, **Megatron-LM/Megatron-Core**, NeMo, TorchTitan, HF Accelerate, Ray Train, JAX (pjit/shard_map)
- [ ] Checkpointing at scale, fault tolerance, elastic training, cluster schedulers (**Slurm**, K8s + Kueue/Volcano)
- [ ] Mixed precision at scale (BF16/FP8 training), activation checkpointing, CPU/NVMe offloading

### 32.2 Distributed Compute
- [ ] **Ray:** tasks, actors, Ray Data, Ray Train, Ray Tune, Ray Serve
- [ ] Spark for data preprocessing, Dask

### 32.3 Scaling Serving
- [ ] Horizontal scaling of stateless APIs, autoscaling on GPU utilization, queue depth, or tokens/s
- [ ] Cold starts (model loading time, weight streaming, snapshotting)
- [ ] Multi-region deployment, global load balancing, edge caching
- [ ] Capacity planning, load testing, cost per request
- [ ] Rate limiting and quotas per tenant, fair scheduling, priority queues

---

## 33. Edge & On-Device AI

- [ ] **Why:** latency, privacy, offline use, cost
- [ ] **Runtimes:** TFLite/LiteRT, Core ML, ONNX Runtime Mobile, ExecuTorch, MediaPipe, **llama.cpp**, **MLX**, WebGPU/WebLLM, Transformers.js
- [ ] Model optimization for edge: quantization (INT8/INT4), pruning, distillation, small models (SLMs, MobileNet, tiny YOLO)
- [ ] **Hardware:** NVIDIA Jetson, Raspberry Pi, Coral TPU, phone NPUs, Apple Neural Engine
- [ ] **On-device LLMs and decision models:** Laya on ONNX/MLX/CoreML, small Gemma/Phi/Qwen variants
- [ ] TinyML (microcontrollers), federated learning on devices

---

## 34. Research, Projects & Career

- [ ] **Reading papers:** arXiv, Papers with Code alternatives, Hugging Face Papers, alphaXiv; the three-pass reading method
- [ ] **Implementing papers from scratch:** micrograd, nanoGPT, a Transformer, a diffusion model, LoRA, DPO
- [ ] **Key papers to know:**
  - AlexNet, ResNet, "Attention Is All You Need", BERT, GPT-2/3, InstructGPT
  - CLIP, DDPM, Latent Diffusion, LoRA, Chinchilla, FlashAttention, DPO
  - DeepSeek-R1, On-Policy Distillation
- [ ] **Staying current:** lab blogs, technical reports of open models, newsletters, AI Twitter/Bluesky, conferences (NeurIPS, ICML, ICLR, CVPR, ACL)
- [ ] **Courses and books:**
  - Andrew Ng ML/DL, fast.ai, CS231n (vision), CS224n (NLP)
  - **CS336** (LLMs from scratch), Karpathy's "Zero to Hero"
  - Hugging Face courses, Chip Huyen's *AI Engineering* and *Designing ML Systems*
- [ ] **Competitions:** Kaggle (tabular, CV, NLP), hackathons
- [ ] **Open source:** contribute to HF, vLLM, LangChain, and similar projects; publish models and datasets on the HF Hub
- [ ] **Portfolio:** end-to-end deployed projects with READMEs, demos, eval results, and blog write-ups
- [ ] **Interview prep:**
  - DSA (LeetCode), ML theory questions, coding ML from scratch
  - ML system design, LLM system design, behavioral questions
- [ ] **Roles:** Data Scientist, ML Engineer, AI Engineer, Research Engineer, Research Scientist, MLOps/Platform Engineer, Applied Scientist

---

## 35. Suggested Learning Order & Projects

| Phase | Sections | Milestone Project |
|---|---|---|
| 1. Foundations | 0, 1, 2, 3 | CLI tool + SQL analytics project |
| 2. Math + Data | 4, 5 | EDA report on a real dataset |
| 3. Classical ML | 6 | Tabular model (XGBoost) served via **FastAPI + Docker** |
| 4. Deep Learning | 7 | Train an MLP/CNN in PyTorch from scratch; build micrograd |
| 5. Computer Vision | 8 | Object detection app (YOLO) with a Gradio demo |
| 6. NLP + LLMs | 9, 10 | Build nanoGPT; fine-tune a small model with LoRA/QLoRA |
| 7. Post-training + Decision models | 11, 12 | DPO/GRPO on a small model; a Laya-based triage router |
| 8. GenAI | 13, 14 | Diffusion LoRA; a voice assistant (Whisper + LLM + TTS) |
| 9. LLM Apps | 17, 19 | **Production RAG:** FastAPI + Redis cache + Qdrant/pgvector + reranker + evals |
| 10. Agents | 17.5 | Multi-tool agent with MCP servers, LangGraph, human-in-the-loop |
| 11. Backend + Frontend | 18, 20 | Full-stack AI app: Next.js + FastAPI + Postgres + Redis + Celery + auth |
| 12. Data Eng + MLOps | 21, 22, 27 | Airflow pipeline → MLflow → CI/CD → drift monitoring |
| 13. Serving + DevOps + Cloud | 23, 24, 25, 26 | Self-host an LLM with vLLM/SGLang on K8s with autoscaling on AWS/GCP |
| 14. System Design + Scaling | 29–32 | Write design docs for 5 ML and 5 LLM systems; a distributed training run with FSDP |
| 15. Safety + Edge + Career | 28, 33, 34 | Red-team your own app; on-device model; portfolio + blog |

**Capstone:** an AI product with a decision-model router (Laya/Jev), RAG + agents on LLMs, FastAPI backend, Redis semantic cache, vector DB, Celery workers, Next.js streaming UI, Docker/K8s deployment, OpenTelemetry + Langfuse monitoring, an eval suite in CI, and autoscaling.

---

*Last updated: September 23, 2026. The AI field moves weekly, so revisit sections 10–12, 17 and 23 regularly.*
