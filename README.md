# GradientBased-Patchscope-for-Neuron




## Architecture & Methodology

Our interpretability pipeline is designed to isolate the exact representation a downstream neuron is "looking for" from upstream layers, translate that representation into interpretable features, and validate the semantic meaning behaviorally. The methodology consists of five primary phases:

### 1. Model & Environment Setup

* **Model:** `gemma-2-2b` loaded via `HookedTransformer` (TransformerLens).
* **Hardware Setup:** Distributed across 2 GPUs using `bfloat16` precision to manage memory while retaining activation fidelity.
* **Tied Embeddings:** Unembedding weights are explicitly cloned and mapped to the final device to support distributed hooking.

### 2. Computing the Generic Mean Vector (The Baseline)

To measure the true signal of an activated concept, we first establish a baseline of "generic" text processing.

* **Dataset:** `NeelNanda/pile-10k` (first 100 samples).
* **Process:** The text is tokenized and truncated. We run a forward pass and extract the pre-residual activations (`hook_resid_pre`) at the target early layer (e.g., Layer 8 or Layer 11) for the final sequence tokens.
* **Output:** A collapsed mean tensor (`Mean_Vector` of shape `[2304]`) representing the average activation state of that layer.

### 3. Gradient-Based Search for the Optimized Vector

Rather than relying on dataset search to find what activates our target neuron (Layer 12, Neuron 8526), we engineer the optimal input vector via gradient ascent.

* **Setup:** We freeze the model weights. The `Mean_Vector` is cloned into a learnable parameter `X` with `requires_grad=True`.
* **Forward Hook:** We inject `X` into an early layer (Layer 8 or 11) at a specific token position in a dummy prompt (`<pad> <pad> <pad> x`).
* **Objective Function:** The loss is defined as the negative activation of Layer 12, Neuron 8526 at the MLP post-hook (`blocks.12.mlp.hook_post`).
* **Optimization:** Using the Adam optimizer (lr=0.05) for 50 steps, we update `X` to maximize the target neuron's firing rate. The final frozen `X` becomes our `Optimized_Vector`.

### 4. SAE Verification (Gemma Scope)

To understand the dense `Optimized_Vector`, we decompose it into human-interpretable, sparse features.

* **Process:** The `Optimized_Vector` is passed through the canonical Gemma Scope SAE corresponding to the injection layer (e.g., `layer_11/width_16k/canonical` or `layer_8/width_16k/canonical`).
* **Output:** We extract the top-k highest activating SAE features to identify the semantic concepts comprising the `Optimized_Vector`.

### 5. Patchscopes & Contrastive Decoding

We validate the semantic meaning of the `Optimized_Vector` by forcing the model to generate text based on it.

* **Prompt:** `"A detailed description of x is that"`
* **Injection:** During the prefill pass, the `Optimized_Vector` is injected at the position of 'x' to create Run A (Signal). Simultaneously, the `Mean_Vector` is injected to create Run B (Noise).
* **Contrastive Decoding:** We calculate the next token logits using a weighted contrastive formula to amplify the signal of the optimized vector:

$$Final\_Logits = L_{act} + \alpha(L_{act} - L_{mean})$$



*(where $\alpha = 1.5$)*
* **Autoregressive Generation:** The model generates tokens based on these manipulated logits, outputting a natural language definition of the concept encoded in the vector.

## Results: A Cross-Layer Math Circuit

By running this pipeline from two different early layers (Layer 11 and Layer 8) and targeting the same Layer 12 neuron, we observed the hierarchical, cross-layer construction of a mathematical concept.

### The Target Node: Layer 12, Neuron 8526

This neuron functions as an **Abstract Mathematics & Definitional Neuron**. When forcibly activated via upstream optimization, it consistently drives the model to output high-level geometric theories and abstract number definitions.

### Late-Stage Assembly (Source: Layer 11)

When optimizing a vector at Layer 11 to trigger the target neuron, the SAE decomposition revealed:

* **Dominant Feature:** `Layer 11, Feature 10839` (Activation: 15.97)
* **Mechanistic Role:** Detects mathematical equations, algebraic variables, and calculation syntax (e.g., `3 + 37 for k`, `z**(1720/63)`).
* **Patchscope Output:** *"it is a set of all the points in the plane that are equidistant from a fixed point"*
* **Analysis:** At Layer 11, the model has assembled low-level syntax into concrete mathematical expressions. The Patchscope output reveals this triggers the definition of a parabola or circle in Layer 12.

### Early-Stage Assembly (Source: Layer 8)

Pushing the source layer back to Layer 8 reveals the foundational building blocks of the circuit:

* **Dominant Feature:** `Layer 8, Feature 302` (Activation: 41.60)
* **Mechanistic Role:** Detects formatting, specifically LaTeX math syntax (`\mkern`, `\frac`, `\smash`) and `MathML` elements.
* **Patchscope Output:** *"it is a number that is not a real number. So, it is a number that is not "*
* **Analysis:** At Layer 8, the representation is strictly structural/syntactical. The Patchscope output proves this syntax representation successfully cascades forward to trigger the definition of an imaginary/complex number in Layer 12.

## The Neuronpedia Discrepancy: AI Labels vs. Mechanistic Truth

A critical finding of this experiment is the divergence between AI-generated auto-interpretability labels (hosted on Neuronpedia) and the actual causal function of the features within the model's computational graph.

### The Case of Layer 8, Feature 302

* **The AI Label:** Auto-interpretability (via Claude-4/GPT-4) labeled this feature as *"copywrite and licensing information."*
* **The Rationale for the Label:** The highest magnitude activations in the dataset for Feature 302 occur on boilerplate text like Apache and MIT software licenses. The LLM annotator generalized the label based on these top-activating examples.
* **The Mechanistic Reality (Polysemanticity):** Our gradient-based search proves Feature 302 is polysemantic. Alongside software licenses, it fires heavily on LaTeX mathematical syntax and spacing elements.
* **Selective Downstream Filtering:** When Layer 12, Neuron 8526 processes information from Layer 8, it completely ignores the copyright semantics of Feature 302. The gradient optimization proved that the target neuron is strictly sensitive to the LaTeX/math-syntax properties of the feature.

