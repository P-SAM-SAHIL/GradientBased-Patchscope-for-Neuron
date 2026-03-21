# Beyond Auto-Interpretability: Mapping a Hierarchical Math Circuit in Gemma-2-2B via Gradient-Based Patchscopes

**Authors:**  
P Sam Sahil, Joseph Miller  

**Affiliations:**  
Independent Researcher  
University of Oxford  

**Journal:** Mechanistic Interpretability & AI Safety Research  
**Date:** 2026-03-17  
**DOI:** 0058vfby  
**License:** CC-BY-4.0  
**URL:** https://mechanistic-interpretability-lab.pubpub.org/pub/0058vfby  

---

## Abstract

This report presents a gradient based Patchscope method for interpreting
a computational circuit in Gemma 2 2B. The study targets Layer 12,
Neuron 8526, a node that consistently behaves as an abstract mathematics
and definitional neuron. The method adapts the Patchscopes
framework[@ghandeharioun2024patchscopesunifyingframeworkinspecting] by
replacing the usual source prompt representation with a continuous
residual vector obtained through gradient ascent. This vector is
optimized to maximize the activation of the target neuron, decomposed
with a Gemma Scope sparse
autoencoder[@lieberum2024gemmascopeopensparse], and then decoded into
natural language with contrastive Patchscope generation. The pipeline
was run from two source layers, Layer 8 and Layer 11, while keeping the
same downstream target. The results show a consistent cross layer
circuit. Earlier residual features provide mathematical syntax like
structure, later residual features organize that structure into concrete
mathematical concepts, and the downstream neuron drives explicit
definitional text. This approach provides direct causal evidence that
goes beyond observational feature labels derived from datasets alone.

Repository: P-SAM-SAHIL/GradientBased-Patchscope-for-Neuron

## Background and Motivation

Mechanistic interpretability seeks to explain neural networks in terms
of simpler internal computations. Sparse autoencoders are central to
this effort because they decompose dense activations into sparse and
more interpretable features. Resources such as Gemma Scope and
Neuronpedia make these features easier to inspect and label. But most
existing workflows remain observational. They describe what features co
occur with in data. They do not directly test what those features do
inside the model.

This limitation becomes clear when researchers try to find inputs that
activate a specific neuron. Earlier methods often search in discrete
token space. That process is slow, unstable, and prone to generating
unnatural text. It also constrains the search to token sequences that
may not cleanly reflect the downstream computation. The present work
avoids that problem by optimizing directly in residual space.

This study also addresses a second problem. Dataset based feature labels
often confuse context with function. A feature that appears near
software licenses can be labeled as a copyright feature even if its real
role in computation is mathematical syntax or formatting. Causal
intervention is needed to resolve that ambiguity. (as The Label
generated for feature on Neuropedia is By LLM)

Patchscopes offers a useful foundation for this intervention. In the
original framework, a hidden representation from a source prompt is
patched into a target prompt, and the model's output reveals what the
representation encodes in natural language. This study keeps that
source, target, patch, and reveal logic, but changes the source object.
Instead of taking a representation from a natural prompt, it constructs
a continuous residual vector through gradient optimization and then
decodes that vector with a Patchscope style target prompt.

## Methodology

The implementation follows the same code structure in both experiments.
Gemma 2 2B is loaded through TransformerLens across two GPUs in
\`bfloat16\`. Because the model is distributed across devices, the
unembedding weights are cloned onto the final device so that tied
embeddings remain consistent during generation.

The target of the intervention is fixed throughout the study. The code
sets \`target_layer = 12\` and \`target_neuron_idx = 8526\`. The source
layer is varied across runs, first with \`early_layer = 8\` and then
with \`early_layer = 11\`.

The first phase computes a baseline residual state. The code loads the
first 100 samples from \`NeelNanda/pile-10k\`, converts them to strings,
tokenizes them with left padding, and keeps the final 32 tokens of each
sequence. It then extracts \`blocks.{early_layer}.hook_resid_pre\` at
the last token position and averages these activations across the batch.
This average becomes the Mean Vector. It represents the typical
background state of the source layer before targeted activation.

The second phase performs gradient based search in residual space. A
learnable vector \`X\` is initialized from the Mean Vector and optimized
with Adam at learning rate \`0.05\` for \`50\` steps. The code injects
\`X\` into \`blocks.{early_layer}.hook_resid_pre\` at the position of
\`x\` in the dummy prompt \`\"\<pad\> \<pad\> \<pad\> x\"\`. The loss is
defined as the negative activation of \`blocks.12.mlp.hook_post\` at
neuron \`8526\`. This makes the optimization directly maximize the
target neuron while keeping the model weights frozen. The final
optimized tensor is stored as the Optimized Vector.

The third phase verifies the optimized signal with a sparse autoencoder.
The code loads the Gemma Scope SAE for the matching source layer with
\`sae_id = f\"layer\_{early_layer}/width_16k/canonical\"\`. The
Optimized Vector is encoded through the SAE, and the top activating
features are extracted. This step identifies the sparse concepts that
dominate the dense residual vector.

The fourth phase applies Patchscope style decoding. The target prompt is
fixed as \"A detailed description of $X$ is that\". The model runs two
contrastive branches with separate KV caches. In the signal branch, the
Optimized Vector is patched into the source layer at the token position
aligned with $X$ . In the noise branch, the Mean Vector is patched into
the same position. The final decoding logits are computed as

$FinalLogits = L_{act} + alpha(L_{act} - L_{mean})$

with $alpha = 1.5$. Generation then proceeds autoregressively for \`20\`
tokens. This stage turns the optimized residual signal into natural
language and serves as the reveal step of the Patchscope.

## Results

The pipeline was run twice against the same downstream target. In both
runs, Layer 12, Neuron 8526 behaved as a definitional mathematics
neuron. When activated through optimized upstream residual vectors, it
pushed the model toward formal mathematical description.

In the first experiment, the source layer was Layer 8. Optimization
started at an activation of \`4.8125\` and rose steadily across 50
steps. By step 40, the activation had reached \`50.0000\`, and during
Patchscope prefill the target neuron reached \`56.0000\`. SAE
decomposition showed that the strongest source layer component was
\`Feature_302\`, with activation \`41.6082\`. The decoded text was: "it
is a number that is not a real number. So, it is a number that is not".
Although the generation was truncated at 20 tokens, its meaning is
clear. The optimized Layer 8 signal drives the model toward the
definition of an imaginary or non real number. This indicates that the
earlier part of the circuit contains mathematical syntax and
representational cues that the downstream neuron can use for abstract
definitional output.

In the second experiment, the source layer was Layer 11. Optimization
began at \`5.8750\` and rose to \`57.0000\` by step 40. During
Patchscope prefill, the target neuron reached \`62.7500\`, which
exceeded the Layer 8 result. SAE analysis identified \`Feature_10839\`
as the dominant component, with activation \`15.9724\`. The decoded
output was: "it is a set of all the points in the plane that are
equidistant from a fixed point and". This is the beginning of a standard
geometric definition of a circle. Again, the output was truncated by the
20 token limit, but the semantic direction is unambiguous. At Layer 11,
the circuit has moved beyond lower level structure and is already
representing a concrete mathematical relation that Layer 12 turns into
formal explanation.

Taken together, the two experiments reveal a hierarchical cross layer
pathway. The Layer 8 intervention suggests that early residual features
provide syntax like or symbolic structure that is useful for
mathematical reasoning. The Layer 11 intervention shows a later stage in
which that structure has already been organized into a concrete
geometric concept. Layer 12, Neuron 8526, appears to sit at the
definitional stage of this circuit, where upstream mathematical
structure is converted into explicit explanatory text.

## Interpretation

These results show why observational labels are often not enough. A
feature can be associated with one theme in a dataset and still perform
a different role in computation. In this study, the optimized vectors
did not produce outputs about document structure, software licenses, or
generic formatting. Which is shown by top SAE Features .They produced
definitions. That is the causal behavior that matters.

The results also clarify how Patchscopes is being used here. The method
does not simply inspect a naturally occurring hidden state. It first
constructs a residual vector that is maximally useful to a chosen
downstream neuron. It then uses Patchscope decoding to reveal what that
vector functionally represents. This makes the procedure both
interventive and explanatory. It is not only asking what information is
present. It is testing what information is sufficient to drive a
specific neuron.

## Conclusion

This study introduces a gradient based Patchscope pipeline for causal
circuit analysis in Gemma 2 2B. The approach optimizes a continuous
residual vector, verifies it with Gemma Scope SAEs, and decodes it
through a Patchscope style contrastive target prompt. When applied to
Layer 12, Neuron 8526, the method reveals a stable abstract mathematics
and definitional circuit. The Layer 8 run points to lower level
mathematical structure. The Layer 11 run reveals a more developed
geometric concept. Both converge on the same downstream function. The
main contribution is methodological and empirical at once. It shows that
continuous residual optimization can be combined with Patchscopes to
move from observational labeling to direct causal explanation.

## Future Work

The next step is stronger validation. The optimization should be
repeated from multiple random initializations to test whether the same
sparse features appear consistently. Feature ablation is also necessary.
If suppressing the dominant source layer feature reduces the activation
of Layer 12, Neuron 8526, that would establish necessity rather than
association.

A second step is forward pass evaluation under controlled prompts. The
relevant source layer features should be tested on pure mathematical
formatting, pure licensing text, and mixed contexts. This will show what
input conditions actually engage the circuit during normal inference.

A third step is direct intervention during unrelated generation. If
forced activation of the source feature or the target neuron shifts the
model toward mathematical definitions in otherwise neutral contexts, the
causal role of the circuit will be even clearer.

Finally, the same pipeline should be scaled across additional layers and
target neurons. That would allow the broader abstract mathematics graph
in Gemma 2 2B to be mapped more systematically and would help correct
misleading feature labels in existing interpretability datasets.



