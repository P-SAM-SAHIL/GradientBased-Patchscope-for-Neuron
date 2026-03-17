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

## **Abstract**

This report shares a new way to understand Large Language Models. We combine gradient vector optimization with Patchscopes.[@ghandeharioun2024patchscopesunifyingframeworkinspecting] This isolates and validates computational circuits across layers.

We applied this method to Gemma 2 2B. We targeted a specific node for abstract math and definitions at Layer 12 Neuron 8526. We ignore search methods based on datasets because token space limits them.

Instead we use gradient ascent in the early residual stream. This builds a vector that drives the target neuron perfectly. Then we use Gemma Scope Sparse Autoencoders. These break the vectors down into features humans understand. Finally we use contrastive decoding to generate plain text explanations of the concepts.

**Repository:**  
[P-SAM-SAHIL/GradientBased-Patchscope-for-Neuron](https://github.com/P-SAM-SAHIL/GradientBased-Patchscope-for-Neuron)

---

## **Background & The Interpretability Gap**

Mechanistic interpretability tries to reverse engineer neural networks into simple algorithms. A main tool here is the Sparse Autoencoder. It breaks dense activations into single clear features.

Tools like Gemma Scope and Neuronpedia[@https://www.neuronpedia.org/gemma-scope#main] make these features easy to access by using AI models to generate labels automatically. But this relies purely on observation. It creates a massive gap in how we understand causal network behavior.

### **1. The Limits of Discrete Feature Visualization**

Finding the exact input that triggers a neuron is difficult. Old methods try to optimize prompts in discrete token space. This is highly inefficient and takes thousands of steps.

It creates broken text that models never actually use. We need an efficient and continuous way to build inputs that trigger a downstream neuron and still decode into readable text.

### **2. The Trap of Observational Labels**

Right now people label features based on how often they appear in a dataset. An AI reads text snippets that trigger a feature and guesses the theme. This falls into a distribution trap.

A feature often just controls formatting like LaTeX spacing. But if it shows up next to software licenses in the training data the AI labels it as a software license feature. It misses the true structural purpose.

---

## **Our Approach: Closing the Gap**

We move away from discrete token search and observational labels. We treat the early residual stream as a continuous proxy for the input sentence.

We use continuous gradient optimization directly on an early vector. This lets us build the exact math needed to fire a downstream node. We pass this optimized vector through a Patchscope to force the model to decode the math into natural text.

This gives us a true causal explanation of what the feature does. It completely bypasses the bias of static datasets.

---

## **Methodology: The Gradient-Based Patchscope Pipeline**

Our pipeline isolates exactly what a downstream neuron looks for. It translates that dense math into clear sparse features and proves its meaning.

We do this in four phases.

### **1. Setting the Baseline**

We need to measure the true signal of an active concept.

First we find a baseline for normal text processing. We load Gemma 2 2B across two GPUs and run 100 samples from a standard dataset. We tokenize the text and extract the early layer activations. We collapse these into a mean tensor.

This creates the **Mean Vector**. It represents the average unactivated state of that layer.

---

### **2. Gradient Ascent in Residual Space**

Old methods struggle with discrete optimization. We optimize in the continuous residual stream instead.

We clone the Mean Vector into a learnable parameter **X**. We freeze the model weights. We inject **X** into an early layer using a dummy prompt.

We define the loss as the negative activation of our target node. We use the Adam optimizer over 50 steps to update **X**. This drives the target neuron to its maximum firing rate.

This frozen tensor becomes our **Optimized Vector**.

---

### **3. SAE Verification**

We need to understand what makes up the dense Optimized Vector.

We break it down using Sparse Autoencoders. We pass the vector through the Gemma Scope SAE for that specific layer. We extract the highest activating features.

This identifies the readable concepts inside the vector.

---

### **4. Contrastive Patchscopes**

Finally we prove the semantic meaning of the Optimized Vector.

We force the model to decode it into plain text. We use a simple prompt asking for a detailed description.

We inject the Optimized Vector to create a **Signal Run**. We inject the Mean Vector to create a **Noise Run**.

We amplify the signal and suppress normal text generation using a weighted contrastive formula:

\[
FinalLogits = L_{act} + \alpha (L_{act} - L_{mean})
\]

We set **α = 1.5**. The model generates text based on this math. It outputs a clear definition of the concept.

---

## **Results: A Hierarchical Cross-Layer Math Circuit**

By running our gradient-based Patchscope pipeline from two distinct upstream layers (Layer 8 and Layer 11) and targeting the same downstream node (Layer 12, Neuron 8526), we observed the hierarchical construction of a mathematical concept.

The target node, **Layer 12, Neuron 8526**, consistently functioned as an *Abstract Mathematics & Definitional Neuron*.

When forcibly activated via upstream continuous optimization, it drove the model to autoregressively generate high-level geometric theories and abstract number definitions.

---

## **Results and the Math Circuit**

We ran our pipeline from Layer 8 and Layer 11. We targeted the same downstream node at Layer 12 Neuron 8526.

We saw the model build a math concept step by step. The target node always works as an abstract math neuron. We forced it to activate. It made the model generate geometric theories and number definitions.

---

### **1. Experiment 1 at Layer 8**

We wanted to find the basic building blocks of this circuit.

We injected our vector at Layer 8 and optimized it. After 50 steps the vector drove the target neuron to a high activation of 56.

We passed this vector through the SAE. The main component is **Feature 302**. We put this into the contrastive decoding context.

The model output said it is a number that is not a real number.

This exposes a huge flaw.

Neuronpedia labels Feature 302 as copyright and licensing info because the feature reacts to software licenses in training data. But the feature actually reacts to math syntax too.

Our intervention proves the downstream neuron ignores the copyright meaning entirely. It only cares about the math syntax. It uses the syntax as a trigger to define complex numbers.

---

### **2. Experiment 2 at Layer 11**

We moved the injection point closer to the target and repeated the process at Layer 11.

The target neuron reached an even higher activation of 62.75. The SAE decomposition showed **Feature 10839** as the main component.

Decoding this vector gave a precise geometric definition. The model described a set of points in a plane equidistant from a fixed point.

Observational labels fail again here.

The AI annotator labels Feature 10839 as the beginning of a document. But the feature actually fires on algebra and equations.

The output confirms Layer 11 turns low level syntax into concrete math concepts. This triggers the geometric definition of a circle in Layer 12.

---

## **Future Work**

This initial study is just the start. We need to turn these findings into a proven framework.

### **1. Validation and Ablation**

We need to prove our vectors are real.

We run gradient ascent using random starting points. This ensures the optimization finds the same features every time.

We also suppress Feature 302 in Layer 8 during a forward pass. If the downstream activation collapses we prove the feature is necessary.

---

### **2. Forward Pass Testing**

We know Feature 302 has multiple meanings.

We test its predictive power in normal operations. We write custom prompts to isolate these concepts.

We feed the model pure math formatting and then pure software licenses. This measures the exact threshold needed to trigger the definitional neuron.

---

### **3. Direct Interventions**

The best way to prove a circuit is to intervene during normal text generation.

We force activate Feature 302 and Neuron 8526 during unrelated prompts. The model pivots to generating math definitions.

This proves the circuit controls the output.

---

### **4. Scaling the Pipeline**

We isolated one pathway so far.

In the next phase we map the entire abstract math graph in Gemma 2 2B. Automating the search across all layers shows how early syntax features work together to form complex concepts.

---

### **5. Correcting Datasets**

This method fixes AI generated labels.

We integrate continuous causal verification into the pipeline. We automatically flag and relabel features that datasets get wrong.

This moves the field from guessing to truth.
