# RHO-1 Not All Tokens Are What You Need

**Paper**: RHO-1: Not All Tokens Are What You Need  
**Authors**: Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, et al. (Microsoft Research)  
**ArXiv**: https://arxiv.org/abs/2404.07965v4  
**Presenter**:  Camden Bibro
**Date**: November 3, 2025

---

## Overview

### Context
Traditional language models apply uniform loss to every token during pre-training - essentially treating all tokens as equally important for learning. This approach, while straightforward, may not be optimal given the varying quality and relevance of tokens in real-world training data.

Consider this: when you read a textbook, do you memorize every word with equal effort? Or do you naturally focus on key concepts while skimming over less important details? RHO-1 brings this intuition to language model training.

### Problem Statement
Despite extensive filtering at the document level, high-quality datasets still contain significant **token-level noise**:
- **51% of tokens** show minimal learning (already known - L→L tokens)
- **11% of tokens** remain persistently difficult (high noise - H→H tokens)  
- Only **26% of tokens** show meaningful learning improvement (H→L tokens)
- **12% of tokens** paradoxically become harder to learn over time (L→H tokens)

This raises a critical question: *Can we train language models more efficiently by selectively focusing on the most valuable tokens?*

### Approach (Selective Language Modeling - SLM)

RHO-1 introduces **Selective Language Modeling (SLM)** - a novel pre-training approach that:

1. Trains a reference model on high-quality data (from scratch)

	- **0.5 B math tokens** from the curated **OpenWebMath** dataset for math-domain experiments, or
	
2. Scores each token based on excess loss between reference and training models

3. Selectively applies loss only to high-value tokens during pre-training

### Key Results

- **Math Performance**: 30% improvement on GSM8k/MATH with 10x fewer tokens
- **General Tasks**: 6.8% average improvement across 15 benchmarks  
- **Efficiency**: Matches DeepSeekMath-7B performance using only 3% of tokens (15B vs 500B)
- **State-of-the-art**: RHO-1-1B achieves 40.6% on MATH (first 1B parameter model to exceed 40%)
![[Screenshot 2025-11-04 at 7.39.24 AM.jpeg]]
---

## Architecture Overview

![[Screenshot 2025-11-03 at 7.43.05 PM.png]]
### Selective Language Modeling (SLM) Pipeline

``` 
Algorithm: Selective Language Modeling (SLM)

Input: 
  - Base model θ
  - Pretraining corpus D
  - High-quality reference corpus D_ref
  - Selection ratio k%

Step 1: Train Reference Model
  θ_ref = TrainLM(D_ref)  // Standard CLM on curated data
  
Step 2: Score Tokens
  for each sequence x in D:
    for each token x_i in x:
    
	  // Reference Loss (reference model trained on data)
      L_ref(x_i) = -log P(x_i|x_<i; θ_ref) 
      
      // Current model loss (fresh model - no prior finetuning)   
      L_θ(x_i) = -log P(x_i|x_<i; θ) 
      
       // Excess loss (difference between current and reference loss) 
      Score(x_i) = L_θ(x_i) - L_ref(x_i)      
      
      
Step 3: Token Selection (per batch)
 
  Let I ← indices of the top ⌈k·|tokens in batch|⌉ entries of S (ties broken arbitrarily)


Step 4: Selective Loss and Update (per batch)

   L_SLM(θ) ← (1/|I|) · Σ_{i∈I} [ −log P_θ(x_i | x_<i) ]
   
   Where, I = set of tokens in top k% of excess loss from scoring step.
   
   θ ← OptimizerStep(θ, ∇_θ L_SLM)

5. Iterate steps 2–4 until convergence or token budget exhausted.

Output: Trained model θ_final
```


#### Token Selection Logic in RHO-1

| **Reference Loss** | **Current Loss** | **Excess Loss (Lcur − Lref)** | **Interpretation**                                                                                                       | **Selected for Training?** |
| ------------------ | ---------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| **Low**            | **High**         | **Large Positive**            | Token was _easy_ for the reference model but _hard_ for the current model → contains valuable, learnable math knowledge. | ✅ **Yes**                  |
| **High**           | **High**         | **Small / ≈ 0**               | Both models struggle → token is likely _noisy_, _rare_, or _unlearnable_.                                                | 🚫 **No**                  |
| **Low**            | **Low**          | **Small / ≈ 0**               | Both models find it _easy_ → redundant or trivial token (e.g., “the”, “and”).                                            | 🚫 **No**                  |
| **High**           | **Low**          | **Negative**                  | Current model already learned it better than the reference → no further learning benefit.                                | 🚫 **No**                  |
### Key Differences from Standard Pretraining

| Aspect               | Causal Language Modeling (CLM)   | Selective Language Modeling (SLM) |
| -------------------- | -------------------------------- | --------------------------------- |
| **Loss Application** | All tokens uniformly             | Top k% tokens by score            |
| **Token Selection**  | None                             | Dynamic, based on excess loss     |
| **Reference Model**  | Not used                         | Guides token importance           |
| **Efficiency**       | Processes all data               | Focuses on high-value tokens      |
| **Convergence**      | Slower, unstable on noisy tokens | Faster, more stable               |

---

## Model Performance

### Math Benchmarks (Few-shot)
- The **GSM8k** and **MATH** columns show accuracy on two math reasoning benchmarks.

| Model              | Size | Tokens | GSM8k  | MATH   | Average    |
| ------------------ | ---- | ------ | ------ | ------ | ---------- |
| **Baseline (CLM)** | 1B   | 15B    | 6.4%   | 2.4%   | 4.4%       |
| **RHO-1-Math**     | 1B   | 9B*    | 29.8%  | 14.0%  | 21.9%      |
| **Improvement**    | -    | -40%   | +23.4% | +11.6% | **+17.5%** |
|                    |      |        |        |        |            |
| **Baseline (CLM)** | 7B   | 15B    | 42.9%  | 22.2%  | 32.6%      |
| **RHO-1-Math**     | 7B   | 10.5B* | 66.9%  | 31.0%  | 49.0%      |
| **Improvement**    | -    | -30%   | +24.0% | +8.8%  | **+16.4%** |

*Selected tokens only

### Fine-tuned Results (MATH Dataset)
After pretraining, the models were fine-tuned specifically on the MATH dataset

| Model         | Size | MATH Score | Previous SOTA          |
| ------------- | ---- | ---------- | ---------------------- |
| **RHO-1**     | 1B   | **40.6%**  | First 1B to exceed 40% |
| **RHO-1**     | 7B   | **51.8%**  | Matches DeepSeekMath   |
| GPT-4 (early) | -    | 42.5%      | For reference          |
 Takeaway: This means RHO-1 learns _faster_ and _better_ because it ignores low-value tokens.
---

## Question 1
If the reference model is already fine-tuned on clean, high-quality math data, why not just use it directly for MATH instead of using it to calculate excess loss and continue pre-training a new RHO-1 model? Why does RHO-1 perform better?

<details>
<summary>Click to reveal answer</summary>
The reference model is small and trained on only 0.5 billion math tokens—enough to score token quality but not to reason or generalize well. Its job is to find useful tokens, not to generate text. The main RHO-1 model, built on a broadly trained base, already has strong general knowledge, so selective math pre-training adds reasoning skills without losing fluency. Using the reference model directly would give a narrow math model with poor general ability.
</details>

---

## Question 2
Consider this: The paper shows that only 26% of tokens meaningfully contribute to learning (H→L category). If we're throwing away 40-70% of tokens during training, what prevents the model from developing "blind spots" or losing important contextual understanding?

<details>
<summary>Click to reveal answer</summary>

The key insight is that SLM doesn't permanently discard tokens - it dynamically selects them based on the current training state. Tokens that are "easy" (L→L) are already well-learned, so skipping them doesn't create blind spots. The reference model acts as a safety net, ensuring that important patterns are preserved. Additionally, token selection changes throughout training: what's considered "unimportant" at 2B tokens might become important at 8B tokens, allowing the model to revisit and refine different aspects of its knowledge as needed.
</details>

---

## Impact

### Paradigm Shift in Pretraining
RHO-1 fundamentally challenges the assumption that all tokens deserve equal treatment during pretraining. This work demonstrates that **quality beats quantity** at the most granular level - individual tokens.

### Immediate Contributions
1. **Efficiency Revolution**: Achieving comparable performance with 3-10x fewer tokens dramatically reduces computational costs
2. **Emergent Behaviors**: Self-correction and reflection emerge without explicit programming
3. **Democratization**: Smaller models (1B) can now compete with much larger models on complex tasks
4. **Training Stability**: Focusing on learnable tokens reduces noise and training instability

### Broader Implications
- **Economic**: Reduces pretraining costs by an order of magnitude
- **Environmental**: Significantly lower carbon footprint for model training  
- **Accessibility**: Enables smaller organizations to train competitive models
- **Future Research**: Opens new directions in curriculum learning, dynamic data selection, and token-level optimization

### Position in the AI Timeline
- **Before RHO-1**: Uniform token treatment, scale-focused improvements
- **RHO-1's Contribution**: Selective token training, quality-focused efficiency
- **Future Directions**: Adaptive curricula, self-improving data selection, multi-resolution training

---

## Critical Analysis

### Strengths
 **Retains Generalizability** (not forgetful): Unlike fine-tuning, can continually pre-train on subject-specific content and see improvements in generalizability and subject-specific assessments. 
 **Empirically Validated**: Strong results across diverse benchmarks  
 **Theoretically Grounded**: Clear mathematical framework with intuitive motivation  
 **Practically Efficient**: Significant computational savings without complex infrastructure  

### Limitations and Areas for Improvement

1. **Reference Model Dependency**: Requires high-quality curated data for reference model training
   - Could explore self-referential approaches or weaker supervision

2. **Fixed Selection Ratio**: Uses static k% throughout training
   - Adaptive selection ratios could further optimize efficiency

### Future Research Directions
- **Multi-granular selection**: Combining token, sentence, and document-level selection
- **Online adaptation**: Dynamically adjusting selection criteria during training
- **Cross-domain transfer**: Using reference models from different domains
- **Theoretical analysis**: Formal convergence proofs and sample complexity bounds

---

## Code Demonstration

```python
import torch
import torch.nn.functional as F

class SelectiveLanguageModeling:
    """
    Simplified implementation of Selective Language Modeling (SLM)
    """
    def __init__(self, model, reference_model, selection_ratio=0.6):
        self.model = model
        self.reference_model = reference_model
        self.selection_ratio = selection_ratio
        
    def compute_token_scores(self, input_ids, attention_mask):
        """
        Compute excess loss scores for each token
        """
        with torch.no_grad():
            # Get reference model loss per token
            ref_outputs = self.reference_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
            ref_loss = F.cross_entropy(
                ref_logits.view(-1, ref_logits.size(-1)),
                input_ids.view(-1),
                reduction='none'
            )
            
        # Get current model loss per token  
        model_outputs = self.model(input_ids, attention_mask=attention_mask)
        model_logits = model_outputs.logits
        model_loss = F.cross_entropy(
            model_logits.view(-1, model_logits.size(-1)),
            input_ids.view(-1),
            reduction='none'
        )
        
        # Compute excess loss (higher = more important to learn)
        scores = model_loss - ref_loss.detach()
        return scores
    
    def selective_loss(self, input_ids, attention_mask, labels):
        """
        Compute loss only on selected tokens
        """
        # Get token scores
        scores = self.compute_token_scores(input_ids, attention_mask)
        
        # Select top k% tokens
        k = int(len(scores) * self.selection_ratio)
        top_k_indices = torch.topk(scores, k).indices
        
        # Create selection mask
        selection_mask = torch.zeros_like(scores)
        selection_mask[top_k_indices] = 1.0
        
        # Compute loss only on selected tokens
        model_outputs = self.model(input_ids, attention_mask=attention_mask)
        model_logits = model_outputs.logits
        
        loss = F.cross_entropy(
            model_logits.view(-1, model_logits.size(-1)),
            labels.view(-1),
            reduction='none'
        )
        
        # Apply selection mask
        selected_loss = (loss * selection_mask).sum() / selection_mask.sum()
        
        return selected_loss

# Usage example
slm_trainer = SelectiveLanguageModeling(
    model=your_model,
    reference_model=reference_model,
    selection_ratio=0.6
)

# In training loop
loss = slm_trainer.selective_loss(input_ids, attention_mask, labels)
loss.backward()
optimizer.step()
```


---

## Resource Links

1. [Original Paper - ArXiv](https://arxiv.org/abs/2404.07965v4)
2. [GitHub Implementation](https://github.com/microsoft/rho)
3. [DeepSeekMath Comparison](https://github.com/deepseek-ai/DeepSeek-Math)

---

## Citation
```bibtex
@article{lin2024rho,
  title={RHO-1: Not All Tokens Are What You Need},
  author={Lin, Zhenghao and Gou, Zhibin and Gong, Yeyun and Liu, Xiao and Shen, Yelong and others},
  journal={arXiv preprint arXiv:2404.07965},
  year={2024},
  organization={Microsoft Research}
}
```

