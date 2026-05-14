# English Speech Script Template

> Target duration: under 15 minutes. A practical pace is 35 to 45 seconds per slide, with up to 60 seconds for key theory or result slides.

## 1. Title

Hello everyone. Our project studies how Transformers can learn to implement preconditioned gradient descent for in-context learning. The paper we reproduce is the NeurIPS 2023 paper by Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra.

## 2. Course Relevance

This paper mainly fits the course topic of optimization error and computational complexity. It interprets a finite-depth attention network as a learned optimization procedure. It also touches expressive capacity, because the model must represent gradient-descent-like updates, and our context-length extension adds a sample-complexity perspective.

## 3. Core Question

The paper does not only ask whether Transformer weights can be manually constructed to simulate gradient descent. The stronger question is whether training over random tasks naturally drives the model toward a gradient-based algorithm.

## 4. In-Context Learning Background

In in-context learning, a prompt contains several examples and a query. The model parameters are fixed, but the model should infer the current task from the prompt and predict the query label.

## 5. Linear Regression Prompt

This slide uses the AI-generated prompt-structure diagram. The paper studies a controlled linear regression setting. The first n examples have labels, while the query label is set to zero to avoid label leakage. This matrix representation makes the theory analytically tractable.

## 6. Linear Self-Attention

The paper removes the softmax and studies linear attention. This is not meant as a stronger practical architecture; it is a simplification that allows the training objective and learned weights to be analyzed. The mask ensures that attention uses only the labeled context examples.

## 7. Theorem 1

The single-layer result shows that the global optimum implements one step of preconditioned gradient descent. The preconditioner adapts to the input covariance, so the learned update is not plain gradient descent.

## 8. Multi-Layer Mechanism

This slide uses the theoretical-mechanism diagram. In the multi-layer setting, each attention layer can correspond to one optimization iteration. Transformer depth can therefore be interpreted as the number of learned optimization steps.

## 9. Theorem 3

This slide uses the Theorem 3 mechanism diagram. Theorem 3 studies the sparse constrained multi-layer setting. The theory predicts that each A matrix should align with the inverse covariance matrix. Experimentally, we check whether the rotated matrix becomes closer to a scaled identity.

## 10. PSE for Theorem 3

This is the paper-scale structural evidence. The blue curve, which measures the rotated distance, decreases clearly, while the raw identity distance does not decrease in the same way. This supports the preconditioner interpretation.

## 11. CRE for Theorem 3

Here is our course-scale reproduction evidence. The loss decreases, and the rotated distance for A0 also decreases strongly. The exact numerical scale is different, but the core structural trend matches the paper.

## 12. Theorem 3 Heatmap

The heatmap gives another view of the same result. After rotation, A0 becomes close to a diagonal or identity-like structure, which is consistent with the inverse covariance preconditioner.

## 13. Theorem 4

This slide uses the Theorem 4 feature-transformation diagram. Theorem 4 relaxes the parameter constraints. The model can now do more than preconditioned gradient descent: A matrices implement preconditioned steps, while B matrices transform the features.

## 14. PSE for Theorem 4

The paper-scale results show that B matrices approach scaled identity structures, while A matrices keep the preconditioned structure. This means the learned algorithm goes beyond standard gradient descent.

## 15. CRE for Theorem 4

In our reproduction, the loss decreases and the structural distances for A and B move in the direction predicted by the theory. This supports the feature-transformation interpretation in the relaxed setting.

## 16. Theorem 4 Heatmap

The B0 heatmap shows a structured feature-transformation matrix. It is not random; it moves toward an identity-like pattern.

## 17. Context-Length Extension

We also study how performance changes as the number of in-context examples increases. Overall, more examples reduce the loss for the Transformer and preconditioned GD. The visible spike around N equals 12 mainly comes from the vanilla GD curve. We interpret it as finite-step fixed-stepsize sensitivity under a non-isotropic covariance structure. Preconditioned GD is much more stable, which supports the paper's message that preconditioning improves optimization.

## 18. PSE versus CRE

PSE refers to the official paper-scale structural evidence. CRE refers to our course-scale reproduction evidence. The goal is not exact numerical equality, but structural consistency with the theory.

## 19. Limitations and Extensions

The analysis applies to linear attention and Gaussian linear regression tasks. It should not be directly generalized to all large language models. Future work could study softmax attention, richer task distributions, and other critical points of the non-convex training objective.

## 20. Conclusion

The final slide uses the summary-framework diagram. The main takeaway is that Transformer in-context learning can be studied as learned optimization. Our reproduction supports the core structural claim: trained linear Transformers can implement preconditioned gradient descent and, in the relaxed setting, feature transformations.
