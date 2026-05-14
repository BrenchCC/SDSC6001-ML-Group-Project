# Image Generation Prompts for Report and PPT

These prompts are repository-independent. They are intended for a web image-generation interface and can be used to create polished conceptual diagrams for the report or a later PPT production pass.

## 1. In-Context Linear Regression Prompt Structure

**Purpose:** Show how a linear-regression task is encoded into a Transformer prompt.

**Prompt:**
Create a clean academic vector-style diagram on a white background. Show a matrix-shaped prompt with columns labeled "context example 1", "context example 2", "...", "context example n", and "query". The top block is feature vectors x, the bottom row is labels y. The query label cell is visibly set to 0 and marked "masked query label". Add arrows from the context examples into a linear Transformer block, then an arrow to a prediction labeled y_hat for the query. Use restrained colors: deep blue for features, teal for labels, gray for masks, and orange for prediction. Use no repository paths, no code snippets, and no decorative gradients. Aspect ratio 16:9.

## 2. Transformer Layer as Preconditioned Gradient Descent

**Purpose:** Explain the main theoretical mechanism.

**Prompt:**
Create a publication-quality conceptual diagram showing a stack of three linear self-attention layers. Under each layer, show the corresponding optimization step: w_0 to w_1 to w_2 to w_3. Each step is labeled "preconditioned gradient descent" and includes a small matrix A_l beside the arrow. Add a covariance matrix Sigma on the left and show A_l approximately proportional to Sigma inverse. The diagram should communicate that Transformer depth corresponds to optimization iterations. Use simple geometric blocks, clean arrows, mathematical notation, and a professional NeurIPS-style visual tone. White background, 16:9 aspect ratio.

## 3. Paper-Scale Evidence vs Course-Scale Reproduction Evidence

**Purpose:** Visualize PSE and CRE terminology.

**Prompt:**
Create a two-column academic comparison diagram. The left column is titled "Paper-scale Structural Evidence (PSE)" and shows polished theoretical plots, a theorem icon, and matrix heatmaps. The right column is titled "Course-scale Reproduction Evidence (CRE)" and shows smaller-scale reproduced curves, heatmaps, and an extension experiment. In the center, place a shared label "same theoretical structure" with arrows pointing to both columns. Use neutral academic colors, consistent icon style, and no mention of hardware. Aspect ratio 16:9.

## 4. Theorem 3 Sparse Setting Mechanism

**Purpose:** Explain why rotated A matrices are checked.

**Prompt:**
Create a mathematical flow diagram for Theorem 3. Show input covariance Sigma, learned matrices A_0, A_1, A_2, and the transformation Sigma^{1/2} A_i Sigma^{1/2}. The transformed matrices should visually become diagonal or identity-like heatmaps. Add a small note "closer to scaled identity" and contrast it with raw A_i matrices that remain less identity-like. Use clear matrix tiles, arrows, and minimal text. White background, 16:9 aspect ratio, academic style.

## 5. Theorem 4 Relaxed Setting and Feature Transformation

**Purpose:** Show the additional role of B matrices.

**Prompt:**
Create a clean academic diagram for a relaxed linear Transformer layer. Split each layer into two coupled parts: a preconditioned gradient step using A_i and a feature transformation using B_i. Show A_i aligned with Sigma inverse and B_i aligned with the identity matrix. Add a small Gram matrix before and after the feature transformation, with the after matrix visually better conditioned. Use restrained blue, green, and amber accents. No code, no repository references, 16:9 aspect ratio.

## 6. Context Length and Sample Complexity Perspective

**Purpose:** Support the extension experiment in the PPT.

**Prompt:**
Create an explanatory chart-style illustration about context length in in-context learning. Show prompts with increasing numbers of examples N = 4, 8, 12, 16, 20 on the x-axis, and decreasing prediction loss on the y-axis. Include four method labels: Linear Transformer, Gradient Descent, Preconditioned Gradient Descent, and OLS. The visual should emphasize that more context examples provide more task information and generally reduce test loss. Use a clean scientific presentation style with readable labels. Aspect ratio 16:9.

## 7. Final Summary Slide Visual

**Purpose:** Provide a closing visual for the PPT.

**Prompt:**
Create a high-level academic summary graphic with four connected nodes: "Optimization Error", "Computational Complexity", "Expressive Capacity", and "Sample Complexity". In the center place "Linear Transformer learns preconditioned gradient descent for ICL". Connect the center to each node with thin arrows. Add subtle matrix and curve motifs in the background, but keep the design uncluttered and readable. White or very light background, professional conference-presentation style, 16:9 aspect ratio.
