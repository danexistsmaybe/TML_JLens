# CS 6501: Trustworthy Machine Learning — Project Catalog

**20 candidate projects · teams of two · six weeks**  
Read [Course Project Instructions](Course_Project_Instructions.md) before selecting a project.

## How to use this catalog

Choose a candidate project or [propose your own](Course_Project_Instructions.md#student-proposed-projects). Every candidate brief gives a research question, a manageable default configuration, public starting resources, baseline comparisons, modules you must implement, evaluation requirements, and highly suggested technical milestones. The numerical configurations below are assignment starting points, not claims about configurations used in the cited papers.

Circuit options include **P06 (grokking), P11 (faithfulness), P12 (induction robustness), and P13 (vision concept circuits)**. The full portfolio includes three conventional explainability, two concept bottleneck, two SAE, two J-lens, four circuit, three domain adaptation/generalization, two TTA, and two TTT projects.

### Required course checkpoints

| Week | Required action |
|---|---|
| 1 | Finish the first-day checklist and give a five-minute proposal talk. A written proposal is not required. |
| 2–5 | Submit one milestone report through Canvas in each week. |
| 6, at the end of the semester | Give a 10-minute final presentation followed by five minutes of Q&A; submit the research report and reproducibility package. |

The W1–W6 technical plans in the briefs are **highly suggested progress targets**. You may adjust the sequence to your project's needs while completing the required checkpoints and weekly reports. Week 1 pilot findings support the proposal talk; Week 6 technical outputs support the final presentation and submissions.

For a project beyond this list, complete an equivalent first-day checklist, present your proposal in the Week 1 five-minute talk, and obtain **advisor approval during that talk**. Establish the minimum implementation and evaluation scope with the advisor. The same Canvas reporting and final-submission requirements apply.

### Difficulty and computing labels

- **D1 — Accessible:** familiar models and straightforward instrumentation.
- **D2 — Intermediate:** custom methods, several modules, and careful controls.
- **D3 — Advanced:** sensitive circuit instrumentation or Jacobian computations.
- **C0:** CPU-friendly; **C1:** one 8–12 GB GPU; **C2:** one 12–16 GB GPU; **C3:** one 16–24 GB GPU recommended.

All computing ranges are **unbenchmarked planning estimates for the minimum complete scope**, including its main comparisons and repetitions. Measure a pilot before committing to full runs. Disk estimates cover the recommended subset, model, and caches, not every optional resource. No core project requires a paid API or multi-GPU system.

## Project selection overview

| ID | Project | Area | Difficulty | Compute | Estimated core compute |
|---|---|---|---|---|---|
| [P01](#p01-attribution-under-distribution-shift) | Attribution under distribution shift | Conventional explainability | D2 | C1 | 6–16 GPU-hours |
| [P02](#p02-realistic-neighborhoods-for-local-explanations) | Realistic neighborhoods for local explanations | Conventional explainability | D1 | C0 | 2–8 CPU-hours |
| [P03](#p03-correct-predictions-wrong-evidence) | Correct predictions, wrong evidence | Conventional explainability/rationales | D2 | C1 | 4–12 GPU-hours |
| [P04](#p04-budgeted-concept-interventions) | Budgeted concept interventions | Concept bottleneck models | D2 | C1 | 6–16 GPU-hours |
| [P05](#p05-measuring-and-reducing-concept-leakage) | Measuring and reducing concept leakage | Concept bottleneck models | D2 | C1 | 4–12 GPU-hours |
| [P06](#p06-predicting-grokking-from-circuit-formation) | Predicting grokking from circuit formation | Circuits/training dynamics | D2 | C1 | 8–30 GPU-hours |
| [P07](#p07-data-selection-for-sae-concept-coverage) | Data selection for SAE concept coverage | Sparse autoencoders | D2 | C2 | 10–30 GPU-hours |
| [P08](#p08-stable-sae-features-versus-causally-useful-features) | Stable SAE features versus causally useful features | Sparse autoencoders | D2 | C2 | 12–35 GPU-hours |
| [P09](#p09-an-economical-jacobian-lens) | An economical Jacobian lens | J-lens | D3 | C3 | 15–50 GPU-hours |
| [P10](#p10-does-the-j-lens-fitting-corpus-change-the-interpretation) | Does the J-lens fitting corpus change the interpretation? | J-lens | D3 | C3 | 20–50 GPU-hours |
| [P11](#p11-when-is-a-discovered-circuit-faithful) | When is a discovered circuit faithful? | Circuit evaluation | D3 | C3 | 10–35 GPU-hours |
| [P12](#p12-robustness-of-induction-head-circuits) | Robustness of induction-head circuits | Circuit robustness | D2 | C1 | 3–12 GPU-hours |
| [P13](#p13-vision-concept-circuits-under-background-shift) | Vision concept circuits under background shift | Vision circuits | D3 | C3 | 15–45 GPU-hours |
| [P14](#p14-preventing-negative-transfer-from-pseudo-labels) | Preventing negative transfer from pseudo-labels | Domain adaptation | D2 | C1 | 6–18 GPU-hours |
| [P15](#p15-domain-generalization-with-an-imperfect-domain-graph) | Domain generalization with an imperfect domain graph | Domain generalization/TRO | D2 | C1 | 6–20 GPU-hours |
| [P16](#p16-generating-useful-training-environments) | Generating useful training environments | Domain generation/generalization | D2 | C2 | 15–40 GPU-hours |
| [P17](#p17-recovering-from-changing-test-environments) | Recovering from changing test environments | Test-time adaptation | D2 | C2 | 10–30 GPU-hours |
| [P18](#p18-test-time-adaptation-under-class-imbalance) | Test-time adaptation under class imbalance | Test-time adaptation | D2 | C2 | 10–30 GPU-hours |
| [P19](#p19-choosing-a-useful-self-supervised-task-for-ttt) | Choosing a useful self-supervised task for TTT | Test-time training | D2 | C2 | 20–50 GPU-hours |
| [P20](#p20-deciding-whether-to-accept-a-test-time-training-update) | Deciding whether to accept a test-time training update | Test-time training | D2 | C2 | 15–40 GPU-hours |

**For a CPU-only team:** start with P02. P05, P06, P14, and P15 have smaller CPU configurations, but their runtimes require a pilot. Difficulty and compute are separate: P11 requires no model training yet has considerable engineering and gradient-computation demands.

## Shared data and model starters

Use these specifications wherever a brief refers to a shared starter.

### S1. CIFAR classification and corruption starter

- **Public data:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html); [CIFAR-10-C downloads and benchmark documentation](https://github.com/hendrycks/robustness).
- **Training/development:** split the official 50,000 training images into 45,000 training and 5,000 development images, stratified with a saved seed. Fit transforms and hyperparameters using these partitions only.
- **Testing:** official 10,000 clean test images and their CIFAR-10-C versions. A core evaluation may use a fixed, class-balanced 2,000-image test subset. Preserve image IDs across corruptions for paired comparisons. CIFAR-10-C versions of the same image are correlated.
- **Development shifts:** generate Gaussian noise, brightness changes, and blur on development images. Hold out at least one corruption family from method tuning. Do not use the benchmark test images as development data.
- **Trainable default:** a small CNN with four convolutional blocks, batch normalization, ReLU, channel widths 32/64/128/128, pooling, and a 10-class head. Train up to 30 epochs; choose the checkpoint using development accuracy. A CIFAR-adapted ResNet-18 is an optional alternative. All methods in a comparison must share the architecture and initial checkpoint.
- **Public fixed-checkpoint option:** [RobustBench](https://github.com/RobustBench/robustbench) provides a CIFAR-10 model named Standard, loaded with dataset=cifar10 and threat_model=Linf. Here it serves as a conventional fixed classifier, not an adversarial-robustness claim. It may be used for P01 or a pilot; source-trained experiments in P16–P20 use your own saved split and checkpoint.
- **Fixed-checkpoint split caveat:** a released checkpoint may have been trained on all 50,000 source training images. In that case, your 5,000 development images are held out from your method's fitting, not from the classifier's training. Disclose this distinction, or train the source classifier on the 45,000-image partition when a fully held-out development estimate is needed.

### S2. CUB concept starter

- **Public data:** [CUB-200-2011 images, attributes, part locations, and segmentations](https://www.vision.caltech.edu/datasets/cub_200_2011/).
- **Default subset:** species IDs 1–40, retaining the official train/test membership. Reserve 20% of official training images per species for development.
- **Concepts:** select 24 binary attributes using training data only; require enough positive and negative examples. Use per-image attribute annotations, not species labels converted into per-image ground truth. Mask missing or low-confidence concept labels and report the rule.
- **Starting encoder:** frozen [ImageNet-pretrained ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Cache pooled features, then train small concept and task heads.
- **Access caveat:** CUB is available for research/education and documents possible overlap with ImageNet pretraining. These projects study concept behavior under a declared pretrained setup.

### S3. Small language-model starter

- **Model/checkpoint:** [GPT-2 Small](https://huggingface.co/openai-community/gpt2); use the supplied tokenizer and record the revision.
- **Corpus:** [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), configuration wikitext-2-raw-v1. Use its official train/validation/test partitions; never concatenate partitions before generating windows.
- **Tools/tutorials:** [TransformerLens documentation](https://transformerlensorg.github.io/TransformerLens/) and [repository](https://github.com/TransformerLensOrg/TransformerLens). Use hooks for access to model internals; implement the designated algorithm yourselves.
- **Conventions:** save whether layer IDs are zero-based, the exact hook location, token position, padding mask, and whether the residual is measured before or after a block. Avoid silently changing normalization or weight-processing conventions between methods.

### S4. Vision SAE starter

- **Model/checkpoint:** frozen [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32), including its official processor.
- **Public probing images:** [ViSAE dataset](https://huggingface.co/datasets/deeprealaiml/ViSAE/tree/main).
- **Public concept vocabulary:** [ViSAE concept list](https://github.com/deep-real/ViSAE/blob/main/concept_set/ours_16K.txt).
- **Reference implementation:** [ViSAE](https://github.com/deep-real/ViSAE). The default projects train their own small SAEs rather than depending on an unreleased SAE checkpoint.
- **Subset and split:** choose a documented 8,000-image subset; split by original image identity into 6,000 training, 1,000 development, and 1,000 testing images. Remove duplicate identities before splitting. Use smaller subsets where the brief specifies a fallback.
- **Activations:** cache one CLS residual per image at two selected block outputs, rather than every patch token at every layer. A useful initial choice is zero-based blocks 3 and 7, after verifying hook shapes.
- **SAE:** start with a ReLU encoder, linear decoder, reconstruction loss plus an L1 penalty, and expansion factor 4. Choose the sparsity penalty on development data. Compare at matched average active-feature count and report dead features.

### S5. Synthetic-data starter

Write a deterministic generator using [Pillow drawing operations](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html), NumPy, or ordinary token sequences. Save the latent variables, labels, masks or prompt roles, generation parameters, and random seeds. Separate random draws and, where specified, compositions/templates between training, development, and testing. The complete generator makes these datasets publicly reproducible; no private dataset is needed.

---

## P01. Attribution under distribution shift

**Area:** conventional explainability  
**Difficulty:** D2 — requires attribution debugging and careful separation of explanation and prediction changes.  
**Compute:** C1; 8–12 GB GPU; estimated 6–16 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** PyTorch autograd, image classification, and basic statistical evaluation.

### Research question

Can instability in an attribution map provide an early warning of prediction failure as corruption severity increases, beyond the model's confidence?

### Start with these resources

- **Required:** [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) — integrated gradients.
- **Required:** [Grad-CAM](https://arxiv.org/abs/1610.02391) — activation-based localization.
- **Research context:** [Are Data-Driven Explanations Robust Against Out-of-Distribution Data?](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Are_Data-Driven_Explanations_Robust_Against_Out-of-Distribution_Data_CVPR_2023_paper.html).
- **Tutorial/checking tools:** [Captum tutorials](https://captum.ai/tutorials/); optional [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292).
- **Data/model:** S1, with the fixed Standard checkpoint permitted.

### System and required original work

Build a pipeline that takes an image and a sequence of corruption severities, produces attribution maps and predictions, and computes an explanation-stability warning score.

**Must implement yourselves:** integrated gradients with a declared baseline and integration rule; Grad-CAM; map alignment/normalization; and the warning score. Captum is a correctness comparator.

Use a fixed reference class when comparing maps across a corruption trajectory. Also report predicted-class maps separately. Otherwise a class change can explain a map change trivially. Develop a score combining rank stability, spatial agreement, or attribution dispersion; compare it with confidence and prediction-margin warnings. Fit thresholds on development trajectories only.

### Experiments and evaluation

Baselines: confidence/margin, each attribution method alone, and a simple map-distance score. Use 500 test image identities, three corruption families, and three nonzero severities plus clean images. Include images whose predicted class stays unchanged.

Report error-detection AUROC, false-alarm rate at a development-selected threshold, warning lead in severity levels, and map stability conditional on unchanged prediction. Evaluate deletion/insertion under two replacement rules and include a randomized-model control. A stable map is not automatically a faithful map.

**Ablations:** remove each component of the score; vary the IG baseline or spatial normalization. Bootstrap by image identity, not by individual corrupted views.

### Minimum scope, suggested milestones, and fallback

Minimum: one model, two attribution methods, one original warning score, and the specified corruption analysis.

**W1:** verify gradients/maps on 20 images. **W2:** complete attribution implementations and baseline trajectories. **W3:** implement the warning system. **W4:** evaluate the frozen score. **W5:** run controls and uncertainty analysis. **W6:** package maps, results, and a failure demonstration.

Fallback: 200 images and two corruption families, retaining both methods and controls. Extension: compare the warning on an independently trained classifier.

**First start:** compute IG and Grad-CAM for five clean/noisy pairs and check that increasing the integration steps stabilizes IG numerically.

---

## P02. Realistic neighborhoods for local explanations

**Area:** conventional explainability  
**Difficulty:** D1 — accessible implementation, with a substantive experimental-design challenge.  
**Compute:** C0; 2–8 CPU-hours; 8 GB system RAM and less than 1 GB disk.  
**Prerequisites:** regression, classification, and tabular preprocessing.

### Research question

Does a dependency-aware perturbation neighborhood improve local surrogate fidelity and stability compared with independent feature perturbations?

### Start with these resources

- **Required:** [“Why Should I Trust You?”](https://arxiv.org/abs/1602.04938) — LIME.
- **Required/context:** [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) — SHAP as a comparison framework.
- **Tutorial/baselines:** [LIME repository and examples](https://github.com/marcotcr/lime); [SHAP examples](https://github.com/shap/shap).
- **Public real data:** [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult).
- **Synthetic-data support:** [scikit-learn synthetic classification documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html). Implement the dependency rules explicitly.

### Data, model, and system

Start with 10,000 synthetic examples containing ten features, including a correlated pair and a known label-generation rule. Include an analytic linear predictor for a numerical correctness check. For Adult, use the published training/test files and reserve 20% of training for development. Fit all encoding and imputation on training only. Use logistic regression and a random forest; no checkpoint is needed.

Build a local-explanation service returning the surrogate coefficients, prediction, fidelity, and a neighborhood-validity diagnostic.

**Must implement yourselves:** weighted local regression, an independent sampler, and a dependency-aware sampler. One feasible design draws complete records from nearby training examples and conditionally perturbs numeric features within compatible categorical strata. Explain which dependencies it preserves.

Your original component is the sampler or its validity-aware weighting rule. Calling LIME/SHAP alone does not complete the project.

### Experiments and evaluation

Explain 100 fixed test cases using equal query budgets of 200 and 500 samples. Compare independent sampling, training-neighbor sampling, and your method. SHAP may be an additional library baseline.

Measure fidelity on a separate, predefined neighborhood rather than on the samples used to fit the surrogate. Report coefficient rank stability over five sampling seeds, neighborhood distance/validity, and runtime. Use the analytic predictor to check implementation correctness; the synthetic label-generation factors are not automatically ground truth for what a trained forest uses.

**Ablations:** remove dependency constraints; remove locality weighting. Include a synthetic dataset with independent features to test whether the new sampler introduces unnecessary bias.

### Minimum scope, suggested milestones, and fallback

Minimum: synthetic and Adult data, two prediction models, and one original sampler.

**W1:** inspect dependencies and establish splits. **W2:** implement weighted surrogates and independent sampling. **W3:** build the sampler and explanation interface. **W4:** compare fidelity/stability. **W5:** analyze independent-feature controls and failures. **W6:** deliver reproducible explanations and report.

Fallback: reduce explanation cases to 50, retaining both datasets and controls. Extension: study feature groups rather than individual coefficients.

**First start:** recover the coefficients of a known linear predictor using your weighted-regression implementation.

---

## P03. Correct predictions, wrong evidence

**Area:** conventional explainability and rationale correctness  
**Difficulty:** D2 — requires controlled data generation and an evidence audit.  
**Compute:** C1; 8–12 GB GPU; estimated 4–12 GPU-hours; approximately 1–3 GB disk.  
**Prerequisites:** CNN training, saliency methods, and image manipulation.

### Research question

Can an attribution-based auditor identify correct predictions that depend on a shortcut rather than the task's intended visual evidence?

### Start with these resources

- **Required:** [Beyond Accuracy: Ensuring Correct Predictions With Correct Rationales](https://arxiv.org/abs/2411.00132).
- **Required:** [Grad-CAM](https://arxiv.org/abs/1610.02391).
- **Released resources/context:** [DCP repository, rationale data, and evaluation examples](https://github.com/deep-real/DCP). Use released resources; generating new rationales through a paid API is unnecessary.
- **Implementation tutorial:** [Captum](https://captum.ai/tutorials/).
- **Data:** S5 for the core; S2/CUB masks and parts for an optional natural-image audit.

### Data, model, and system

Generate 64×64 images with one circle, square, or triangle; the class is the shape. Save its foreground mask. Independently vary size, position, and foreground color. Correlate background texture with the shape label in 90% of training examples; use a weaker development correlation and reverse it in one test environment.

Create 10,000 training, 2,000 development, and 2,000 test images with disjoint generation seeds. Train a small CNN on the biased data and another on randomized backgrounds. The foreground mask defines intended evidence, while actual reliance is tested through interventions.

**Must implement yourselves:** the generator and masks, Grad-CAM, foreground/background counterfactuals, and an auditor combining localization and intervention sensitivity. The system returns a prediction, evidence map, audit score, and flagged explanation.

Develop the auditor on development data. Use object-preserving background swaps to test shortcut reliance. Removing the foreground tests a different question and may create unrealistic inputs; report it separately.

### Experiments and evaluation

Baselines: confidence-only flagging, foreground attribution mass, and background-swap sensitivity alone.

Report ordinary accuracy, intended-evidence localization, accuracy under background reversal, and audit precision/recall against a declared intervention-based shortcut criterion. On correctly classified examples, measure the fraction passing both label correctness and the stated evidence test. Describe this as operational double correctness under your evaluator.

**Ablations:** remove localization or counterfactual sensitivity from the auditor; change the shortcut strength. Include a randomized-background classifier as a negative control.

### Minimum scope, suggested milestones, and fallback

Minimum: controlled synthetic data, two source-trained classifiers, and an original auditor.

**W1:** generate and inspect images/masks. **W2:** train baselines and implement maps. **W3:** build counterfactuals and auditor. **W4:** test reversed backgrounds. **W5:** ablate the audit and inspect false flags. **W6:** demonstrate correct predictions with contrasting evidence.

Fallback: 5,000 training images; keep all three classes and both controls. Extension: audit 100 CUB examples using object masks and selected part annotations, clearly separating object localization from attribute-specific evidence.

**First start:** construct ten pairs with identical shapes and different backgrounds; confirm that labels and foreground masks remain unchanged.

---

## P04. Budgeted concept interventions

**Area:** concept bottleneck models  
**Difficulty:** D2 — concept-label handling, intervention policies, and cost-aware evaluation.  
**Compute:** C1; 8–12 GB GPU; estimated 6–16 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** multi-label classification and small neural-network heads.

### Research question

Which concept should a system ask to correct next when only a few concept interventions are available?

### Start with these resources

- **Required:** [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html).
- **Additional context:** [Promises and Pitfalls of Black-Box Concept Learning Models](https://arxiv.org/abs/2106.13314).
- **Tutorial/reference baseline:** [ConceptBottleneck repository](https://github.com/yewsiang/ConceptBottleneck).
- **Data/model:** S2, including per-image attributes and the ResNet-18 checkpoint.

### Model and implementation

Use frozen image features. Train a 24-output sigmoid concept head, then a species-classification head on concept probabilities. Include an independent/sequential CBM baseline and a direct feature-to-label classifier. Reserve concept annotations of final-test images exclusively for the simulated intervention evaluator.

**Must implement yourselves:** concept-head/task-head training, concept replacement, and a budgeted selection policy. A feasible original policy combines concept uncertainty with the expected change in the task distribution when the concept is set to 0 or 1. Estimate the two outcome probabilities from the current concept head; do not inspect true concepts or species labels when selecting.

The system displays the predicted concepts, chooses a query, accepts its returned concept value, and recomputes the task prediction. Evaluate budgets of 0, 1, 2, 4, and 8 queried concepts.

### Experiments and evaluation

Baselines: random selection, uncertainty-only selection, and a fixed global concept ranking computed on development data. A true-error or label-aware oracle can be an explicitly privileged upper bound.

Report task accuracy versus intervention cost, area under the intervention-budget curve, concept accuracy/calibration, and results by initial concept-error count. Add 10% response noise in the simulated annotator. Do not query missing/low-confidence annotations; use the same eligible concept set for all policies.

**Ablations:** remove uncertainty or task influence from the policy; use uniform concept costs versus declared nonuniform costs. Train three seeds using the same source split.

### Minimum scope, suggested milestones, and fallback

Minimum: 40 species, 24 concepts, one CBM, three selection baselines, and one original policy.

**W1:** inspect annotation confidence and cache features. **W2:** train concept/task baselines. **W3:** implement the interactive policy. **W4:** evaluate budgets on untouched test images. **W5:** study noisy responses and policy ablations. **W6:** demonstrate an intervention sequence and report results.

Fallback: 20 species and 12 concepts, selected using the same rules. Extension: allow abstention when a query's expected benefit is lower than its cost.

**First start:** replace one predicted concept with its simulated observed value and verify that only the concept bottleneck, not image features, changes.

---

## P05. Measuring and reducing concept leakage

**Area:** concept bottleneck models  
**Difficulty:** D2 — identifying what a bottleneck actually communicates.  
**Compute:** C1; 8–12 GB GPU; estimated 4–12 GPU-hours; approximately 1–2 GB disk.  
**Prerequisites:** joint losses, representation probes, and controlled synthetic experiments.

### Research question

Can a model hide task-relevant information in soft concept values even when its declared concepts appear accurate, and can a constrained bottleneck reduce this channel?

### Start with these resources

- **Required:** [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html).
- **Required:** [Promises and Pitfalls of Black-Box Concept Learning Models](https://arxiv.org/abs/2106.13314).
- **Reference/tutorial:** [ConceptBottleneck implementation](https://github.com/yewsiang/ConceptBottleneck).
- **Data:** S5; optional S2 after the synthetic experiment works.

### Data and model

Generate images with three explicitly annotated binary concepts, such as shape, foreground color, and size. Add a fourth visible factor, such as stripe orientation, that is excluded from the declared concept set. Define the task so that it partly depends on this excluded factor. Balance the excluded factor within every declared concept combination.

Create 12,000/2,000/2,000 train/development/test examples. A complete concept vector therefore cannot perfectly determine the task label. Train a small image encoder with independent, sequential, and jointly optimized CBMs.

**Must implement yourselves:** these CBM objectives; a probe comparing task prediction from true binary concepts versus soft concepts; a within-concept-combination shuffle diagnostic; and a constrained bottleneck.

For the original component, implement stochastic binary sampling, quantization, or a regularizer penalizing within-concept-combination variation. Tune it on development data. A within-combination shuffle uses concept annotations only in the diagnostic evaluator; it is not a deployment capability.

### Experiments and evaluation

Baselines: true-concept-only classifier, direct image classifier, and unconstrained soft CBM. Compare your constraint with hard thresholding.

Report task accuracy, concept accuracy/calibration, excluded-factor predictability, the performance gap between soft and true-concept inputs, and sensitivity to concept correction. Interpret these as leakage diagnostics under the specified generator, not a universal proof of leakage.

**Ablations:** remove the constraint; vary its strength. Include a control task whose label depends only on declared concepts and test altered excluded-factor correlations. Use three training seeds and equal model capacity.

### Minimum scope, suggested milestones, and fallback

Minimum: one generator, three CBM training variants, one constrained bottleneck, and two complementary diagnostics.

**W1:** verify balanced latent factors. **W2:** train the CBM baselines. **W3:** implement diagnostics and constraint. **W4:** compare leakage/accuracy tradeoffs. **W5:** run the concept-complete control and ablations. **W6:** package a concept-correction demonstration.

Fallback: use vector inputs and a small MLP while retaining the same latent-factor and evaluation design. Extension: investigate a selected CUB attribute set, acknowledging that real attributes can be incomplete or noisy.

**First start:** estimate the task accuracy attainable from the true concepts alone and verify that it is below the direct-input model's attainable accuracy.

---

## P06. Predicting grokking from circuit formation

**Area:** mechanistic interpretability, circuits, and training dynamics  
**Difficulty:** D2 — small-model training with a demanding prospective evaluation.  
**Compute:** C1; 8–12 GB GPU; estimated 8–30 GPU-hours; approximately 2–5 GB disk. A smaller CPU configuration is possible after timing a pilot.  
**Prerequisites:** transformer basics, Fourier analysis, and training checkpoints.

### Research question

Can a circuit-based warning rule developed on some training runs predict delayed generalization on unseen runs better than loss and weight-based baselines?

### Start with these resources

- **Required:** [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217).
- **Required:** [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177).
- **Author tutorial/reference:** [Grokking analysis and notebooks](https://github.com/neelnanda-io/Grokking).
- **Model instrumentation:** [TransformerLens documentation](https://transformerlensorg.github.io/TransformerLens/).
- **Data:** generate all ordered input pairs for modular addition; no download or pretrained checkpoint is needed.

### Data, model, and prediction protocol

Use a one-layer transformer with width 64 or 128, four attention heads, a small MLP, and distinct tokens for the integers and equality marker. Start with p=59 and a 30% arithmetic-pair training fraction. Use three disjoint arithmetic partitions: training, development, and final testing. Save the split seed and checkpoint every 200–500 updates.

Run a pilot with AdamW and a small development-only weight-decay/learning-rate check. Delayed generalization is not guaranteed by dataset size alone. Set a maximum training budget after the pilot.

Plan six main runs: three development runs for constructing the warning and three held-out seeds for evaluating it. Add one reduced-training-fraction run as a transfer challenge and one configuration that fails to grok within the budget, if feasible. These small run counts support exploratory conclusions, not broad statistical claims.

### System and required original work

**Must implement yourselves:** the training/checkpoint pipeline; Fourier concentration and at least one intervention-based progress metric; and an online warning rule. The rule may use trends in circuit concentration or necessity scores.

Select frequencies from the current or earlier checkpoint only. Do not identify the final circuit and retrospectively use it to predict its own formation. Freeze the warning rule before examining held-out trajectories.

Define grokking in advance, for example sustained development accuracy above 90% after early overfitting; use a separate final arithmetic-test partition to confirm generalization. The target event is evaluated offline and never supplied to the warning algorithm during a held-out run.

### Experiments and evaluation

Baselines: training/development loss trends, weight norm, and simple Fourier concentration. Compare methods at matched alarm false-positive rates. Report warning lead time, false alarms, missed events, event-time error, and circuit intervention effects.

Do not compute correlation across every checkpoint as though checkpoints were independent samples. Report per-run outcomes and runs without an event as censored at the training limit.

**Ablations:** remove temporal smoothing or causal information; compare head-only ablation with Fourier-component ablation. Changing the modulus alone is not the original contribution.

### Minimum scope, suggested milestones, and fallback

Minimum: one arithmetic task, held-out-seed warning evaluation, one transfer condition, and a non-event analysis.

**W1:** obtain a delayed-generalization pilot. **W2:** launch main runs and implement metrics. **W3:** construct/freeze the warning on development runs. **W4:** evaluate held-out runs. **W5:** analyze transfer and false alarms. **W6:** package checkpoint trajectories and prediction demonstrations.

Fallback: width 64, less frequent checkpointing, and a 20,000-update cap justified by the pilot. If delayed generalization remains absent, study why the warning fails under that configuration rather than selecting only successful runs. Extension: transfer to modular subtraction.

**First start:** train one small model and plot training/development accuracy, loss, and Fourier concentration against updates.

---

## P07. Data selection for SAE concept coverage

**Area:** sparse autoencoders and vision concepts  
**Difficulty:** D2 — activation caching, sparse training, and independent concept evaluation.  
**Compute:** C2; 12–16 GB GPU; estimated 10–30 GPU-hours; approximately 8–15 GB disk.  
**Prerequisites:** frozen vision models, reconstruction losses, and sparse representations.

### Research question

Can deliberate selection of probing images improve SAE concept coverage and intervention usefulness at a fixed data and sparsity budget?

### Start with these resources

- **Required:** [Inside the Visual Mind / ViSAE](https://arxiv.org/abs/2606.06664).
- **Required:** [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) — SAE foundation.
- **Tutorial/reference:** [ViSAE training, concept reading, and tracing examples](https://github.com/deep-real/ViSAE).
- **Data/model:** S4, including the released probing images, vocabulary, and CLIP checkpoint.

### Data and model

Use the 6,000-image training pool from S4. Select equal-budget subsets of 1,000 and 3,000 images. Do not select using development/test activations. Cache two CLS residual layers and train a separate small SAE for each layer, using expansion factor 4.

Reduce the concept vocabulary to a documented 200–500-term subset using training data. A feasible selector balances rare candidate concepts and visual diversity using frozen CLIP similarities; treat these similarities as selection proxies.

**Must implement yourselves:** activation caching; a ReLU SAE with sparsity/reconstruction training; your image selector; and a coverage evaluator. Reference code is for comparison and understanding.

### Original component and system

Develop a selector that balances concept diversity against redundant images. Compare random selection and feature-space diversity selection at equal image counts, training updates, and average active-feature counts. The system should output an image manifest, trained dictionaries, feature examples, and coverage statistics.

Keep evaluation independent of the selection proxy. Use a small synthetic set with known color/shape/texture labels and a blinded manual audit of 20–30 selected features on held-out real images. The same CLIP score used for selection and labeling cannot be the sole evidence of semantic correctness.

### Experiments and evaluation

Report reconstruction error, average active features, dead-feature fraction, independently verified concept precision/coverage, and coverage per training image. Include reconstruction-only substitution versus feature editing to separate reconstruction damage from intervention effects.

**Ablations:** remove rarity weighting; remove diversity weighting. Train three SAE seeds at one primary budget and use the second budget for a smaller scaling check. Report uncertainty over seeds and audit disagreements.

### Minimum scope, suggested milestones, and fallback

Minimum: two layers, three selection strategies, one SAE architecture, and independent semantic checks.

**W1:** download a subset and verify layer hooks. **W2:** cache activations and implement the SAE. **W3:** implement selection and feature labeling. **W4:** compare matched-budget dictionaries. **W5:** audit concepts and interventions. **W6:** deliver a feature browser/table and analysis.

Fallback: one layer and a 3,000-image pool; retain all selectors and independent checks. Extension: compare a TopK SAE with the ReLU SAE after matching sparsity.

**First start:** extract one CLS vector per image from 100 images and overfit a tiny SAE to verify reconstruction dimensions and gradients.

---

## P08. Stable SAE features versus causally useful features

**Area:** sparse autoencoders and language-model representations  
**Difficulty:** D2 — feature matching and intervention controls.  
**Compute:** C2; 12–16 GB GPU; estimated 12–35 GPU-hours; approximately 4–8 GB disk.  
**Prerequisites:** transformer hooks, sparse training, and similarity matching.

### Research question

Are features that recur across independently trained SAEs more useful for explaining model behavior than unstable features?

### Start with these resources

- **Required:** [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600).
- **Behavioral context:** [Interpretability in the Wild: An IOI Circuit in GPT-2 Small](https://arxiv.org/abs/2211.00593).
- **Reference training examples:** [dictionary_learning](https://github.com/saprmarks/dictionary_learning).
- **Tutorial/data/model:** S3 and TransformerLens documentation.

### Data and model

Freeze GPT-2 Small and cache residual outputs at one middle layer for 200,000–500,000 WikiText-2 training tokens in windows of at most 128 tokens. Use official validation/test partitions for SAE selection/evaluation.

Train three ReLU SAEs with identical expansion factor 4, data, and target sparsity but different initializations. As a task-specific causal evaluation, generate 500 development and 500 test IOI prompts with disjoint names/templates after verifying GPT-2 competence.

### System and required original work

**Must implement yourselves:** SAE training, a cross-seed feature matcher, an activation-editing evaluator, and a consensus feature-selection rule.

Match normalized decoder directions using cosine similarity and held-out activation agreement. Address one-to-many feature splitting instead of assuming a one-to-one match always exists. Select recurring features without accessing final-test intervention scores.

For editing feature k, preserve the SAE reconstruction residual:

    edited activation = original activation + decoder × (edited code − original code)

Compare this with reconstruct-and-replace. An unchanged code must leave the model output unchanged.

### Experiments and evaluation

Baselines: individual neurons, PCA directions with matched selection budgets, and feature selection by frequency alone. Compare stable and unstable features matched by activation frequency, magnitude, and count.

Report reconstruction/sparsity, cross-seed match quality, intervention changes in IOI logit difference, language-model loss change on held-out WikiText, and the consensus subset's effects. Decoder similarity alone does not establish semantic identity or causal importance.

**Ablations:** remove activation agreement from matching; remove consensus weighting. Use random directions and random matched-size feature sets as controls. Test more than one edit magnitude.

### Minimum scope, suggested milestones, and fallback

Minimum: one layer, three SAE seeds, one matcher, one consensus rule, and task plus general-text intervention checks.

**W1:** cache a pilot and verify IOI competence. **W2:** train SAEs. **W3:** implement matching/consensus. **W4:** evaluate interventions. **W5:** run matched-feature controls and residual-preservation checks. **W6:** present matched features and causal effects.

Fallback: 100,000 tokens and expansion factor 2, retaining three seeds and both evaluation types. Extension: add a second layer or study feature splitting explicitly.

**First start:** confirm that encoding/decoding with an unchanged-code residual-preserving edit produces exactly the original activation.

---

## P09. An economical Jacobian lens

**Area:** Jacobian lens and mechanistic interpretation  
**Difficulty:** D3 — Jacobian operators, numerical validation, and approximation tradeoffs.  
**Compute:** C3; 16–24 GB GPU; estimated 15–50 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** Jacobians, Jacobian-vector/vector-Jacobian products, and residual-stream hooks.

### Research question

How much computation and memory can be saved when estimating a Jacobian lens while preserving its directional accuracy and useful readouts?

### Start with these resources

- **Required:** [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/).
- **Reference/tutorial:** [Jacobian-lens repository and walkthrough](https://github.com/anthropics/jacobian-lens).
- **Baseline context:** [Eliciting Latent Predictions with the Tuned Lens](https://arxiv.org/abs/2303.08112).
- **Numerical tutorial:** [PyTorch Jacobians, JVPs, and VJPs](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html).
- **Model:** [Qwen2.5-0.5B base checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B); corpus: [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext).

### Data and estimator

Choose three intermediate blocks. Fit on 100 training sequences of at most 64 tokens; reserve separate official validation/test windows. Read the reference estimator's conventions: it aggregates causal derivatives over valid source positions and current/future target positions. A same-position derivative is a different estimator and must be labeled as an ablation.

Start with an exact dense Jacobian on a tiny width-32, two-block model. For the real model, represent the averaged transport as an operator and use randomized low-rank estimation or streamed projection with ranks 8, 16, and 32. Preserve the model's final normalization and unembedding conventions.

### System and required original work

**Must implement yourselves:** the correctly masked averaged Jacobian operator; an economical estimator; finite-difference checks; and a budget-based rank or sample-count selection rule. The public fitter may supply a comparison on a small configuration, not your core implementation.

The system fits/saves a lens and displays token readouts across layers. Include a logit-lens baseline and a dense reference on the tiny model. On the real model, compare predicted directional changes with directly computed derivatives on held-out directions; do not require a full dense reference fit.

### Experiments and evaluation

Measure derivative approximation error, small-intervention prediction error, readout rank agreement where a reference is available, concept-token retrieval on controlled prompts, peak memory, and wall time.

Use public repository evaluation prompts as development examples and create held-out variants with disjoint entities/templates. Do not equate next-token agreement with concept interpretation. Include random transports and shuffled readout labels as negative controls.

**Ablations:** rank, prompt count, and source/target aggregation choice. Test finite differences at several small edit scales to identify the local linear regime.

### Minimum scope, suggested milestones, and fallback

Minimum: tiny-model correctness, three real-model layers, three budgets, and an original budget-selection rule.

**W1:** reproduce a tiny exact Jacobian. **W2:** verify masked operator calculations. **W3:** implement approximation/rank selection. **W4:** fit and evaluate real-model lenses. **W5:** study numerical controls and costs. **W6:** deliver saved lenses and a readout viewer/table.

Fallback: one layer, 50 prompts, and ranks 8/16; retain exact correctness and held-out directional tests. Extension: compare with a separately trained tuned lens.

**First start:** compare a width-32 Jacobian-vector product with central finite differences before loading the larger model.

---

## P10. Does the J-lens fitting corpus change the interpretation?

**Area:** Jacobian lens under domain change  
**Difficulty:** D3 — estimator consistency, domain control, and causal validation.  
**Compute:** C3; 16–24 GB GPU; estimated 20–50 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** P09-style Jacobian concepts; this project is independent and needs its own small correctness check.

### Research question

Which lens readouts transfer between fitting corpora, and can corpus weighting improve interpretation on a new prompt domain?

### Start with these resources

- **Required:** [Global workspace/J-lens paper](https://transformer-circuits.pub/2026/workspace/).
- **Reference and public prompt sets:** [Jacobian-lens repository](https://github.com/anthropics/jacobian-lens).
- **Baseline context:** [Tuned Lens](https://arxiv.org/abs/2303.08112).
- **Numerical tutorial:** [PyTorch Jacobian operations](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html).
- **Model/corpora:** [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), and [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

### Data and model

Freeze one model and two intermediate blocks. Fit separate lenses on 100 WikiText training windows and 100 TinyStories training windows, matching sequence length and token budgets. Reserve domain-specific validation and final-test windows before fitting.

Create controlled prompts with known entities or properties and explicit target tokens. Split by template/entity, checking tokenization. Include repository prompt sets for development only. Natural-corpus likelihood and controlled concept retrieval measure different outcomes.

### System and required original work

**Must implement yourselves:** the corpus-conditioned estimator, a corpus-weighted pooled estimator, the readout evaluator, and an intervention check. Standard hooks/autograd are allowed.

One feasible original method weights fitting examples by their similarity to a deployment domain using features from the frozen model. Choose the weighting rule on development data. Match compute budgets for separate, uniform-pooled, and weighted-pooled lenses; report the estimator's masking and aggregation conventions.

Produce a cross-domain evaluation matrix and a viewer comparing lens readouts for the same prompt. Do not replace the Jacobian estimator with a learned linear probe and call it a J-lens.

### Experiments and evaluation

Baselines: logit lens, each domain-specific lens, and uniform pooling. Measure controlled-token retrieval, cross-domain derivative/intervention prediction error, readout agreement, runtime, and sensitivity to sample count.

Validate selected directions through small positive/negative activation edits and compare actual output changes with lens predictions. Readout words alone do not establish what the model “thinks.”

**Ablations:** remove corpus weighting; replace semantic similarity with random weights; control for token frequency and length. Use repeated fitting-sample seeds. Final-test prompts cannot determine corpus weights.

### Minimum scope, suggested milestones, and fallback

Minimum: one model, two layers, two corpora, three fitting strategies, and a cross-domain intervention analysis.

**W1:** test hooks and tiny numerical correctness. **W2:** prepare disjoint corpus samples. **W3:** implement conditioned/weighted fitting. **W4:** build the evaluation matrix. **W5:** run frequency controls and interventions. **W6:** package lens comparisons and conclusions.

Fallback: one layer, 50 sequences per corpus, and short contexts. Extension: add a synthetic code or arithmetic corpus with a clearly documented generator.

**First start:** compare logit-lens readouts on ten matched-length windows from each corpus and verify that every lens uses the same residual convention.

---

## P11. When is a discovered circuit faithful?

**Area:** circuit discovery and evaluation  
**Difficulty:** D3 — edge instrumentation, ablation definitions, and methodological interpretation.  
**Compute:** C3; 16–24 GB GPU; estimated 10–35 GPU-hours; approximately 2–5 GB disk. No model training, but gradients are required.  
**Prerequisites:** transformer computation graphs, activation patching, and autograd.

### Research question

How much does apparent circuit quality depend on the evaluation intervention, and can an intervention-aware selection rule produce more reliable circuits?

### Start with these resources

- **Required:** [Towards Automated Circuit Discovery](https://arxiv.org/abs/2304.14997).
- **Required:** [Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806).
- **Required:** [Transformer Circuit Faithfulness Metrics Are Not Robust](https://arxiv.org/abs/2407.08734).
- **Task context:** [How Does GPT-2 Compute Greater-Than?](https://arxiv.org/abs/2305.00586).
- **Tutorial/reference baselines:** [auto-circuit](https://github.com/UFO-101/auto-circuit), [EAP-IG](https://github.com/hannamw/EAP-IG), and S3/TransformerLens.

### Data, model, and circuit definition

Freeze GPT-2 Small. Use year-span greater-than prompts following the task paper or the released task loaders. Verify that candidate two-digit continuations have the intended tokenization and that the unmodified model shows the expected preference.

Prepare approximately 500 discovery, 500 development, and 1,000 final-test prompts with documented disjoint template/noun variants. A matched corrupted prompt changes the relevant number while preserving length and other content.

Define nodes as attention-head and MLP outputs and edges as their contributions through the residual stream to downstream computations. Use one consistent graph convention across methods. A graph restricted to selected layers must retain explicit boundary inputs and be reported as a partial-circuit study.

### System and required original work

**Must implement yourselves:** an edge-attribution-patching scorer, a matched-size circuit evaluator, and an intervention-aware selection rule. Libraries may provide hook/patch primitives and reference baselines.

Score edges on discovery prompts. Evaluate zero ablation, token-position mean ablation with means estimated from discovery data, and resample ablation from matched corrupted prompts. The rule may penalize variability across development interventions or maximize a declared worst-case performance criterion.

Select circuit size/threshold on development data. Evaluate the fixed selected circuit under all interventions on final-test prompts; do not rediscover it separately on test data. EAP-IG is an optional additional discovery baseline.

### Experiments and evaluation

Baselines: ordinary EAP top-k circuits, random size-matched circuits, and a simple component-ablation ranking.

Plot faithfulness versus retained-edge count. Report unnormalized task scores as well as normalized retention; omit or flag normalized scores when the full-versus-ablated denominator is near zero. Measure held-out task preference, score variability, membership overlap, and runtime. Circuit overlap is supplementary evidence, not the primary quality metric.

**Ablations:** remove the stability penalty; remove one intervention from selection. Inspect examples where interventions disagree and explain which computational question each asks. Universal invariance across all ablations is not required.

### Minimum scope, suggested milestones, and fallback

Minimum: one task/model, one discovery method, three ablation definitions, and one original selection rule.

**W1:** establish task competence and identity-patch correctness. **W2:** implement edge scoring. **W3:** build the evaluator and selection rule. **W4:** evaluate fixed circuits. **W5:** study disagreement and template transfer. **W6:** deliver graphs, curves, and a limitations analysis.

Fallback: use a fixed candidate graph over four late blocks with explicit boundaries and smaller batches; narrow the claim accordingly. Extension: compare EAP with EAP-IG or add a second task.

**First start:** patch no edges and then all candidate edges on ten prompt pairs; verify both boundary cases before circuit discovery.

---

## P12. Robustness of induction-head circuits

**Area:** circuit robustness and pattern completion  
**Difficulty:** D2 — approachable inference experiments with essential causal controls.  
**Compute:** C1; 8–12 GB GPU; estimated 3–12 GPU-hours; approximately 1–3 GB disk.  
**Prerequisites:** attention heads, tokenization, and activation hooks.

### Research question

Do induction mechanisms continue to copy the intended continuation under distractors or competing patterns, and can robust head selection improve transfer across these conditions?

### Start with these resources

- **Required:** [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
- **Required/tutorial:** [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
- **Model/tools:** S3 and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
- **Data:** your own reproducible token-sequence generator; no language-model training is needed.

### Data and model

Freeze GPT-2 Small. Generate sequences containing a prior pair A B, distractor tokens, and a final A; B is the intended copied token. Ensure A and B are distinct, document the vocabulary, and exclude unwanted accidental matches.

Use 500 discovery, 500 development, and 1,000 test sequences with separate seeds and token-pair lists. Core perturbations are longer distractor gaps, competing A C pairs, and imperfect repeated contexts. Use lengths 32/64/128. For competing patterns, specify whether the desired rule is most-recent or original association; do not silently treat an ambiguous target as an error.

### System and required original work

**Must implement yourselves:** the generator, induction attention score, individual/joint head ablations, clean-to-corrupt activation patching evaluator, and a perturbation-aware selection policy.

Select up to eight candidate heads using discovery attention and output-contribution evidence. Evaluate joint as well as individual removal to detect redundancy. Restrict detailed interaction analysis to these candidates.

Your original policy selects a small head set using development robustness, compared with selecting by clean attention scores alone. The system should show the sequence, attention evidence, causal effect, selected set, and performance by perturbation.

### Experiments and evaluation

Baselines: whole-model copying, clean-score top-k heads, random matched-size sets, and a simple greedy causal ranking.

Report target probability, target-versus-distractor logit difference, top-1 copy accuracy, ablation effect, intervention recovery, and generalization to held-out perturbation strengths. Measure the mismatch between high induction attention and actual causal contribution.

**Ablations:** remove perturbation-aware selection; select heads independently instead of considering redundancy. Keep head sets fixed at test time. Claimed head subclasses are possible findings, not required outcomes.

Token rarity may be studied using a stated public-corpus frequency proxy; do not claim known rarity in GPT-2's unavailable training distribution.

### Minimum scope, suggested milestones, and fallback

Minimum: one pretrained model, three perturbation families, eight or fewer detailed candidate heads, and an original selection policy.

**W1:** confirm copying and generate clean/corrupt pairs. **W2:** identify candidates and implement patching. **W3:** build robust selection. **W4:** evaluate held-out conditions. **W5:** test joint effects and ambiguity controls. **W6:** deliver a robustness profile and demonstration.

Fallback: lengths 32/64 and 500 final-test sequences; retain all perturbation types. If GPT-2 fails the clean pilot, use a documented small pretrained attention-only model with demonstrated induction behavior. Extension: investigate frequency-proxy effects or head specialization.

**First start:** compare target-token probability for ten sequences with and without their preceding A B pair.

---

## P13. Vision concept circuits under background shift

**Area:** vision circuits and shortcut reliance  
**Difficulty:** D3 — small SAE training, cross-layer interventions, and reconstruction controls.  
**Compute:** C3; 16–24 GB GPU; estimated 15–45 GPU-hours; approximately 8–15 GB disk.  
**Prerequisites:** ViT residuals, SAEs, classification heads, and activation editing.

### Research question

Which concept pathways remain useful when foreground/background correlations change, and can a restrained circuit-based edit reduce shortcut reliance?

### Start with these resources

- **Required:** [Inside the Visual Mind / ViSAE](https://arxiv.org/abs/2606.06664).
- **Additional rationale context:** [Beyond Accuracy](https://arxiv.org/abs/2411.00132).
- **Tutorial/reference:** [ViSAE tracing examples](https://github.com/deep-real/ViSAE).
- **Data:** S4 plus [Waterbirds download and official split metadata](https://github.com/kohpangwei/group_DRO#waterbirds).
- **Model:** S4/CLIP ViT-B/32; train your own task head and small SAEs.

### Data and model

Use official Waterbirds training/development/test membership. Train a linear bird-type classifier on frozen CLIP image embeddings using the biased training partition. Group labels identify bird/background groups for reporting; declare any development-time use.

Train SAEs at two block outputs on a disjoint ViSAE training subset. Limit detailed graph tracing to at most 16 source features, 16 destination features, and 100 discovery images. Label selected features using the public vocabulary and verify a small sample independently.

### System and required original work

**Must implement yourselves:** small SAE training; residual-preserving feature edits; a cross-layer intervention graph; and a budgeted feature/pathway edit rule.

For a source feature, perturb its code while retaining the reconstruction residual, rerun the remaining vision blocks, and measure downstream feature and classifier changes. Treat graph edges as intervention effects under this operator, not proof of autonomous semantic mechanisms.

Use training/development images to choose an edit strength and at most four edited features. The deployment rule must not read the test bird/background labels. A useful original rule penalizes edits that harm foreground-sensitive predictions while targeting background-sensitive pathways.

### Experiments and evaluation

Baselines: unedited classifier, random matched-size edits, direct concept suppression without pathway information, and zero-code versus residual-preserving edits.

Report mean and worst-group accuracy, group sample counts, calibration, classifier-logit changes, edge stability under edit scales, and reconstruction-only substitution effects. Evaluate on the full official test split for classification; trace a fixed 100-image test subset for detailed graph comparisons.

**Ablations:** remove the foreground-preservation term; select features without graph information. An unchanged-code edit and pure reconstruction substitution are required controls. Test whether effects persist across repeated SAE seeds or report sensitivity for a smaller subset.

### Minimum scope, suggested milestones, and fallback

Minimum: two layers, a limited graph, one original restrained-edit rule, and worst-group evaluation.

**W1:** train the task head and verify hooks. **W2:** train/cache small SAEs. **W3:** build the intervention graph. **W4:** evaluate selected edits. **W5:** analyze reconstruction artifacts and group failures. **W6:** deliver concept graphs and before/after predictions.

Fallback: 3,000 probing images and eight features per layer, keeping two layers and all controls. If the classifier has little measurable shortcut reliance, report the null result rather than creating a test-dependent failure. Extension: compare a second frozen vision encoder.

**First start:** apply an unchanged-code edit and verify that logits remain unchanged; then compare a background-related feature edit with a pure reconstruction replacement.

---

## P14. Preventing negative transfer from pseudo-labels

**Area:** unsupervised domain adaptation  
**Difficulty:** D2 — confidence control and strict source/target data separation.  
**Compute:** C1; 8–12 GB GPU; estimated 6–18 GPU-hours; approximately 1–3 GB disk.  
**Prerequisites:** supervised CNN training and pseudo-labeling.

### Research question

Can an unlabeled reliability signal reduce harmful target adaptation better than confidence thresholding alone?

### Start with these resources

- **Required:** [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) — adaptation setting and baseline context.
- **Required:** [FixMatch](https://arxiv.org/abs/2001.07685) — confidence and consistency ideas; its original setting is semi-supervised learning.
- **Data/tutorials:** [MNIST loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) and [USPS loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html).
- **Model:** a student-trained two- or three-block digit CNN; no external checkpoint required.

### Data and adaptation setting

Use MNIST training images as the labeled source, reserving 10,000 for source development. Use the official USPS training partition as unlabeled target training data; the USPS test partition is final evaluation only. Resize both to 28×28 grayscale and document normalization.

Tune policies using source development images under synthetic appearance changes. Target training labels are hidden from the adapter. They may be opened only in a post hoc analysis after settings are frozen. Never use target test performance for checkpoint selection.

Run the ordinary transfer and two predefined stronger target-appearance conditions, such as added noise and contrast change, applied consistently to target train/test images. Use the same source initializations for every adaptation method.

### System and required original work

**Must implement yourselves:** source training; pseudo-label loss and selection; a reliability score; and an adaptation pipeline that logs acceptance and class coverage.

Develop a score combining confidence with prediction consistency across weak, digit-preserving augmentations. Use class-aware weighting or a source-agreement safeguard to avoid reinforcing a few predicted classes. Maintain labeled source training batches during adaptation; this is UDA, not source-free TTA.

### Experiments and evaluation

Baselines: source-only classifier, all pseudo-labels, and confidence-threshold pseudo-labeling. A DANN baseline is optional.

Report target accuracy and macro-F1, source accuracy retention, negative-transfer change relative to source-only, accepted-label fraction, and pseudo-label precision in the post hoc evaluator. Compare equal adaptation steps and source/target batch counts. Use three source-training seeds.

**Ablations:** remove consistency; remove class balancing or the source safeguard. Test an intentionally overpermissive threshold as a stress control, but do not tune target conditions after seeing final results.

### Minimum scope, suggested milestones, and fallback

Minimum: one transfer direction, three target-appearance conditions, and one original reliability policy.

**W1:** inspect partitions and train a source pilot. **W2:** implement pseudo-label baselines. **W3:** build the reliability policy. **W4:** run frozen adaptation settings. **W5:** analyze negative transfer and acceptance errors. **W6:** deliver logs, examples, and report.

Fallback: smaller CNN and 20,000 source training examples, retaining the full target split. Extension: reverse USPS→MNIST or add DANN.

**First start:** evaluate a source-trained model on USPS without adaptation and inspect 20 confident errors.

---

## P15. Domain generalization with an imperfect domain graph

**Area:** domain generalization and topology-aware robust optimization  
**Difficulty:** D2 — graph construction and a small minimax optimizer.  
**Compute:** C1; 8–12 GB GPU; estimated 6–20 GPU-hours; approximately 1–3 GB disk. The synthetic core may run on CPU after a pilot.  
**Prerequisites:** graph similarity, group risks, and constrained optimization.

### Research question

How accurate must a source-domain graph be to help unseen-domain generalization, and can uncertainty-aware priors reduce damage from incorrect edges?

### Start with these resources

- **Required:** [Topology-aware Robust Optimization](https://arxiv.org/abs/2307.13943).
- **Evaluation context:** [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434).
- **Data/reference:** [TRO repository and DG-15/DG-60 data](https://github.com/deep-real/TRO).
- **Baselines/tutorials:** [group_DRO](https://github.com/kohpangwei/group_DRO) and [DomainBed](https://github.com/facebookresearch/DomainBed).
- **Access note:** TRO's larger benchmark integrations are not prerequisites; begin with the available synthetic data.

### Data, graph, and model

Use DG-15 with a documented source/held-out-domain split, preserving the dataset's domain identities. Make development groups from source data or leave one source domain out for model selection. Declare separately whether the final held-out groups represent interpolation or extrapolation.

If the released data layout is inconvenient, generate rotated-MNIST domains at fixed angles. Split original MNIST image identities first, then rotate, so an image cannot appear in both source training and final testing.

Train a small MLP for DG-15 or a small CNN for rotated MNIST. Construct graphs from source-domain features only. Known source-angle adjacency is a permitted source prior; target examples or target labels cannot determine graph edges.

### System and required original work

**Must implement yourselves:** domain-risk computation; source graph construction; a TRO-style optimizer; graph perturbations; and an uncertainty-aware prior.

One manageable objective is:

    minimize over model parameters:
        maximize over nonnegative group weights q summing to 1:
            sum(q × group risk) − lambda × squared_distance(q, graph_prior)

Implement alternating model/weight updates and check simplex feasibility. A feasible original prior bootstraps source-domain distances, downweights unstable edges, and shrinks uncertain graph-centrality weights toward uniform weights.

### Experiments and evaluation

Baselines: ERM, Group DRO, and an uncorrected topology prior. Compare missing, added, and rewired edges at predefined rates, including an edge-free/uniform case.

Report held-out mean/worst-domain accuracy, source risk, graph stability, q distributions, and performance versus graph corruption. Use three training seeds and several graph-perturbation seeds without treating all combinations as independent model runs.

**Ablations:** remove uncertainty shrinkage; use centroid distance rather than the chosen graph distance. Keep model capacity and optimization steps matched. A negative result on a poor graph is informative.

### Minimum scope, suggested milestones, and fallback

Minimum: one synthetic benchmark, one original graph-uncertainty rule, three baselines, and graph-misspecification analysis.

**W1:** establish domains and an ERM baseline. **W2:** implement group optimization and graphs. **W3:** build uncertainty priors. **W4:** run graph perturbations. **W5:** compare interpolation/extrapolation and ablations. **W6:** deliver graph/risk visualizations.

Fallback: fewer source domains and a two-layer MLP; preserve source-only graph construction. Extension: DG-60 or a compact image-domain benchmark.

**First start:** implement Group DRO on a few groups and verify that every update keeps q nonnegative and normalized.

---

## P16. Generating useful training environments

**Area:** domain generation and domain generalization  
**Difficulty:** D2 — controllable augmentation and label-preservation analysis.  
**Compute:** C2; 12–16 GB GPU; estimated 15–40 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** image augmentation, classifier training, and experimental controls.

### Research question

Which generated training environments improve unseen-shift robustness, and when does increasing augmentation diversity damage label semantics?

### Start with these resources

- **Required:** [Learning to Learn Single Domain Generalization](https://arxiv.org/abs/2003.13216).
- **Required:** [AugMix](https://arxiv.org/abs/1912.02781).
- **Reference/tutorials:** [M-ADA](https://github.com/joffery/M-ADA), [AugMix implementation](https://github.com/google-research/augmix), and [DomainBed](https://github.com/facebookresearch/DomainBed).
- **Data/model:** S1; use a student-trained source classifier, not a foundation-model training run.

### Data and generated environments

Train on CIFAR-10 source training images. Create three controllable environment families: color/contrast shifts, low-frequency amplitude changes, and mild noise/blur. Preserve spatial structure when modifying Fourier amplitudes; document magnitude bounds.

Select parameters on development images and a source-only leave-one-generated-family-out evaluation. Keep CIFAR-10-C test images and at least one test corruption family out of generation-policy tuning.

### System and required original work

**Must implement yourselves:** a parameterized environment generator; a severity/diversity sampler; the training pipeline; and a label-preservation diagnostic.

Your original component should choose environments using a declared tradeoff between challenge, diversity, and semantic preservation. A feasible sampler updates its training distribution using source-development performance and a transform-strength cap. Keep its information budget explicit.

The system takes a source image, returns an environment variant and its parameters, and logs how often each environment is used. A generative diffusion model is unnecessary.

### Experiments and evaluation

Baselines: ERM, fixed random augmentation, and AugMix or a documented lightweight approximation. Compare equal training epochs and images processed.

Report clean accuracy, per-corruption accuracy, worst-condition performance, and robustness versus generation strength. Manually audit 100 fixed development variants using an explicit label-preservation rubric, and report ambiguous/damaged examples. A model's confidence alone cannot certify unchanged semantics.

**Ablations:** remove diversity weighting; remove the preservation cap. Include a deliberately excessive transform condition to reveal failure rather than assuming more diversity is always better. Use three training seeds at one selected configuration.

### Minimum scope, suggested milestones, and fallback

Minimum: three environment families, one original sampler, matched-budget baselines, and a preservation audit.

**W1:** implement/inspect transform grids. **W2:** train ERM and fixed augmentation. **W3:** build the sampler. **W4:** evaluate held-out corruptions. **W5:** audit labels and ablate selection. **W6:** deliver generated examples and robustness curves.

Fallback: 20,000 source training images, smaller CNN, and two severity levels. Extension: a second source dataset or a stronger domain generator with a clear resource budget.

**First start:** render a 5×5 grid of transform strengths for ten images and identify parameter ranges where humans can still recognize the original class.

---

## P17. Recovering from changing test environments

**Area:** source-free test-time adaptation  
**Difficulty:** D2 — stateful evaluation, drift detection, and reliable resets.  
**Compute:** C2; 12–16 GB GPU; estimated 10–30 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** batch normalization, entropy minimization, and streaming evaluation.

### Research question

When should an adapting classifier keep its recent state, partially reset, or return to its original checkpoint as the environment changes?

### Start with these resources

- **Required:** [Tent](https://arxiv.org/abs/2006.10726).
- **Required:** [Continual Test-Time Domain Adaptation / CoTTA](https://arxiv.org/abs/2203.13591).
- **Reference/tutorials:** [Tent example](https://github.com/DequanWang/tent) and [CoTTA](https://github.com/qinenergy/cotta). Tent's repository is archived but remains useful as a reference.
- **Data/model:** S1; train and save your own source CNN with batch normalization.

### Data and deployment protocol

Construct streams such as clean→noise→blur→clean and gradual severity ramps. Use class-balanced segments and disjoint image identities within a stream. Compare at least three fixed segment orders and lengths.

Tune detection thresholds on development-image streams with generated corruptions. During final deployment, permit only the source checkpoint, saved training statistics, and current/past unlabeled inputs. Store labels in a separate evaluator.

Use predict-before-update as the primary protocol, updating only BN affine parameters with one entropy step per eligible batch. Start with batch size 32. Declare separately any adapt-then-predict result.

### System and required original work

**Must implement yourselves:** Tent-style entropy/parameter selection, the stream runner, a drift statistic, and a reset/recovery policy. A feasible detector tracks changes in prediction entropy and intermediate feature moments.

Compare full reset, selective BN reset, and no reset. When resetting, handle optimizer moments and normalization buffers as well as model parameters. Keep batch construction and update budgets identical across methods.

### Experiments and evaluation

Baselines: source-only, test-time normalization, uninterrupted Tent, and periodic reset Tent.

Report pre-update accuracy over time, cumulative error, post-change recovery length, clean-return accuracy, false reset frequency, skipped updates, and runtime. Use matched source-training seeds and stream identities.

**Ablations:** remove drift detection; reset parameters but not optimizer state; use only entropy or only feature moments. Separate the effects of class composition from corruption changes.

### Minimum scope, suggested milestones, and fallback

Minimum: one model, two stream patterns, three orders, and an original recovery controller.

**W1:** train source and verify a Tent pilot. **W2:** implement leakage-free streams and resets. **W3:** build the controller. **W4:** evaluate frozen thresholds. **W5:** analyze recovery and state ablations. **W6:** deliver accuracy timelines and a replay demonstration.

Fallback: two corruption families and shorter streams; retain clean return and state controls. Extension: small batches or a teacher-based continual adapter.

**First start:** adapt for several batches, restore the checkpoint/optimizer/buffers, and verify that predictions match those before adaptation.

---

## P18. Test-time adaptation under class imbalance

**Area:** test-time adaptation under label and covariate shifts  
**Difficulty:** D2 — prior estimation and class-sensitive streaming analysis.  
**Compute:** C2; 12–16 GB GPU; estimated 10–30 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** classification calibration, entropy losses, and batch construction.

### Research question

When does entropy minimization reinforce majority-class predictions, and can an unlabeled prior-aware update reduce this failure?

### Start with these resources

- **Required:** [Tent](https://arxiv.org/abs/2006.10726).
- **Required:** [Detecting and Correcting for Label Shift with Black Box Predictors](https://arxiv.org/abs/1802.03916).
- **Additional context:** [Efficient Test-Time Model Adaptation without Forgetting / EATA](https://arxiv.org/abs/2204.02610).
- **Tutorial/baselines:** [Tent](https://github.com/DequanWang/tent), [EATA](https://github.com/mr-eggplant/EATA), and S1.

### Data and permitted information

Train an S1 source model. The evaluator constructs class-balanced, moderately imbalanced, and strongly imbalanced CIFAR-10-C streams, plus a class-burst stream. Use fixed manifests and batch sizes 4, 16, and 32.

Labels may determine stream construction and final scoring but never enter the adapter. Before deployment, estimate calibration/confusion statistics on source development data and save them. This is a source-free deployment method with declared saved source statistics.

### System and required original work

**Must implement yourselves:** Tent-style updates, an unlabeled moving-window prior estimate, and a prior-aware filtering/weighting rule.

One feasible approach regularizes a source-confusion-based prior estimate and caps per-class update weights. A simpler prediction-count estimator is a baseline. Under simultaneous corruption and label shift, the label-shift assumption can fail; investigate this instead of claiming the prior is known.

Use predict-before-update. Tune smoothing and clipping on generated development streams. Do not give the algorithm the evaluator's true target proportions.

### Experiments and evaluation

Baselines: source-only, normalization-only, standard Tent, and confidence-filtered Tent. A true-prior method may be a clearly privileged diagnostic upper bound.

Report overall accuracy, macro-F1/balanced accuracy, minority-class recall, predicted-class concentration, prior-estimation error in the evaluator, and runtime. Include label-only shift without corruption to distinguish estimation failure from corrupted representations.

**Ablations:** remove weighting; remove confidence filtering; remove regularization of the prior estimate. Use at least three stream sampling seeds and matched source model initializations.

### Minimum scope, suggested milestones, and fallback

Minimum: one source model, two corruption families, three class-mixture settings, and one original prior-aware rule.

**W1:** train source and prepare stream manifests. **W2:** implement baseline adapters and prior estimation. **W3:** build the update rule. **W4:** evaluate frozen settings. **W5:** analyze minority failures and violated assumptions. **W6:** deliver class-wise timelines and report.

Fallback: batch sizes 16/32 and one corrupted family plus label-only shift. Extension: distinguish persistent prior change from transient class bursts.

**First start:** run source-only and Tent on balanced versus class-burst streams and inspect predicted-class counts before adding prior correction.

---

## P19. Choosing a useful self-supervised task for TTT

**Area:** test-time training with self-supervision  
**Difficulty:** D2 — joint training and fair comparisons between auxiliary tasks.  
**Compute:** C2; 12–16 GB GPU; estimated 20–50 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** shared encoders, auxiliary losses, and adaptation loops.

### Research question

Which self-supervised objective helps classification under which distribution shift, and can a training-time alignment rule make the auxiliary task more useful?

### Start with these resources

- **Required:** [Test-Time Training with Self-Supervision](https://proceedings.mlr.press/v119/sun20b.html).
- **Required:** [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728).
- **Author tutorial/reference:** [TTT CIFAR code](https://github.com/yueatsprograms/ttt_cifar_release).
- **Data/model:** S1; train your own shared encoder, task head, and auxiliary heads.

### Training and deployment

Compare rotation prediction (0/90/180/270 degrees) with masked low-resolution image reconstruction. Use the same small encoder and supervised training split; keep the classification head and total training budget matched. Reconstruction needs a small decoder; report its parameter overhead.

Train with classification plus auxiliary loss. At deployment, use unlabeled test images to update a declared subset of encoder parameters, keeping the classification head fixed. Start with one update per batch, batch size 32, and predict-before-update. Compare continuous adaptation with episodic restoration.

### System and required original work

**Must implement yourselves:** both auxiliary objectives and transformations; joint training; test-time update/restoration; and a training-time auxiliary weighting rule.

A feasible original rule uses supervised/auxiliary gradient agreement on source training batches to adjust auxiliary weight or select update layers. The supervised gradient is allowed during source training only. Freeze the rule's deployment decisions before testing; target labels never determine an update.

### Experiments and evaluation

Baselines: supervised-only source model, each jointly trained model without adaptation, fixed-weight rotation TTT, and fixed-weight reconstruction TTT. Separating joint-training benefits from test-time-update benefits is required.

Report clean/per-corruption accuracy, adaptation gains relative to each method's own non-adapted checkpoint, auxiliary loss, source-development gradient agreement, forgetting, and runtime. Use three seeds for the main task comparison; limit the secondary hyperparameter grid.

**Ablations:** remove adaptive weighting; change which encoder layers update; compare matched adaptation steps. Use a random-label auxiliary task or an incompatible transformation as a small negative control.

### Minimum scope, suggested milestones, and fallback

Minimum: two auxiliary tasks, one original training-time rule, and before/after adaptation comparisons under two corruption families.

**W1:** implement transformations and source pilot. **W2:** train fixed-weight baselines. **W3:** implement alignment weighting and TTT loops. **W4:** evaluate frozen models. **W5:** compare objectives, layers, and controls. **W6:** deliver adaptation curves and report.

Fallback: 20,000 source training images and a low-resolution reconstruction target; preserve both tasks and the matched comparisons. Extension: a third auxiliary objective or a second source dataset.

**First start:** overfit 100 source images on both classification and auxiliary losses, then verify that test-time updates leave the task head unchanged.

---

## P20. Deciding whether to accept a test-time training update

**Area:** selective test-time training  
**Difficulty:** D2 — transactional updates and unlabeled risk signals.  
**Compute:** C2; 12–16 GB GPU; estimated 15–40 GPU-hours; approximately 3–6 GB disk.  
**Prerequisites:** self-supervised joint training, optimizer state, and uncertainty evaluation.

### Research question

Can an unlabeled acceptance rule reject harmful self-supervised updates while retaining useful adaptation?

### Start with these resources

- **Required:** [Test-Time Training with Self-Supervision](https://proceedings.mlr.press/v119/sun20b.html).
- **Required/context:** [Efficient Test-Time Model Adaptation without Forgetting](https://arxiv.org/abs/2204.02610) — selective-update ideas in a different adaptation method.
- **Tutorial/references:** [TTT code](https://github.com/yueatsprograms/ttt_cifar_release) and [EATA](https://github.com/mr-eggplant/EATA).
- **Data/model:** S1; train a source classifier with a rotation auxiliary head. This project does not depend on P19.

### Data and deployment protocol

Prepare clean→corrupted→clean streams and stationary corruption streams with disjoint image identities within each stream. Tune the gate on source-development images under generated shifts, reserving at least one corruption family for final testing.

Permit the original checkpoint, current model state, and unlabeled inputs at deployment. Use no source replay examples and no target labels in the gate. Predict-before-update is the primary protocol.

### System and required original work

**Must implement yourselves:** joint classification/rotation training; candidate self-supervised updates; an unlabeled acceptance score; and complete rollback of rejected updates.

One feasible score combines auxiliary-loss improvement, prediction consistency under mild augmentations, and divergence from a frozen source model. Calibrate thresholds on development data, then freeze them. An auxiliary-loss decrease alone is a baseline and does not establish classification improvement.

Snapshot parameters, relevant buffers, and optimizer state before each candidate update. Restore all of them on rejection. The system logs the score, decision, and subsequent predictions so that rejection can be audited.

### Experiments and evaluation

Baselines: source-only, accept-all TTT, auxiliary-improvement-only acceptance, and a random gate matched by acceptance rate.

Report pre-update cumulative error, clean-return accuracy, acceptance rate, runtime, and evaluator-only accepted-harmful/rejected-beneficial rates. To classify an update as harmful, the evaluator may compare with/without-update predictions on held-out future examples. These diagnostic labels cannot feed back into the live gate.

**Ablations:** remove consistency or source divergence from the score; restore weights without optimizer state as a failure control. Compare matched accepted-update budgets and three source-training seeds.

### Minimum scope, suggested milestones, and fallback

Minimum: one auxiliary task, one original unlabeled gate, two stream patterns, and rollback correctness.

**W1:** train source/auxiliary model. **W2:** implement accept-all adaptation and rollback. **W3:** build/calibrate the gate. **W4:** evaluate frozen decisions. **W5:** analyze false rejections, harmful updates, and state controls. **W6:** demonstrate accepted and rejected updates.

Fallback: fewer stream examples and one candidate step; retain the matched-rate baseline and full rollback. Extension: adaptive update size or a reconstruction auxiliary task.

**First start:** reject a deliberately large candidate update and verify that predictions, buffers, and the next optimizer step match a run where the candidate update never occurred.

---

## Final reminders

- Every project must include its designated original component, a working system, and controlled evaluation. Reading or executing the linked code is preparation and baseline work.
- Keep the minimum scope complete before adding another dataset, model, or method.
- Report negative findings and resource limitations honestly. Strong causal or generalization claims require the controls described in the brief.
- Complete the first-day checklist and five-minute proposal talk in Week 1, and submit milestone reports through Canvas in Weeks 2–5.
- In Week 6, at the end of the semester, give a 10-minute final presentation plus five minutes of Q&A and submit the research report and reproducibility package.
- Student-proposed projects require an equivalent first-day checklist and advisor approval during the proposal talk.
- Follow the [general instructions](Course_Project_Instructions.md) for the required checkpoints, submissions, reproducibility, and the suggested assessment rubric.
