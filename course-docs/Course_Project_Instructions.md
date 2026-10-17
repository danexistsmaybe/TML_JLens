# CS 6501: Trustworthy Machine Learning — Course Project Instructions

**Audience:** graduate students in Computer Science with basic research experience  
**Team size:** one or two students  
**Duration:** six weeks  
**Project choices:** [20 candidate projects](Project_Descriptions.md) or a student-proposed project approved by the advisor during the Week 1 proposal talk.

**Navigation:** [Purpose](#1-purpose-and-expected-outcome) · [Choosing a project](#2-choosing-a-project) · [Milestones](#3-proposal-and-six-week-milestones) · [Implementation](#4-what-you-must-implement) · [Experiments](#5-experimental-design-and-data-access) · [Resources](#6-public-resources-and-responsible-interpretation) · [Submission](#7-final-submission) · [Rubric](#8-suggested-assessment-rubric) · [First day](#9-first-day-checklist)

## 1. Purpose and expected outcome

Choose a candidate project or propose your own project and build a reproducible system that investigates a focused question about trustworthy machine learning. Your work should connect a concrete implementation to evidence about explainability, internal mechanisms, or reliability under distribution shift.

A complete project includes:

1. A working baseline and a student-written implementation of the required core modules.
2. An original design choice, such as a selection policy, estimator, diagnostic, or adaptation controller.
3. Controlled experiments that distinguish your contribution from simpler alternatives.
4. A research conclusion supported by results, limitations, and failure cases.

An original contribution does not need to be publication-level novelty. Reproducing a paper's plots, changing a random seed, or executing a released repository alone is insufficient. A careful negative result can earn full credit. You are evaluated on the quality of the system, experiments, and reasoning, not on obtaining an accuracy improvement.

Each candidate description specifies a **minimum complete scope** and an **optional extension**. For a student-proposed project, establish the minimum scope with the advisor during the proposal talk. Finish the minimum scope before attempting extensions. Projects are independent; none requires access to another team's implementation or results.

## 2. Choosing a project

Read the catalog overview, then read the full briefs for two or three candidates. Consider your interests, experience with model internals, and computing access.

### Student-proposed projects

You may propose a project beyond the candidate list. It should address a course-relevant trustworthy ML question, require your own code and a working system, use accessible resources, and fit the six-week schedule.

Complete an equivalent first-day checklist for your project: identify relevant readings, obtain the core data/model, establish data splits, run a small feasibility experiment, estimate computing needs, identify baselines and the original component, and divide the work.

Present this evidence in the **five-minute Week 1 proposal talk**. A student-proposed project **requires the advisor's approval during that talk**, including agreement on its question, minimum scope, resources, and evaluation plan. All general implementation, reporting, final-submission, and assessment requirements also apply to approved student-proposed projects.

### Difficulty scale

| Level | Meaning | Useful preparation |
|---|---|---|
| D1 — Accessible | Familiar models and straightforward experiments; research design remains essential. | Python, basic ML evaluation, and data processing. |
| D2 — Intermediate | Several interacting modules, custom losses or interventions, and careful experimental controls. | PyTorch training and debugging; comfort reading research papers. |
| D3 — Advanced | Substantial instrumentation, Jacobian estimation, or circuit evaluation with sensitive methodological choices. | Strong PyTorch/autograd skills and patience with numerical and experimental debugging. |

Difficulty measures conceptual and engineering complexity, not prestige. All levels can produce excellent projects. The same minimum standards for originality and evidence apply to every level.

### Computing scale

| Level | Intended default |
|---|---|
| C0 — CPU-friendly | A laptop or workstation is sufficient for the core scope. |
| C1 — Light GPU | One 8–12 GB GPU is recommended; workloads are small or mostly use frozen models. |
| C2 — Moderate GPU | One 12–16 GB GPU is recommended; expect several training or gradient-based evaluation runs. |
| C3 — Substantial single-GPU | One 16–24 GB GPU is recommended; careful batching and caching are required. |

The catalog lists **estimated total GPU-hours for the minimum scope**, including its main baseline comparisons and repetitions. These are planning ranges, not measured runtimes, guaranteed allocations, or cloud prices. CPU-hours are listed where appropriate. Larger grids, extra models, and unsuccessful debugging runs can substantially increase costs. A GPU-hour means one GPU occupied for one hour; it is not elapsed calendar time or student labor.

Before scheduling full experiments, measure one small run and estimate:

    total compute ≈ measured time per run × planned runs + evaluation overhead

Record the hardware, precision, batch size, sequence length or image resolution, and peak memory. Confirm hardware availability early. No core project requires a paid API, proprietary dataset, multi-GPU training, or training a foundation model.

## 3. Proposal and six-week milestones

### Week 1: first-day checklist and proposal talk

Every team must finish the [first-day checklist](#9-first-day-checklist) during Week 1 and present its proposal in a **five-minute talk**. The talk is the proposal submission; a written proposal is not required.

Use the talk to explain:

1. Your project ID or proposed title, team, research question, and motivation.
2. The system you will build and the original component you will implement.
3. Your readings, data/model access, data splits, and first-start or equivalent pilot findings.
4. Baselines, evaluation metrics, computing estimate, and six-week scope.
5. Division of work, an expected outcome, a plausible failure outcome, and unresolved assumptions.

For a student-proposed project, obtain the advisor's approval during this talk.

### Weeks 2–5: required Canvas milestone reports

Each team must submit **one milestone report through Canvas in each of Weeks 2, 3, 4, and 5**. The technical milestones below and in the project briefs are **highly suggested progress targets**. Teams may adjust their work sequence to their project; the weekly reports remain required.

Keep each report concise and include:

- Project ID/title, team members, and reporting week.
- Work completed and each member's contribution.
- Current evidence: relevant results, figures, implementation checks, or a clear account of unsuccessful experiments.
- Progress relative to the suggested target, blockers, and justified scope or schedule changes.
- The next week's plan and any computing/resource issues.

### Week 6: final presentation and submission

At the end of the semester, in Week 6, each team must give a **10-minute final project presentation followed by five minutes of Q&A** and submit a **research report and reproducibility package**. See [Final submission](#7-final-submission) for details.

The instructor will publish calendar deadlines, talk schedules, and submission details. Week 6 is the final project week at the end of the semester.

| Week | Required deliverable | Highly suggested technical progress |
|---|---|---|
| 1 | Completed first-day checklist and five-minute proposal talk; advisor approval during the talk for student-proposed projects | Dataset inspection, fixed split, resource-access check, a working pilot, and a measured computing estimate. |
| 2 | Canvas milestone report | Baselines, core implementation, and small correctness checks. |
| 3 | Canvas milestone report | Working end-to-end system, initial comparison, revised compute estimate, and saved system state. |
| 4 | Canvas milestone report | Main experiments under the predefined evaluation protocol. |
| 5 | Canvas milestone report | Ablations, repetitions, uncertainty estimates, and failure analysis. |
| 6 | 10-minute final presentation + five-minute Q&A; research report and reproducibility package | Completed analysis, reproducible results, and clear conclusions. |

Use the suggested milestones to identify when to narrow scope. If the pilot reveals weak model competence, missing resources, or unreasonable cost, use the candidate brief's fallback or revise your own project's scope and document the change. Discuss material changes to the research question or evaluation protocol with the advisor before the main evaluation.

## 4. What you must implement

Implement the modules explicitly marked **Must implement yourselves** in your selected brief, or the corresponding core modules agreed with the advisor for your own project. Connect them into your own data-to-results pipeline. A notebook is acceptable if it supports a clean restart and automated reproduction.

| Reuse | Allowed role |
|---|---|
| PyTorch, NumPy, scikit-learn, plotting packages | Standard operations, optimizers, numerical routines, and visualization. |
| Pretrained model weights and official processors/tokenizers | Fixed starting models, with the exact model revision recorded. |
| Model-hook libraries | Accessing activations, gradients, or patch locations. |
| Public data loaders and split metadata | Dataset access and reference preprocessing. |
| Authors' complete methods | Comparison baselines, cited and distinguished from your contribution. |
| Existing implementations of your designated core module | Reading and correctness comparison; do not submit a lightly modified copy as your implementation. |

Maintain a short reuse statement listing each external repository, its role, any adapted files, and its license. Explain algorithmic differences between your implementation and the baseline.

Use small checks that expose meaningful errors: a Jacobian versus finite differences, an identity patch that preserves outputs, a known synthetic rationale, or a restored checkpoint that restores predictions. A large collection of tests that simply repeats the implementation is unnecessary.

Follow course and university policies on collaboration and generative AI. Where assistance is permitted, disclose substantial assistance, verify the result, and retain responsibility for understanding every submitted component.

## 5. Experimental design and data access

### Shared requirements

- Separate training, development, and final testing. Select hyperparameters, thresholds, features, and stopping rules using training/development data.
- Freeze the main protocol before final evaluation. Reusing test images under different corruptions does not create independent test examples.
- Fit preprocessing, concept selection, PCA, normalizers, and graph construction using only permitted data.
- Compare methods on the same model, starting checkpoint, examples, and computational budget. Separate architectural or training-budget changes from method effects.
- Use at least two relevant baselines and two ablations of your contribution, as specified in the candidate brief or agreed evaluation plan for your own project.
- For newly trained core methods, target three independent seeds unless the brief specifies a different run-level design. For fixed-model studies, repeat generated-data or resampling seeds. Report all completed planned runs.
- Give mean and variation across independent runs. Paired confidence intervals over examples may supplement this, but do not treat correlated checkpoints or repeated image corruptions as independent samples.
- Report performance by condition, not only an aggregate. Include representative failures and computational cost.

### Different distribution-shift settings

| Setting | Permitted information |
|---|---|
| Domain generalization | Source training/development environments only. Held-out target inputs and labels are excluded from fitting, model selection, and graph construction. |
| Unsupervised domain adaptation | Labeled source training data and the declared unlabeled target training partition. Target test labels are evaluation-only. |
| Source-free test-time adaptation | Starting model and declared saved source statistics. No source examples or target labels in the deployment loop. |
| Test-time training | Train an auxiliary self-supervised task before deployment; update with unlabeled deployment examples using the declared protocol. |

Test-time training here refers to the self-supervised adaptation setting of [Sun et al.](https://proceedings.mlr.press/v119/sun20b.html), not to recent recurrent sequence-model architectures also called TTT.

For online evaluation, declare whether predictions occur before or after updating on the current unlabeled batch. The default in this catalog is **predict first, update second**; an adapt-then-predict evaluation may be added and must be labeled separately. Never update on future batches. Specify what persists between batches, when resets occur, and whether normalization statistics and optimizer states are restored.

Some evaluations simulate interventions using known concepts or construct imbalanced streams using labels. This is allowed only in the evaluator. It does not grant those labels to an intervention-selection policy or an adaptation algorithm.

## 6. Public resources and responsible interpretation

Primary papers, public repositories, dataset landing pages, and official tutorials are linked in each description. Links were checked during preparation on **October 11, 2026**; access can change. The links are starting resources, not confirmation that every historical environment or downloadable checkpoint runs unchanged.

Record package versions and repository revisions. Start with a current, compatible environment instead of assuming an old requirements file remains appropriate. Preserve dataset license terms and download instructions. CUB has research/education restrictions and a documented possible overlap with ImageNet pretraining; disclose pretrained-model use rather than claiming contamination-free generalization.

Synthetic datasets are deliberately included when exact ground truth matters. Your generator is part of the implementation: document its rules and save split seeds and manifests. Do not tune a generator to make only your method succeed.

Interpret conclusions at the level supported by evidence:

- Agreement with annotations is different from faithfulness to model computation.
- Attention concentration is different from a demonstrated causal contribution.
- An internal intervention can establish a model-component effect under that intervention; it does not establish real-world causality.
- A token readout from a lens does not, by itself, establish a model's intentions or reasoning.
- Results on one model or synthetic task should be described with that scope.

## 7. Final submission

Every team must complete all three final deliverables in Week 6: the final presentation with Q&A, the research report, and the reproducibility package.

### Research report

Suggested length: **6–8 pages**, excluding references and optional appendices. Include:

1. Question, motivation, and precise contribution.
2. Relationship to two or more relevant papers.
3. Model, data, permitted information, and split construction.
4. Method and system design, including the student-written components.
5. Baselines, metrics, and experimental protocol.
6. Main results, ablations, uncertainty, and failure analysis.
7. Compute use, limitations, and what the evidence supports.
8. A brief individual contribution statement.

Do not fill space with a general literature survey. Use diagrams or tables where they clarify the actual system or comparison.

### Reproducibility package

Provide:

- A README explaining setup, data access, a small demonstration, and the main experiments.
- Source code, environment specification, configuration files, and seed/split manifests.
- Baseline and proposed-method results in machine-readable form, plus scripts that regenerate the figures.
- Small checkpoints or explicit checkpoint-download instructions where licensing and size permit.
- A reuse/citation statement and individual contribution statements.
- A measured compute summary and a list of any experiments omitted from the original plan.

A suggested repository layout is:

    README.md
    requirements.txt or environment.yml
    src/
    configs/
    scripts/
    manifests/
    results/
    checkpoints/    # small artifacts or download instructions

Large datasets need not be submitted. Provide access instructions and enough metadata to reconstruct the exact subset.

### Final project presentation (Week 6)

At the end of the semester, each team gives a **10-minute presentation followed by five minutes of Q&A**.

Present the research question, system and original implementation, experimental protocol, main baseline comparisons, ablations, and conclusions. Include an informative failure case and discuss limitations. A brief live or recorded demonstration may help explain the system, but it should fit within the 10-minute presentation.

All team members should be prepared to explain the main design and evaluation decisions and answer questions. The research report and reproducibility package are required in addition to the presentation.

## 8. Suggested assessment rubric

The instructor may adjust these weights before releasing the assignment.

| Criterion | Points | What earns credit |
|---|---:|---|
| Core implementation and original component | 30 | Correct, understandable student-written modules integrated into a working system. |
| Experimental design and execution | 25 | Appropriate baselines, controlled comparisons, clean data access, and useful ablations. |
| Research analysis | 20 | Evidence-based conclusions, uncertainty, failures, and limits. |
| Reproducibility and computing discipline | 15 | Runnable package, recorded settings, accessible resources, and justified scope. |
| Report, presentations, and team contributions | 10 | Clear communication and substantial participation by both students. |
| **Total** | **100** | |

**More difficult level, larger computing scale, optional extensions DO earn extra credit.**

## 9. First-day checklist

Every team must complete this checklist in Week 1 and present its findings in the five-minute proposal talk. For your own project, complete the equivalent steps using your selected resources and pilot.

1. Select a candidate project or formulate your own course-relevant project; identify the original component.
2. Open the candidate's required readings or your own relevant readings and download only the core data/model.
3. Create the split manifest before training or discovery.
4. Run the first-start experiment in the candidate brief or an equivalent small feasibility experiment for your own project.
5. Check that the main phenomenon is measurable and estimate full cost.
6. Divide implementation and analysis responsibilities, with both students reviewing each other's work.
7. Prepare the five-minute proposal talk with your plan, pilot findings, baselines, evaluation, and unresolved assumptions. Student-proposed projects must receive advisor approval during the talk.
