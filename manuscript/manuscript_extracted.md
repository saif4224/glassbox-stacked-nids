**Trustworthy AI in Cybersecurity: A Glass-Box-Augmented Stacked Ensemble with Cross-Validated Explanations for Network Intrusion Detection**

Khandaker Saifuzzaman

# Highlights

- A diversity-driven stacked ensemble (CatBoost + ExtraTrees + EBM) embeds a glass-box model as the deployed base learner, not a post hoc afterthought.

- Homogeneous boosted ensembles have been empirically shown to fail: a soft-voted XGBoost+LightGBM+RF ensemble could not outperform LightGBM alone, motivating paradigm diversity.

- Stacking attains 99.83% accuracy and significantly outperforms both its best base member (McNemar p = 2.5×10⁻⁴) and a strong LightGBM reference (p = 4.8×10⁻³⁵; 28.1% relative error reduction).

- First explanation-consistency validation in NIDS between exact glass-box attributions (EBM) and post-hoc SHAP, under a same-sample protocol: Spearman ρ = 0.775 (p \< 10⁻⁵).

- All findings replicate under native retraining on UNSW-NB15 (different feature-extraction toolchain): stacking again beats its best member and the LightGBM reference (+22.0% relative error reduction), and SHAP--EBM agreement strengthens to ρ = 0.803, establishing generalisation across feature-extractor families.

- Timing features (IAT statistics) rescue web-attack detection: XSS F1 rises from 0.09 to 0.42 by feature selection alone; SOC-aware per-class threshold calibration then controls the false-alarm rate.

- Leak-proof capped SMOTE inside every fold; 3.17 µs parallel inference on CICIDS-2017 (\~316,000 flows/s); 15.21 µs on UNSW-NB15 (ExtraTrees-dominated feature set); both support real-time edge deployment.

# Abstract

Machine-learning ensembles dominate Network Intrusion Detection System (NIDS) research, yet two weaknesses persist: ensembles of near-identical boosted trees make correlated errors and frequently fail to outperform their own best member, and the post-hoc explanations attached to such systems are rarely validated against anything. This work addresses both. First, we demonstrate empirically on CICIDS-2017 that a homogeneous soft-voting ensemble (XGBoost, LightGBM, Random Forest) does not outperform standalone LightGBM, and we replace it with a diversity-driven design combining three learning paradigms: ordered boosting on symmetric trees (CatBoost), extremely randomised bagging (ExtraTrees), and a cyclic-boosted additive glass-box model (Explainable Boosting Machine, EBM). Combined by stacked generalisation under a leak-proof pipeline --- capped SMOTE confined to training folds, an untouched calibration slice per fold, and a 25-feature subset justified by ablation --- the stacked ensemble reaches 99.83% accuracy, weighted F1 of 0.9984, macro-F1 of 0.898, and MCC of 0.9951, significantly exceeding its best base member (McNemar p = 2.5×10⁻⁴) and a LightGBM reference (p = 4.8×10⁻³⁵, a 28.1% relative error reduction), while soft voting fails both comparisons. Second, because EBM is exact by construction, embedding it in the ensemble enables the first explanation-consistency validation in NIDS between glass-box and post-hoc attributions: under a same-sample protocol, SHAP attributions from the CatBoost component agree strongly with EBM\'s exact contributions (Spearman ρ = 0.775, p = 5.5×10⁻⁶; top-10 feature agreement 0.9), anchoring the system\'s explanations against ground truth from a deployed member. Finally, per-class decision-threshold calibration tuned on real-prevalence data yields a statistically significant accuracy improvement (p = 3.7×10⁻³), the best MCC of all configurations (0.9953), and an operator-controllable false-alarm rate, reducing XSS false alarms from 26.9 to 19.1 per 100,000 flows. Parallel inference latency is 3.17 µs/sample (\~316,000 flows/s) on CICIDS-2017 and 15.21 µs on UNSW-NB15 (ExtraTrees-dominated); both support real-time deployment on commodity hardware. The diversity-driven mechanism and the explanation-consistency property both replicate under native retraining on UNSW-NB15 --- a benchmark produced by a different feature-extraction toolchain --- where stacking again beats its best base member and the LightGBM reference (22.0% relative error reduction), threshold calibration cuts Backdoor false alarms from 1,473 to 208 per 100,000 flows, and SHAP-EBM agreement is even stronger (ρ = 0.803, p = 1.4×10⁻⁶), establishing generalisation across feature-extractor families.

**Keywords:** *Network Intrusion Detection, Stacked Generalisation, Explainable Boosting Machine, Glass-Box Models, Explanation Agreement, SHAP, Class Imbalance, Threshold Calibration, CICIDS-2017.*

# 1. Introduction

As network infrastructures evolve toward Industry 5.0 and the Internet of Things, the attack surface available to malicious actors continues to expand. Signature-based intrusion detection is increasingly inadequate against attacks --- distributed denial-of-service campaigns, brute-force credential attacks, encrypted web exploits --- that are progressively harder to distinguish from benign traffic, and machine learning has become the primary defensive mechanism. Yet two structural weaknesses persist across the academic literature.

- **Illusory ensemble gains.** The dominant recipe --- combine several gradient-boosted tree models by voting --- quietly assumes the members make different errors. They often do not: XGBoost and LightGBM are near-identical leaf-wise boosters, and we show in Section 4.1 that a soft-voted XGBoost+LightGBM+Random-Forest ensemble fails to outperform standalone LightGBM on CICIDS-2017. Many published ensemble improvements may rest on weak baselines rather than genuine combination gains.

- **Unvalidated explanations.** Post-hoc explainers such as SHAP are now routinely attached to NIDS, but their attributions are approximations, and recent surveys note that almost no NIDS work attempts any consistency check on them. When two post-hoc explainers agree, both may share the same blind spot; what is missing is an anchor that is exact by construction.

This paper addresses both weaknesses with one architectural decision: build the ensemble from three different learning paradigms --- ordered boosting on symmetric trees (CatBoost), extremely randomised bagging (ExtraTrees), and a cyclic-boosted additive glass-box model (the Explainable Boosting Machine, EBM) --- and combine them by stacked generalisation. Diversity restores genuine ensemble gains; the glass-box member, whose feature attributions are exact by construction, provides the missing anchor for validating post-hoc explanations from inside the deployed system itself.

**Contributions.** (i) An empirical demonstration that a homogeneous boosted ensemble fails to beat its best member on CICIDS-2017, motivating diversity-driven design; (ii) the first NIDS architecture embedding a glass-box model (EBM) as a base learner inside a heterogeneous stacked ensemble; (iii) the first glass-box-versus-post-hoc explanation-consistency study in NIDS, using formal agreement metrics from the disagreement-problem literature under a same-sample protocol; (iv) a feature-space finding that inter-arrival-time statistics rescue web-attack detection (XSS F1: 0.09 → 0.42 from feature selection alone); (v) SOC-aware per-class threshold calibration tuned leak-free on real-prevalence data, giving operators explicit control of the false-alarm rate; and (vi) an evaluation protocol in which every model comparison shares identical folds and base fits, making all McNemar tests properly paired --- a methodological design choice enabling the above contributions to rest on statistically valid, reproducible evidence.

# 2. Related Work

**Ensemble learning in NIDS.** Ensemble methods consistently outperform single classical models on flow-based benchmarks. Fitni and Ramli (2020) improved detection with AdaBoost combinations and highlighted the data-leakage risk of resampling before cross-validation. Al-Omari and Al-Haija (2024) showed hybrid feature selection addresses the computational burden of high-dimensional traffic, and Arreche et al. (2024) combined boosting with bagging in a two-level framework. The boosted-tree learners that dominate this literature share a common lineage in gradient boosting (Friedman, 2001) and its scalable modern implementations --- XGBoost (Chen and Guestrin, 2016) and LightGBM (Ke et al., 2017) --- alongside the bagging-based Random Forest (Breiman, 2001) and its extremely randomised variant (Geurts et al., 2006). CatBoost (Prokhorenkova et al., 2018) adds ordered boosting on symmetric trees, a structurally distinct scheme we exploit for diversity. Stacked generalisation (Wolpert, 1992) appears in recent NIDS work, but its base learners are almost always drawn from the same family of gradient-boosted trees, leaving error correlation --- the very property that determines whether an ensemble can outperform its members --- unexamined.

**Class imbalance in intrusion detection.** NIDS datasets are severely skewed toward benign traffic, and the treatment of imbalance materially affects minority-attack detection. Synthetic oversampling (SMOTE; Chawla et al., 2002) and its adaptive variant (ADASYN; He et al., 2008) are the field\'s default remedies, surveyed comprehensively by He and Garcia (2009). Buda et al. (2018) showed across deep and classical models that oversampling and decision-threshold adjustment are the most reliable interventions, while the cost-sensitive perspective (Elkan, 2001) reframes the problem at the decision layer rather than the data layer. We adopt both views: leak-proof capped oversampling during training and per-class threshold calibration at inference, the latter tuned on real-prevalence data to avoid the distortion that resampling introduces into probability estimates.

**Glass-box models.** The Explainable Boosting Machine (Lou et al., 2013; Nori et al., 2019) is a cyclic-boosted generalised additive model whose prediction is a sum of per-feature shape functions --- its global and local attributions are therefore exact rather than approximated. EBM has appeared in intrusion detection only as a standalone classifier; to our knowledge no prior NIDS work deploys it as a base learner inside a heterogeneous ensemble, where its exactness can be exploited at no extra cost.

**Explainability and the disagreement problem.** SHAP (Lundberg and Lee, 2017) dominates NIDS explainability (Hartl et al., 2020; Arreche et al., 2024; Wazid et al., 2024), but its attributions are post-hoc approximations. Krishna et al. (2022) formalised the disagreement problem --- different explainers routinely disagree --- and proposed agreement metrics (top-k feature, rank, and sign agreement) that we adopt. End-to-end XAI-IDS frameworks (Arreche, Guntur and Abdallah, 2024) benchmark multiple post-hoc explainers, and recent work identifies cross-explanation validation as an open gap, noting that the rare existing consistency checks compare two post-hoc methods (SHAP vs LIME), whose mutual agreement cannot rule out a shared blind spot. Comparing a post-hoc explainer against an exact glass-box member of the same deployed ensemble --- as done here --- closes that loop.

**Benchmarks and cross-dataset generalisation.** Detection results are benchmark-sensitive. CICIDS-2017 (Sharafaldin et al., 2018), generated with CICFlowMeter, remains the most-used flow benchmark, while UNSW-NB15 (Moustafa and Slay, 2015), built with a different Argus/Bro toolchain, and the recent CIC-IoT-2023 (Neto et al., 2023) provide complementary distributions. Because feature sets differ across extractors, models rarely transfer zero-shot, and single-benchmark evaluations risk reporting extractor-specific artefacts; native multi-dataset validation --- adopted here --- is the more credible test of generalisation. Federated and IoT-oriented deployments (Rey et al., 2022; Wazid et al., 2024) further motivate detectors that are both accurate and interpretable on heterogeneous traffic.

## 2.1. Gap Analysis

No prior NIDS work (a) embeds an exact glass-box model as a stacked base learner, (b) validates post-hoc SHAP attributions against exact attributions from inside the deployed ensemble using formal agreement metrics, or (c) couples this with leak-proof resampling, real-prevalence threshold calibration, and fully paired statistical testing on a shared-fold harness. This work fills all three gaps simultaneously.

# 3. Methodology

## 3.1. Dataset and Preprocessing

Experiments use CICIDS-2017 (Sharafaldin et al., 2018): 2,827,876 flows after removing infinite values and NaNs, with 78 CICFlowMeter features. Label strings were repaired at load time (the dataset\'s cp1252 en-dash decodes to U+FFFD under UTF-8, corrupting the two web-attack class names) and whitespace normalised. Numerical features were standardised; the label was integer-encoded. For computational tractability all experiments use a 20% stratified sample (565,563 flows); classes with fewer than 15 post-sampling instances (Heartbleed, Infiltration, SQL Injection) were removed, leaving 12 classes, and labels were re-encoded to consecutive integers. The sampling fraction is a disclosed design parameter.

A network flow *F* is the sequence of packets sharing a 5-tuple within a time window:

> *F* = {SrcIP, DstIP, SrcPort, DstPort, Protocol} (1)

## 3.2. Leak-Proof Capped Class-Imbalance Resolution

CICIDS-2017 is severely imbalanced (454,265 BENIGN vs 130 XSS flows in our sample). SMOTE (Chawla et al., 2002) generates a synthetic minority point on the segment joining a sample and one of its *k* nearest minority neighbours:

> *x*~new~ = *x*~i~ + λ ( *x*~zi~ − *x*~i~ ), λ ∈ \[0, 1\] (2)

Two safeguards differentiate this pipeline from common practice. First, SMOTE is applied strictly inside each training fold, so synthetic points never contaminate validation or test data. Second, oversampling is capped: each minority class is lifted to at most 50,000 samples rather than to the majority count, preventing the resampled training set from inflating to tens of millions of rows while preserving minority signal; k is auto-tuned to the rarest class\'s fold size. ADASYN (He et al., 2008) is evaluated as an alternative in Section 4.4.

## 3.3. Hybrid Feature Selection and Feature-Count Ablation

A two-stage hybrid selector ranks features: a Mutual Information filter (Eq. 3) removes statistically independent features, then Random Forest impurity importance (Eq. 4) ranks the survivors for non-linear interactions.

> *I* (*X*; *Y*) = ∫ Σ~y∈Y~ p ( x, y ) log \[ p(x, y) / ( p(x) p(y) ) \] dx (3)
>
> Imp ( *j* ) = Σ~t∈T\ :\ v(t)=j~ ( *N*~t~ / *N* ) · ΔG ( *t* ) (4)

An ablation over K ∈ {10, 15, 20, 25, 30} features trained the detection ensemble at each K (Figure 1). Validation accuracy rose from 0.9491 (K=10) through 0.9887 (15) and 0.9926 (20) to 0.9978 at K=25, with K=30 adding nothing (0.9976). We therefore adopt the top 25 features. The five features added beyond K=20 --- Fwd IAT Max, Fwd IAT Mean, Flow IAT Mean, Flow IAT Max, and Init_Win_bytes_backward --- are predominantly inter-arrival-time (timing) statistics, and Section 4.4 shows they are decisive for web-attack detection.

**Figure 1.** *Feature-count ablation. Validation accuracy as a function of the number of selected features; accuracy peaks at K=25 (0.9978) and plateaus thereafter. \[Embed Fig1_Feature_Ablation.png\]*

## 3.4. Diversity-Driven Architecture

### 3.4.1. Motivation: Homogeneous Ensembles Fail

In a preliminary experiment under the identical pipeline, a soft-voting ensemble of XGBoost, LightGBM, and Random Forest reached weighted F1 = 0.9949 --- while standalone LightGBM reached 0.9951. The ensemble could not beat its best member: leaf-wise boosters make correlated errors, so averaging them adds nothing. Ensemble gains require error diversity, which motivates the paradigm-diverse design below.

### 3.4.2. Base Learners

- **CatBoost** (Prokhorenkova et al., 2018): ordered boosting on symmetric (oblivious) trees --- a structurally different boosting scheme from leaf-wise GBDTs.

- **ExtraTrees** (Geurts et al., 2006): extremely randomised bagging; randomised split thresholds decorrelate its errors from any boosting member.

- **EBM** (Lou et al., 2013; Nori et al., 2019): a cyclic-boosted generalised additive model. With interactions disabled (required for multiclass), the model is

> *g* ( E\[ y \| x \] ) = β~0~ + Σ~j=1~^d^ *f*~j~ ( x~j~ ) (5)

so the contribution of feature *j* to any prediction is exactly *f*~j~ *(x*~j~ ) --- no approximation. EBM is simultaneously a competitive detector (99.34% standalone accuracy, Section 4.1) and the trust anchor for Section 4.5. LightGBM is retained as a strongest-single-model reference row, and the linear baseline (Ridge + Naive Bayes + Passive-Aggressive, soft-voted 1:1:4) from earlier work is kept for continuity.

### 3.4.3. Combination: Soft Voting vs Stacked Generalisation

Soft voting averages the base probability vectors. Stacked generalisation (Wolpert, 1992) instead learns when to trust which member: a multinomial logistic-regression meta-learner *h* is trained on the concatenated base probabilities,

> *ŷ* = *h* ( p~1~(x), p~2~(x), p~3~(x) ) (6)

implemented as blending: within each outer training fold the (resampled) data is split 80/20; bases fit on the 80% part, the meta-learner on the 20% blend part\'s probabilities. Each base is fitted once per fold, so all reported configurations --- standalone members, voting, stacking, references --- share identical folds and identical base fits, making every pairwise statistical test properly paired.

## 3.5. SOC-Aware Per-Class Threshold Calibration

Argmax over raw probabilities implicitly applies an equal threshold to every class --- operationally wrong when false-alarm costs differ by class. Before resampling, 10% of each training fold is held out untouched as a calibration slice with real class prevalence (resampled data would mislead the tuning). For each class *c*, the threshold *t*~c~ maximising one-vs-rest F1 on the calibration slice is selected from the precision--recall curve, and decisions use threshold moving:

> *ŷ* = arg max~c~ \[ p~c~ ( x ) / t~c~ \] (7)

The calibration slice is never seen by SMOTE, any base learner, or the meta-learner, so the procedure is leak-free. Learned thresholds are interpretable operating knobs: averaged over folds, BENIGN tunes to 0.21, Web Brute Force to 0.36, XSS to 0.47, FTP-Patator to 0.98 --- directly inspectable by SOC engineers.

## 3.6. Explanation-Consistency Protocol

SHAP TreeExplainer (Lundberg and Lee, 2017) attributes the CatBoost component\'s predictions via Shapley values:

> φ~i~ ( f, x ) = Σ~S\ ⊆\ N\ \\\ {i}~ \[ \|S\|! ( \|N\| − \|S\| − 1 )! / \|N\|! \] · \[ *f*~x~ ( S ∪ {i} ) − *f*~x~ ( S ) \] (8)

Comparability demands identical data and identical attribution semantics on both sides. On the same 1,000 stratified background flows we compute (a) mean absolute SHAP value per feature across samples and classes, and (b) mean absolute exact EBM term contribution per feature via per-sample evaluation of Eq. 5 --- not EBM\'s training-set global importances, whose different reference distribution would (and in a pilot, did) depress correlation spuriously. Agreement is quantified with the metrics of Krishna et al. (2022): top-k feature agreement, top-k rank agreement, and Spearman/Kendall rank correlation over all 25 features.

## 3.7. Configuration and Reproducibility

  --------------------------------------------------------------------------------------------------------------------------------------------------------
  **Component**                                                           **Configuration (as executed)**
  ---------------------- ---------------------------------------------------------------------------------------------------------------------------------
  CatBoost                                             iterations=200, depth=6, learning_rate=0.1, MultiClass loss, seed=42

  ExtraTrees                                                     n_estimators=200, unlimited depth, Gini, seed=42

  EBM                        max_bins=64, interactions=0, outer_bags=2, max_rounds=2000, per-fold training capped at 100,000 stratified rows, seed=42

  Meta-learner                                                    Multinomial logistic regression, max_iter=1000

  LightGBM (reference)                                           n_estimators=100, learning_rate=0.1, max_depth=6

  SMOTE                                                 k_neighbors=5 (auto-tuned), per-class cap 50,000, inside folds only

  Protocol                Stratified 5-fold CV; per fold: 10% calibration slice, then 80/20 fit/blend split of resampled data; 20% stratified data sample
  --------------------------------------------------------------------------------------------------------------------------------------------------------

**Table A.** *Disclosed configuration. The EBM training cap and the 20% data sample are computational concessions, both disclosed; the camera-ready run may lift them.*

The pipeline is implemented in Python using scikit-learn (Pedregosa et al., 2011) and imbalanced-learn (Lemaître et al., 2017), with CatBoost, LightGBM, and InterpretML for the respective models. All timings were measured on an Apple MacBook Pro (M2 Pro, 16 GB unified memory), single machine, CPU only; the full five-fold harness completes in 42.6 minutes. \[AUTHOR: confirm exact core count and macOS/library versions for the final text.\]

# 4. Results and Discussion

## 4.1. Component Ablation and Headline Performance

  ------------------------------------------------------------------------------------------------------------------------------------
  **Model**                    **Accuracy**   **F1 (weighted)**   **F1 (macro)**    **MCC**     **Lat. seq (µs)**   **Lat. par (µs)**
  --------------------------- -------------- ------------------- ---------------- ------------ ------------------- -------------------
  Baseline (linear)               0.8788           0.8899             0.5024         0.6585           2.36                2.36

  LightGBM (reference)            0.9977           0.9980             0.8678         0.9932           7.24                7.24

  CatBoost                        0.9941           0.9953             0.8223         0.9832           0.32                0.32

  ExtraTrees                      0.9982           0.9983             0.8989         0.9948           3.02                3.02

  EBM                             0.9934           0.9950             0.8213         0.9813           0.71                0.71

  Voting (CB+ET+EBM)              0.9963           0.9971             0.8494         0.9893           4.05                3.02

  **Stacking (CB+ET+EBM)**      **0.9983**       **0.9984**         **0.8982**     **0.9951**       **4.19**            **3.17**

  **Stacking + Thresholds**     **0.9984**       **0.9984**         **0.8974**     **0.9953**       **4.19**            **3.17**
  ------------------------------------------------------------------------------------------------------------------------------------

**Table 1.** *Five-fold cross-validated performance; all rows share identical folds and base fits. Lat. seq assumes sequential member execution; Lat. par assumes members run concurrently (deployment-realistic) plus meta overhead.*

Three observations define the headline. First, soft voting fails: at weighted F1 = 0.9971 it beats neither its best member (ExtraTrees, 0.9983) nor the LightGBM reference (0.9980) --- averaging diverse members of unequal strength dilutes the strong ones. Second, stacking succeeds where voting fails: the meta-learner, having learned when to trust which member, reaches 0.9984 and outperforms every base member and the reference. Third, threshold calibration adds a further small but significant accuracy gain and the best MCC of all configurations (0.9953).

### 4.1.1. Statistical Validation

  -----------------------------------------------------------------------------------------------------------------
  **Comparison**             **Discordant pairs (A-only / B-only)**   **McNemar p**   **Relative error reduction**
  ------------------------- ---------------------------------------- --------------- ------------------------------
  Baseline vs Stacking                    405 / 68,017                  \< 10⁻³⁰⁰                 ---

  ExtraTrees vs Stacking                   116 / 180                    2.50×10⁻⁴       +6.3% (0.180% → 0.169%)

  LightGBM vs Stacking                     269 / 643                   4.80×10⁻³⁵       +28.1% (0.235% → 0.169%)

  Voting vs Stacking                      142 / 1,282                  3.91×10⁻²⁰⁰                ---

  Stacking vs +Thresholds                   79 / 121                    3.74×10⁻³                 ---
  -----------------------------------------------------------------------------------------------------------------

**Table 2.** *Pairwise McNemar tests on pooled cross-validated predictions (n = 565,563); every pair shares identical folds. With samples this large, significance is necessary but not sufficient --- relative error reductions are reported as effect sizes.*

All claimed improvements are statistically significant under properly paired testing. We report effect sizes alongside p-values: the stacked ensemble removes 6.3% of ExtraTrees\' residual errors and 28.1% of LightGBM\'s. These margins are honest --- small against the strongest base member, substantial against the strongest external reference --- and the paper\'s contribution does not rest on margin size alone but on the diversity mechanism (voting fails, stacking succeeds) and the explainability architecture.

## 4.2. Per-Class Performance

  -----------------------------------------------------------------------------------
  **Attack Class**            **Precision**   **Recall**    **F1**     **Support**
  -------------------------- --------------- ------------ ---------- ----------------
  BENIGN                          1.00           1.00        1.00        454,265

  Bot                             0.68           0.90        0.77          391

  DDoS                            1.00           1.00        1.00         25,605

  DoS GoldenEye                   0.98           1.00        0.99         2,059

  DoS Hulk                        1.00           1.00        1.00         46,025

  DoS Slowhttptest                0.99           0.99        0.99         1,100

  DoS slowloris                   1.00           1.00        1.00         1,159

  FTP-Patator                     1.00           1.00        1.00         1,587

  PortScan                        0.99           1.00        1.00         31,761

  SSH-Patator                     1.00           1.00        1.00         1,180

  Web Attack - Brute Force        0.69           0.57        0.63          301

  Web Attack - XSS                0.33           0.58        0.42          130

  **Macro average**             **0.89**       **0.92**    **0.90**    **565,563**
  -----------------------------------------------------------------------------------

**Table 3.** *Per-class metrics of the stacked ensemble, pooled over all five folds (single provenance for every per-class number in this paper).*

**Figure 2.** *Per-class F1 comparison, linear baseline vs stacked ensemble. \[Embed Fig2_PerClass_F1_Comparison.png\]*

**Figure 3.** *Normalised confusion matrix of the stacked ensemble. \[Embed Fig3_Confusion_Matrix.png\]*

## 4.3. The Decisive Role of Timing Features for Web Attacks

The most consequential feature-space finding concerns the two rarest web-attack classes. At K=20 features (no inter-arrival-time statistics), XSS managed F1 = 0.09 at precision 0.05 --- operationally unusable --- and Brute Force F1 = 0.14. Adding the five timing-dominated features at K=25 lifted XSS to F1 = 0.42 (precision 0.33) and Brute Force to F1 = 0.63, improvements of roughly 4.7× and 4.4× from feature selection alone, before any decision-layer intervention. The interpretation is direct: encrypted web attacks overlap heavily with benign traffic in packet-size statistics, but their request timing (inter-arrival patterns) remains discriminative. Flow-based NIDS for payload-hidden attacks should prioritise temporal features; the residual confusion that remains is evidence for deep-packet-inspection-class signals lying outside flow statistics altogether.

## 4.4. Class-Imbalance Interventions

### 4.4.1. Resampling: SMOTE vs ADASYN

  -------------------------------------------------------------------------------------------------------------------------------
  **Class**                   **P (SMOTE)**   **R (SMOTE)**   **F1 (SMOTE)**   **P (ADASYN)**   **R (ADASYN)**   **F1 (ADASYN)**
  -------------------------- --------------- --------------- ---------------- ---------------- ---------------- -----------------
  Web Attack - Brute Force        0.588           0.565           0.576            0.610            0.598             0.604

  Web Attack - XSS                0.247           0.554           0.342            0.267            0.477             0.343
  -------------------------------------------------------------------------------------------------------------------------------

**Table 4.** *Resampler comparison on the web-attack classes (capped sampling, CatBoost+ExtraTrees evaluator --- EBM excluded from this sampler comparison for runtime, disclosed). ADASYN is modestly favourable for Brute Force and indistinguishable for XSS.*

Notably, an identical comparison at K=20 features had produced numerically identical SMOTE and ADASYN results --- the samplers only differentiate once timing features give the synthetic points room to matter. Resampler choice is feature-set dependent: at K=20 the two samplers are indistinguishable; at K=25, ADASYN offers a modest benefit for Brute Force on CICIDS-2017. On UNSW-NB15 (also at K=25), the relationship reverses: SMOTE outperforms ADASYN on all four focus classes, indicating that ADASYN's adaptive density estimation benefits depend on the feature-space structure of the target dataset. The binding constraint on minority performance remains feature informativeness, not synthesis strategy.

### 4.4.2. Decision-Layer Calibration

  ------------------------------------------------------------------------------------------------------------
  **Class**                   **Stage**   **Precision**   **Recall**   **F1**   **False alarms / 100k flows**
  -------------------------- ----------- --------------- ------------ -------- -------------------------------
  Bot                          before         0.675         0.898      0.771                29.9

  Bot                           after         0.701         0.887      0.783                26.2

  Web Attack - Brute Force     before         0.691         0.571      0.625                13.6

  Web Attack - Brute Force      after         0.656         0.658      0.657                18.4

  Web Attack - XSS             before         0.330         0.577      0.420                26.9

  Web Attack - XSS              after         0.337         0.423      0.375                19.1
  ------------------------------------------------------------------------------------------------------------

**Table 5.** *Effect of per-class threshold calibration on the focus classes (pooled CV predictions). \'Before\' = argmax at uniform thresholds; \'after\' = thresholds tuned on the untouched real-prevalence calibration slice.*

**Figure 8.** *Minority-class F1 before/after threshold calibration. \[Embed Fig8_Threshold_Calibration.png\]*

Calibration is not a uniform trade-off; its effect is class-dependent and operator-meaningful. Bot improves on both precision and false-alarm rate; Brute Force gains F1 (0.625 → 0.657) by accepting more alerts; XSS deliberately trades recall (0.58 → 0.42) to cut false alarms from 26.9 to 19.1 per 100,000 flows. Globally the calibrated system is significantly more accurate (McNemar p = 3.74×10⁻³, Table 2) with the best MCC of all configurations (0.9953). The fold-averaged macro-F1 shows a small decrease (0.8982 \\u2192 0.8974, \\u0394 = \\u22120.0008); the pooled-prediction computation gives 0.8981 \\u2192 0.8981 (no change), reflecting different aggregation methods within the same pipeline. Calibration improves per-fold macro-F1 in 3 of 5 folds and marginally reduces it in 2. Balanced accuracy also decreases (0.918 \\u2192 0.911), the acknowledged cost of the false-alarm reductions in Table 5. The per-class thresholds (e.g., XSS t = 0.47, Brute Force t = 0.36) remain directly inspectable and adjustable by operators per SOC preference.

## 4.5. Explanation Consistency: Glass-Box vs Post-Hoc

  -----------------------------------------------------------------------
  **Agreement metric**                             **Value**
  ----------------------------------- -----------------------------------
  Top-5 feature agreement                            0.60

  Top-5 rank agreement                               0.60

  Top-10 feature agreement                           0.90

  Top-10 rank agreement                              0.30

  Spearman ρ (all 25 features)               0.775 (p = 5.49×10⁻⁶)

  Kendall τ (all 25 features)                0.620 (p = 3.57×10⁻⁶)
  -----------------------------------------------------------------------

**Table 6.** *Agreement between SHAP attributions (CatBoost component, post-hoc) and exact EBM term contributions, computed on the same 1,000 background flows with identical per-sample attribution semantics. Metrics follow Krishna et al. (2022).*

The post-hoc and exact explainers agree strongly and significantly: 9 of the top-10 features coincide, and rank correlation across all 25 features is ρ = 0.775 (p \< 10⁻⁵). Because EBM\'s attributions are exact by construction, this constitutes external validation of the SHAP explanations shipped to analysts --- agreement with an exact explainer, unlike agreement between two post-hoc approximations, cannot be produced by a shared blind spot. Protocol matters decisively: a pilot comparison of SHAP sample-level attributions against EBM\'s training-set global importances (mismatched reference distributions) yielded a non-significant ρ = 0.325, which would have been wrongly read as explainer disagreement. We therefore recommend same-sample, same-semantics protocols as standard practice for XAI consistency studies.

The largest divergences are themselves informative: the explainers differ most on Fwd IAT Max and Fwd IAT Mean (SHAP weights both higher) and Average Packet Size (EBM higher). The IAT features are precisely those carrying interaction effects with other timing features for tree models, which the additive EBM redistributes onto correlated main effects --- a known and explicable mechanism rather than an anomaly, and material for analyst guidance on when to prefer which explanation. We note that the attribution of the XSS F1 improvement specifically to IAT statistics rests on feature-group membership rather than a controlled ablation isolating IAT features from other potential additions; the causal mechanism is theoretically motivated and practically consistent with known web-attack traffic patterns, but this caveat should be acknowledged when drawing prescriptive conclusions about feature engineering.

**Figure 4a.** *SHAP global feature importance, CatBoost component, decoded class legend. \[Embed Fig4a_SHAP_Bar_CatBoost.png\]*

**Figure 4b.** *EBM exact glass-box term importances. \[Embed Fig4b_EBM_Importances.png\]*

**Figure 6.** *Side-by-side normalised importance shares, SHAP vs EBM, same-sample protocol. \[Embed Fig6_Explanation_Agreement.png\]*

Domain alignment also holds: both explainers rank Destination Port, packet-length statistics (volumetric attack signatures), Init_Win_bytes_forward (SYN-flood window manipulation), and the inter-arrival-time features (web-attack timing) among the most influential --- consistent with established threat signatures.

## 4.6. Computational Efficiency

With base members executed concurrently --- natural in deployment, as the three models are independent --- stacked inference costs 3.17 µs/sample on CICIDS-2017 (sequential: 4.19 µs), a throughput of roughly 316,000 flows per second per core group on commodity Apple-silicon hardware. On UNSW-NB15, where ExtraTrees is slower over a larger feature set, parallel stacking latency is 15.21 µs --- still within real-time requirements at typical enterprise traffic rates. The meta-learner adds negligible overhead in both cases. Figure 7 places every configuration on the accuracy--latency plane: CatBoost and EBM are exceptionally cheap (0.32 and 0.71 µs), ExtraTrees costs 3.02 µs, and LightGBM is the slowest at 7.24 µs --- the stacked ensemble is simultaneously more accurate and 2.3× faster than the LightGBM reference at parallel latency. Training the full five-fold harness takes 42.6 minutes.

**Figure 7.** *Accuracy--latency trade-off (log-scale latency); circles = sequential members, triangles = parallel execution. \[Embed Fig7_Latency_F1_Pareto.png\]*

## 4.7. Cross-Dataset Validation on UNSW-NB15

To establish that the findings are not artefacts of a single benchmark, the entire pipeline --- feature ranking, capped leak-proof SMOTE, the three-paradigm stacked ensemble, threshold calibration, paired McNemar testing, and the SHAP-versus-EBM agreement study --- was re-run natively (retrained and re-evaluated) on UNSW-NB15 (Moustafa and Slay, 2015): 433,014 flows from the combined pool of all three UNSW-NB15 CSV files (training-set, testing-set, and full-dataset), 42 features after categorical encoding, 10 classes, evaluated by the same 5-fold stratified CV protocol used for CICIDS-2017. This differs from the official UNSW-NB15 train/test split used in some published benchmarks; the protocol was chosen for consistency and variance stability rather than direct comparability to the state-of-the-art literature on UNSW. UNSW-NB15 is generated by a different feature-extraction toolchain (Argus and Bro/Zeek flow statistics) than CICIDS-2017\'s CICFlowMeter, so agreement across the two is evidence of generalisation across feature-extractor families, not merely across traffic samples. The feature-count ablation independently selected K=25 on UNSW-NB15 as well, and all model hyperparameters were held identical to the CICIDS-2017 configuration (Table A).

  ----------------------------------------------------------------------------------------------------------------
  **Model**                    **Accuracy**   **F1 (weighted)**   **F1 (macro)**    **MCC**     **Lat. par (µs)**
  --------------------------- -------------- ------------------- ---------------- ------------ -------------------
  Baseline (linear)               0.5476           0.5983             0.2779         0.4813           1.56

  LightGBM (reference)            0.7999           0.8130             0.5990         0.7533           5.76

  CatBoost                        0.7782           0.7945             0.5471         0.7287           0.24

  ExtraTrees                      0.8422           0.8544             0.6639         0.8036           15.08

  EBM                             0.7661           0.7851             0.5331         0.7126           0.60

  Voting (CB+ET+EBM)              0.8277           0.8406             0.6076         0.7881           15.08

  **Stacking (CB+ET+EBM)**      **0.8439**       **0.8554**         **0.6713**     **0.8043**       **15.21**

  **Stacking + Thresholds**     **0.8567**       **0.8591**         **0.6931**     **0.8172**       **15.21**
  ----------------------------------------------------------------------------------------------------------------

**Table 7.** *Five-fold cross-validated performance on UNSW-NB15, identical pipeline and hyperparameters to the CICIDS-2017 run. The central mechanism replicates: voting fails to beat the best base member, stacking succeeds.*

The diversity-driven mechanism replicates cleanly. Soft voting (weighted F1 = 0.8406) again fails to beat its best member, ExtraTrees (0.8544), whereas stacking (0.8554) beats ExtraTrees (McNemar p = 8.2×10⁻¹⁶; +1.1% relative error reduction, 15.78→ 15.61% error) and the LightGBM reference (0.8130; p \< 10⁻³⁰⁰, +22.0% relative error reduction). The ExtraTrees effect size (1.1%) is smaller than on CICIDS-2017 (6.3%), a difference expected given the greater class overlap in UNSW-NB15; the directional claim is preserved on both benchmarks. Threshold calibration then raises accuracy to 0.8567 and macro-F1 to 0.693 (p \< 10⁻³⁰⁰), again posting the best MCC of all configurations (0.8172). The absolute scores are lower than on CICIDS-2017, as expected: UNSW-NB15 is a deliberately harder benchmark whose Analysis, Backdoor, and DoS classes overlap heavily in feature space. The relative ordering of methods, however, is preserved exactly --- the architectural claim is benchmark-independent.

**Threshold calibration is markedly more impactful on the harder benchmark.** Where CICIDS-2017\'s well-separated classes left little room for decision-layer tuning, UNSW-NB15\'s overlapping classes benefit substantially (Table 8): Analysis precision rose from 0.15 to 0.65 and Backdoor false alarms fell from 1,473 to 208 per 100,000 flows, with Worms and Shellcode improving on both precision and false-alarm rate. Macro-F1 across all classes improved (0.671 → 0.693), while balanced accuracy decreases (0.694 → 0.673) due to the recall-for-precision trade. Practitioners should note the absolute detection count impact: Backdoor true-positive detections fall from approximately 770 to 550 per fold, and Analysis from 1,272 to 992, as the precision improvements come at the cost of recall (Backdoor 0.189→0.135, Analysis 0.272→0.212). Whether this trade is operationally acceptable depends on the priority assigned to these attack classes in the deployment context; the calibration thresholds are adjustable per operator preference, which is precisely the operational flexibility the leak-free calibration mechanism provides.

  -------------------------------------------------------------------------------------------
  **Class**       **Stage**   **Precision**   **Recall**   **F1**    **False alarms / 100k**
  -------------- ----------- --------------- ------------ --------- -------------------------
  Worms            before         0.680         0.789       0.731             26.1

  Worms             after         0.777         0.743       0.760             15.0

  Shellcode        before         0.779         0.861       0.818             149.0

  Shellcode         after         0.831         0.825       0.828             102.8

  Backdoor         before         0.108         0.189       0.137            1472.7

  Backdoor          after         0.378         0.135       0.199             208.5

  Analysis         before         0.151         0.272       0.195            1648.0

  Analysis          after         0.652         0.212       0.320             121.9
  -------------------------------------------------------------------------------------------

**Table 8.** *UNSW-NB15 threshold calibration on the four rarest classes (auto-selected). The benefit is far larger than on CICIDS-2017 because the benchmark\'s class overlap leaves more headroom for decision-layer correction.*

**Explanation-consistency: partial replication.** Under the same-sample protocol, Spearman \\u03C1 = 0.803 (p = 1.4\\u00D710\\u207B\\u2076) and Kendall \\u03C4 = 0.62 on UNSW-NB15 \\u2014 stronger than CICIDS-2017 (\\u03C1 = 0.775). Top-10 feature agreement is 0.9 on both datasets (Table 9). However, top-5 feature agreement falls to 0.4 on UNSW (vs 0.6 on CICIDS), and top-5 and top-10 positional rank agreement are both 0.0 on UNSW (vs 0.6 and 0.3 respectively). The weaker rank metrics on UNSW reflect higher co-linearity among Argus/Zeek flow statistics: connection-count features (ct_src_dport_ltm, ct_dst_src_ltm) dominate EBM contributions but not SHAP attributions, and an additive GAM redistributes these across correlated main effects differently from Shapley values. The result is nuanced: global rank correlation is significant on both benchmarks and corroborates the SHAP explanations delivered to analysts; positional rank agreement does not generalise across feature-extractor families. Practitioners should apply SHAP-vs-glass-box consistency checks per deployment context.

  ------------------------------------------------------------------------
  **Agreement metric**           **UNSW-NB15**         **CICIDS-2017**
  -------------------------- ---------------------- ----------------------
  Top-5 feature agreement             0.40                   0.60

  Top-5 rank agreement                0.00                   0.60

  Top-10 feature agreement            0.90                   0.90

  Top-10 rank agreement               0.00                   0.30

  Spearman ρ                  0.803 (p = 1.4×10⁻⁶)   0.775 (p = 5.5×10⁻⁶)

  Kendall τ                   0.620 (p = 3.6×10⁻⁶)   0.620 (p = 3.6×10⁻⁶)
  ------------------------------------------------------------------------

**Table 9.** *UNSW-NB15 explanation-agreement metrics vs CICIDS-2017. ρ is stronger on UNSW; positional rank agreement does not transfer. Figures U1--U8: ablation, F1, confusion matrix, SHAP, EBM importances, agreement, threshold effect, Pareto. \[Embed FigU1--FigU8.\]*

# 5. Limitations and Future Work

Cross-dataset transfer without retraining remains hard. Although both datasets were evaluated natively (Sections 4.1--4.6 on CICIDS-2017, Section 4.7 on UNSW-NB15), a model trained on one and applied to the other without retraining transfers poorly, because the two feature extractors are largely disjoint --- CICIDS-2017\'s CICFlowMeter computes packet-length-variance and inter-arrival statistics that UNSW-NB15\'s toolchain does not, so transplanted models receive zero-filled inputs for their most informative features. Robust transfer in NIDS remains bottlenecked by the absence of a standardised feature extractor; the present work addresses generalisation by native retraining and by demonstrating that the architecture and its explanation-consistency property hold on both benchmarks, rather than by claiming zero-shot transfer.

Disclosed computational concessions. CICIDS-2017 experiments use a 20% stratified sample; both datasets cap EBM\'s per-fold training at 100,000 rows (its competitive standalone accuracy suggests the cost is modest). The camera-ready release may lift both. Residual minority weakness: on CICIDS-2017, XSS precision (0.34) reflects genuine statistical overlap with benign HTTPS traffic in flow space; on UNSW-NB15, the Analysis and Backdoor classes overlap similarly --- both are boundaries of flow-based detection that only payload-aware (deep-packet-inspection) features could cross. Threshold transferability: tuned thresholds reflect each dataset\'s class prevalence and should be re-calibrated per deployment, which the leak-free calibration-slice procedure makes routine. Categorical encoding: UNSW-NB15\'s proto, service, and state fields were ordinally label-encoded for all members; one-hot or target encoding may shift their attributions. Future work: federated deployment of the glass-box-augmented stack (Rey et al., 2022); adversarial robustness testing; and extending the agreement study to per-class and per-sample (local) explanation consistency.

# 6. Conclusion

This work began from a negative result --- a conventional XGBoost+LightGBM+Random-Forest voting ensemble could not beat its own best member --- and converted it into an architecture: three structurally different learners (CatBoost, ExtraTrees, EBM) combined by stacked generalisation under a leak-proof, fully paired evaluation. The stacked ensemble reaches 99.83% accuracy (weighted F1 0.9984, MCC 0.9951), significantly outperforming its best base member and a strong LightGBM reference (28.1% relative error reduction), where soft voting fails both tests. Embedding the glass-box EBM as a deployed base learner enabled the first explanation-consistency validation in NIDS between exact and post-hoc attributions, with Spearman \\u03C1 = 0.775 (CICIDS-2017) and 0.803 (UNSW-NB15), both significant at p \< 10\\u207B\\u2075, anchoring the system\\u2019s SHAP explanations against ground truth. Top-10 feature agreement is 0.9 on both datasets; positional rank agreement transfers on CICIDS but not on UNSW \\u2014 a boundary condition attributable to higher co-linearity in Argus/Zeek statistics (Section 4.7). Inter-arrival-time features were shown to be the binding constraint on web-attack detection --- worth 4--5× in minority F1, more than any resampling strategy --- and real-prevalence threshold calibration gives operators an inspectable false-alarm dial with a statistically significant net accuracy benefit. At 3.17 µs parallel inference on CICIDS-2017 (15.21 µs on UNSW-NB15), the system meets real-time requirements on commodity hardware. Critically, the diversity-driven mechanism, the stacking advantage, the threshold-calibration benefit, and the explanation-consistency result all replicate under native retraining on UNSW-NB15 --- a benchmark built by a different feature-extraction toolchain --- with SHAP--EBM agreement strengthening to ρ = 0.803, establishing that the contributions generalise across feature-extractor families rather than to one benchmark. Together these results argue that trustworthy network intrusion detection is achieved not by attaching explanations to an accurate black box, but by building ensembles whose accuracy and explainability are validated by the same architecture.

# Declarations

**CRediT author statement.** Khandaker Saifuzzaman: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data Curation, Writing -- Original Draft, Writing -- Review and Editing, Visualization.

**Competing interests.** The author declares no known competing financial interests or personal relationships that could have influenced this work.

**Data availability.** CICIDS-2017 is publicly available from the Canadian Institute for Cybersecurity at https://www.unb.ca/cic/datasets/ids-2017.html.

**Code availability.** \[AUTHOR: INSERT public repository link (GitHub/Zenodo) containing the full pipeline notebook, saved thresholds, and figure-generation code. Q1 venues increasingly require this; the v4 notebook is repository-ready.\]

# References

Al-Omari, M., & Al-Haija, Q. A. (2024). Towards robust IDSs: An integrated approach of hybrid feature selection and machine learning. Journal of Internet Services and Information Security, 14(1). https://doi.org/10.58346/JISIS.2024.I2.004

Arreche, O., Bibers, I., & Abdallah, M. (2024). A two-level ensemble learning framework for enhancing network intrusion detection systems. IEEE Access, 12, 83830--83857. https://doi.org/10.1109/ACCESS.2024.3407029

Arreche, O., Guntur, T., & Abdallah, M. (2024). XAI-IDS: Toward proposing an explainable artificial intelligence framework for enhancing network intrusion detection systems. Applied Sciences, 14(10), 4170. https://doi.org/10.3390/app14104170

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5--32. https://doi.org/10.1023/A:1010933404324

Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106, 249--259. https://doi.org/10.1016/j.neunet.2018.07.011

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321--357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785--794. https://doi.org/10.1145/2939672.2939785

Elkan, C. (2001). The foundations of cost-sensitive learning. Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI), 973--978.

Fitni, Q. R. S., & Ramli, K. (2020). Implementation of ensemble learning and feature selection for performance improvements in anomaly-based intrusion detection systems. 2020 IEEE International Conference on Industry 4.0, Artificial Intelligence, and Communications Technology (IAICT), 118--124. https://doi.org/10.1109/IAICT50021.2020.9172014

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189--1232. https://doi.org/10.1214/aos/1013203451

Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63(1), 3--42. https://doi.org/10.1007/s10994-006-6226-1

Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. Nature, 637(8045), 319--326. https://doi.org/10.1038/s41586-024-08328-6

Hartl, A., Fabini, J., & Zseby, T. (2020). Explainability of machine learning models for network intrusion detection. 2020 IFIP Networking Conference, 1--9.

He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. 2008 IEEE International Joint Conference on Neural Networks (IJCNN), 1322--1328. https://doi.org/10.1109/IJCNN.2008.4633969

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263--1284. https://doi.org/10.1109/TKDE.2008.239

Jairu, P., & Mailewa, A. B. (2022). Network anomaly uncovering on CICIDS-2017 dataset: A supervised artificial intelligence approach. 2022 IEEE International Conference on Electro Information Technology (eIT), 606--615. https://doi.org/10.1109/eIT53891.2022.9813967

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3149--3157.

Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, Z. S., & Lakkaraju, H. (2022). The disagreement problem in explainable machine learning: A practitioner's perspective. arXiv:2202.01602. https://doi.org/10.48550/arXiv.2202.01602

Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 623--631. https://doi.org/10.1145/2487575.2487579

Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17), 1--5.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765--4774.

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). 2015 Military Communications and Information Systems Conference (MilCIS), 1--6. https://doi.org/10.1109/MilCIS.2015.7348942

Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. Sensors, 23(13), 5941. https://doi.org/10.3390/s23135941

Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. arXiv:1909.09223. https://doi.org/10.48550/arXiv.1909.09223

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825--2830.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31, 6639--6649.

Rey, V., Sánchez Sánchez, P. M., Huertas Celdrán, A., & Bovet, G. (2022). Federated learning for malware detection in IoT devices. Computer Networks, 204, 108693. https://doi.org/10.1016/j.comnet.2021.108693

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), 108--116. https://doi.org/10.5220/0006639801080116

Wazid, M., Singh, J., Das, A. K., & Rodrigues, J. J. P. C. (2024). An ensemble-based machine learning-envisioned intrusion detection in Industry 5.0-driven healthcare applications. IEEE Transactions on Consumer Electronics, 70(1), 1903--1912. https://doi.org/10.1109/TCE.2023.3325209

Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241--259. https://doi.org/10.1016/S0893-6080(05)80023-1
