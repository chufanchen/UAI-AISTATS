# UAI-AISTATS

## [Hybrid Reinforcement Learning: Learning Policies from Offline Data and Online Interaction](https://arxiv.org/abs/2505.13768)  
**Authors**: Ruiquan Huang, Donghao Li, Chengshuai Shi, Cong Shen, Jing Yang
**Conference**: AISTATS 2025  
**Tags**: Hybrid RL, Offline RL, Online RL, Sub-optimality Gap, Regret Minimization, Theoretical RL, Concentrability Coefficient

---


### 🧠 Core Idea

This paper introduces a unified hybrid Reinforcement Learning (RL) algorithm, augmenting confidence-based online methods with offline data. This approach fundamentally improves learning by:

*   **Accelerating convergence** for optimal policy identification (achieving lower sub-optimality gap).
*   **Reducing exploration costs** during online interaction (leading to lower cumulative regret).
*   **Revealing a critical insight**: the *ideal* offline data coverage depends distinctly on the learning objective (optimal policy coverage for sub-optimality, diverse sub-optimal policy coverage for regret).

---

### ❓ Problem Statement

#### What problem is the paper solving?

Pure offline and online RL methods have complementary weaknesses:
- **Offline RL**: No exploration, prone to distributional shift.
- **Online RL**: Incurs high exploration costs and "cold-start" performance issues.

The core problem is to design a unified hybrid algorithm that rigorously combines offline data and online interaction to achieve superior performance, overcoming these individual weaknesses. This requires carefully balancing exploration-exploitation, managing uncertainty from offline data (avoiding optimism), and providing robust theoretical guarantees.

---

Okay, let's refine your "Motivation" to include details about previous work and their bounds, referencing the paper's own characterization in Table 1 and Section 4.1.

---

### 🎯 Motivation

While hybrid RL empirically shows promise, combining offline data and online interactions for efficiency, prior theoretical studies have faced limitations:

*   **Limited Unified Understanding**: Existing works often focus on specific settings (e.g., tabular MDPs [Li et al., 2023, Xie et al., 2021b], linear MDPs [Wagenmaker and Pacchiano, 2023], or bandits [Agrawal et al., 2023, Cheung and Lyu, 2024]), lacking a general framework and unified analysis across different RL problems.
*   **Complex/Less General Concentrability**: Previous analyses often rely on specific or complex concentrability coefficients (e.g., $C^*$ [Xie et al., 2021b], $C^*(\sigma)$ [Li et al., 2023], $C_{o2o}$ [Wagenmaker and Pacchiano, 2023], or all-policy coefficients [Tan et al., 2024]), which can be harder to interpret or less general than desired.
*   **Suboptimal or Specific Bounds**:
    *   For **sub-optimality gap**, prior hybrid results typically scale as $\tilde{O}(\sqrt{C_{\text{off}}/(N_0 + N_1) + \sqrt{C_{\text{on}}/N_1}})$ (e.g., as discussed for Li et al. [2023], Tan and Xu [2024]). Pure offline yields $\tilde{O}(\sqrt{C_{cl}/N_0})$, and pure online $\tilde{O}(1/\sqrt{N_1})$.
    *   For **regret minimization**, previous works show bounds like $\tilde{O}(\sqrt{N_1}\sqrt{C_{\text{off}}N_1/N_0} + \sqrt{C_{\text{on}}N_1})$ (e.g., Tan and Xu [2024]), or $\tilde{O}(N_1/\sqrt{N_0/C + N_1})$ in specific bandit cases [Cheung and Lyu, 2024]. Pure online regret is $\tilde{O}(\sqrt{N_1})$.

This work aims to address these gaps by:
*   **Theoretically analyzing** the hybrid setting within a unified framework, providing simpler and tighter bounds.
*   Introducing a **novel, more general concentrability coefficient** $\mathtt{C}(\pi|\rho)$ for clearer analysis.
*   Uncovering a **fundamental separation** in the desired coverage properties of offline data for sub-optimality gap vs. regret minimization, a key insight previously uncharacterized.
*   Achieving **state-of-the-art and order-wise optimal** performance that strictly outperforms pure online/offline methods across both metrics.

---

### 🛠️ Method Overview

This paper introduces a unified hybrid reinforcement learning (RL) framework that combines an offline dataset $\mathcal{D}_0$ of size $N_0$, collected under a behavior policy $\rho$, with online interactions over $N_1$ episodes.

The framework leverages confidence-based online RL algorithms augmented with the offline data to address two key learning objectives: minimizing the sub-optimality gap of the learned policy and minimizing the cumulative online regret.

The core of the proposed framework is an "oracle algorithm," denoted as Alg, which takes a dataset $\mathcal{D}$ as input and outputs an estimated value function $\hat{V}^{\pi}_{\text{Alg}}(\mathcal{D})$ for any policy $\pi$, along with a high-probability uncertainty function $\hat{U}^{\pi}_{\text{Alg}}(\mathcal{D})$ that bounds the estimation error: $\hat{U}^{\pi}_{\text{Alg}}(\mathcal{D}) \geq V^{\pi}_{M^*} - \hat{V}^{\pi}_{\text{Alg}}(\mathcal{D})$ with probability at least $1-\delta$.

The unified algorithm proceeds as follows:

For $t=1, \dots, N_1$ online episodes:

1. Augment the online dataset $\mathcal{D}_{t-1}$ collected so far with the offline dataset $\mathcal{D}_0$ to form $\mathcal{D}_0 \cup \mathcal{D}_{t-1}$.
2. Call the oracle algorithm: $(\hat{V}^{\pi}_{\text{Alg}}, \hat{U}^{\pi}_{\text{Alg}}) \leftarrow \text{Alg}(\mathcal{D}_0 \cup \mathcal{D}_{t-1})$.
3. Select the online policy $\pi_t$ using the optimism-in-face-of-uncertainty principle: $\pi_t = \arg \max_{\pi} \hat{V}^{\pi}_{\text{Alg}} + \hat{U}^{\pi}_{\text{Alg}}$.
4. Execute $\pi_t$ to collect a trajectory $\tau_t$.
5. Update the online dataset: $\mathcal{D}_t = \mathcal{D}_{t-1} \cup \{\tau_t\}$.

For sub-optimality gap minimization, after $N_1$ episodes:

1. Call the oracle algorithm with the full dataset: $(\hat{V}^{\pi}_{\text{Alg}}, \hat{U}^{\pi}_{\text{Alg}}) \leftarrow \text{Alg}(\mathcal{D}_0 \cup \mathcal{D}_{N_1})$.
2. Output the policy $\hat{\pi}$ using the pessimism principle: $\hat{\pi} = \arg \max_{\pi} \hat{V}^{\pi}_{\text{Alg}} - \hat{U}^{\pi}_{\text{Alg}}$.

---

### 📐 Theoretical Contributions

The theoretical analysis relies on two key concepts: the uncertainty level $U_{M^*}(\pi)$ and the concentrability coefficient $C(\pi|\rho) = (U_{M^*}(\pi)/U_{M^*}(\rho))^2$. $U_{M^*}(\pi)$ quantifies the minimum estimation error for $V^{\pi}_{M^*}$ from offline data, while $C(\pi|\rho)$ measures how well the behavior policy $\rho$ covers the target policy $\pi$. Additionally, the oracle algorithm is assumed to satisfy an Eluder-type condition, $\sum_{t=1}^{N_1} \hat{U}^{\pi_t}_{\text{Alg}}(\mathcal{D}_{t-1})^2 \leq C^2_{\text{Alg}}$, bounding the cumulative uncertainty of chosen policies.

#### 1. **Sub-optimality Bound**

The sub-optimality gap of the output policy $\hat{\pi}$ is bounded by $\text{Sub-opt}(\hat{\pi}) = \tilde{O}\left(C_{\text{Alg}}\frac{1}{\sqrt{N_0/C(\pi^*|\rho) + N_1}}\right)$, where $\pi^*$ is the optimal policy. This bound demonstrates that the hybrid approach achieves a performance scaling as if it had $N_0/C(\pi^*|\rho) + N_1$ effective samples, combining the offline and online contributions. A smaller $C(\pi^*|\rho)$ (better coverage of $\pi^*$ by $\rho$) leads to a faster reduction in the sub-optimality gap.

#### 2. **Regret Bound**

The cumulative regret over $N_1$ online episodes is bounded by $\text{Regret}(N_1) = \tilde{O}\left(C_{\text{Alg}}\sqrt{N_1}\sqrt{\frac{N_1}{N_0/C(\pi^{-\epsilon}|\rho) + N_1}}\right)$, where $C(\pi^{-\epsilon}|\rho)$ is the maximum concentrability coefficient over policies $\pi^{-\epsilon}$ whose sub-optimality gap is at least $\epsilon = \tilde{O}(1/\sqrt{N_0+N_1})$. This result shows a significant speed-up factor of $\sqrt{N_1/(N_0/C(\pi^{-\epsilon}|\rho) + N_1)}$ compared to the $\tilde{O}(\sqrt{N_1})$ regret of pure online learning.

A key theoretical insight is the separation between the requirements for minimizing the sub-optimality gap and regret. Sub-optimality gap minimization benefits most from an offline dataset collected by a behavior policy $\rho$ that provides good coverage of the optimal policy $\pi^*$ (small $C(\pi^*|\rho)$). Regret minimization, however, benefits most from a behavior policy $\rho$ that provides good coverage of sub-optimal policies (small $C(\pi^{-\epsilon}|\rho)$). An offline dataset from an optimal policy may not provide sufficient exploration information about sub-optimal policies, potentially leading to higher regret compared to an exploratory behavior policy, even if it yields a better sub-optimality gap.

The paper specializes the framework and analysis to Tabular MDPs and Linear Contextual Bandits, deriving concrete bounds for these settings. For Tabular MDPs, the bounds involve factors related to $|\mathcal{X}|, |\mathcal{A}|, H$. For Linear Contextual Bandits with feature dimension $d$, the bounds depend on $d$. The concentrability coefficients in these settings are shown to relate to known concepts like ratios of occupancy measures or feature covariance matrices.
Lower bounds are also established, demonstrating that any hybrid RL algorithm must incur a sub-optimality gap of $\Omega\left(\frac{1}{\sqrt{N_0/C(\pi^*|\rho) + N_1}}\right)$ and regret of $\Omega\left(\frac{N_1}{\sqrt{N_0/C(\pi^{-\epsilon}|\rho) + N_1}}\right)$. These lower bounds match the derived upper bounds up to logarithmic factors, indicating the proposed framework is order-wise optimal.

---

### 📊 Experiments

Experiments are conducted on:

- **Linear Contextual Bandits**: MovieLens
- **Tabular Finite-Horizon MDPs**: Mountain Car

Goals:
- Compare hybrid vs. pure online and offline algorithms.
- Evaluate:
  - Sub-optimality of final policy,
  - Cumulative regret over episodes,
  - Effects of varying offline data coverage.

**Key result**: Hybrid method consistently outperforms both baselines when offline data has moderate but imperfect coverage. The experiments empirically demonstrate the distinction in desired offline data properties for the two different objectives.

---

### 📈 Key Takeaways

- Hybrid RL algorithms can **strictly dominate** offline or online methods alone  in both sub-optimality gap and online regret, with matching lower bounds.
- New Concentrability Coefficient: A novel coefficient $\mathtt{C}(\pi|\rho)$ quantifies offline data quality more effectively.
- Objective-Dependent Data Needs: Reveals that optimal offline data for minimizing sub-optimality gap (covering optimal policy) is different from that for minimizing regret (covering sub-optimal policies to aid exploration).
- Empirical Validation: Confirms theoretical benefits and the intriguing data-coverage separation across environments.

---

### 📚 Citation

```bibtex
@Article{Huang2025AugmentingOR,
 author = {Ruiquan Huang and Donghao Li and Chengshuai Shi and Cong Shen and Jing Yang},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Augmenting Online RL with Offline Data is All You Need: A Unified Hybrid RL Algorithm Design and Analysis},
 volume = {abs/2505.13768},
 year = {2025}
}
```
