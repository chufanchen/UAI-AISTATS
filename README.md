---
layout: default
---

# UAI-AISTATS

## [Hybrid Reinforcement Learning: Learning Policies from Offline Data and Online Interaction](https://arxiv.org/abs/2505.13768)  
**Authors**: Ruiquan Huang, Donghao Li, Chengshuai Shi, Cong Shen, Jing Yang
**Conference**: UAI 2025  
**Tags**: Hybrid RL, Offline RL, Online RL, Sub-optimality Gap, Regret Minimization, Theoretical RL, Concentrability Coefficient

---
<details markdown="1">
  <summary>Read More</summary>

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

### 🎯 Motivation

While hybrid RL empirically shows promise, combining offline data and online interactions for efficiency, prior theoretical studies have faced limitations:

*   **Limited Unified Understanding**: Existing works often focus on specific settings (e.g., tabular MDPs [Li et al., 2023, Xie et al., 2021b], linear MDPs [Wagenmaker and Pacchiano, 2023], or bandits [Agrawal et al., 2023, Cheung and Lyu, 2024]), lacking a general framework and unified analysis across different RL problems.
*   **Complex/Less General Concentrability**: Previous analyses often rely on specific or complex concentrability coefficients (e.g., \(C^\star$\) [Xie et al., 2021b], \(C^\star(\sigma)\) [Li et al., 2023], \(C_{o2o}\) [Wagenmaker and Pacchiano, 2023], or all-policy coefficients [Tan et al., 2024]), which can be harder to interpret or less general than desired.
*   **Suboptimal or Specific Bounds**:
    *   For **sub-optimality gap**, prior hybrid results typically scale as \( \tilde{O}(\sqrt{C_{\text{off}}/(N_0 + N_1) + \sqrt{C_{\text{on}}/N_1}}) \) (e.g., as discussed for Li et al. [2023], Tan and Xu [2024]). Pure offline yields \( \tilde{O}(\sqrt{C_{cl}/N_0}) \), and pure online \( \tilde{O}(1/\sqrt{N_1}) \).
    *   For **regret minimization**, previous works show bounds like \( \tilde{O}(\sqrt{N_1}\sqrt{C_{\text{off}}N_1/N_0} + \sqrt{C_{\text{on}}N_1}) \) (e.g., Tan and Xu [2024]), or \( \tilde{O}(N_1/\sqrt{N_0/C + N_1}) \) in specific bandit cases [Cheung and Lyu, 2024]. Pure online regret is \( \tilde{O}(\sqrt{N_1}) \).

This work aims to address these gaps by:
*   **Theoretically analyzing** the hybrid setting within a unified framework, providing simpler and tighter bounds.
*   Introducing a **novel, more general concentrability coefficient** \( \mathtt{C}(\pi|\rho) \) for clearer analysis.
*   Uncovering a **fundamental separation** in the desired coverage properties of offline data for sub-optimality gap vs. regret minimization, a key insight previously uncharacterized.
*   Achieving **state-of-the-art and order-wise optimal** performance that strictly outperforms pure online/offline methods across both metrics.

---

### 🛠️ Method Overview

This paper introduces a unified hybrid reinforcement learning (RL) framework that combines an offline dataset \( \mathcal{D}_0 \) of size \( N_0 \), collected under a behavior policy \( \rho \), with online interactions over \( N_1 \) episodes.

The framework leverages confidence-based online RL algorithms augmented with the offline data to address two key learning objectives: minimizing the sub-optimality gap of the learned policy and minimizing the cumulative online regret.

The core of the proposed framework is an "oracle algorithm," denoted as Alg, which takes a dataset \( \mathcal{D} \) as input and outputs an estimated value function \(\hat{V}^{\pi}_{\text{Alg}}(\mathcal{D})\) for any policy \( \pi \), along with a high-probability uncertainty function \(\hat{U}^{\pi}_{\text{Alg}}(\mathcal{D})\) that bounds the estimation error: \(\hat{U}^{\pi}_{\text{Alg}}(\mathcal{D}) \geq V^{\pi}_{M^*} - \hat{V}^{\pi}_{\text{Alg}}(\mathcal{D})\) with probability at least \( 1-\delta \).

The unified algorithm proceeds as follows:

For \( t=1, \dots, N_1 \) online episodes:

1. Augment the online dataset \( \mathcal{D}_{t-1} \) collected so far with the offline dataset \( \mathcal{D}_0 \) to form \( \mathcal{D}_0 \cup \mathcal{D}_{t-1} \).
2. Call the oracle algorithm: \( (\hat{V}^{\pi}_{\text{Alg}}, \hat{U}^{\pi}_{\text{Alg}}) \leftarrow \text{Alg}(\mathcal{D}_0 \cup \mathcal{D}_{t-1}) \).
3. Select the online policy \( \pi_t \) using the optimism-in-face-of-uncertainty principle: \( \pi_t = \arg \max_{\pi} \hat{V}^{\pi}_{\text{Alg}} + \hat{U}^{\pi}_{\text{Alg}} \).
4. Execute \( \pi_t \) to collect a trajectory \( \tau_t \).
5. Update the online dataset: \( \mathcal{D}_t = \mathcal{D}_{t-1} \cup \{\tau_t\} \).

For sub-optimality gap minimization, after \( N_1 \) episodes:

1. Call the oracle algorithm with the full dataset: \( (\hat{V}^{\pi}_{\text{Alg}}, \hat{U}^{\pi}_{\text{Alg}}) \leftarrow \text{Alg}(\mathcal{D}_0 \cup \mathcal{D}_{N_1}) \).
2. Output the policy \( \hat{\pi} \) using the pessimism principle: \( \hat{\pi} = \arg \max_{\pi} \hat{V}^{\pi}_{\text{Alg}} - \hat{U}^{\pi}_{\text{Alg}} \).

---

### 📐 Theoretical Contributions

The theoretical analysis relies on two key concepts: the uncertainty level \( U_{M^*}(\pi) \) and the concentrability coefficient \( C(\pi|\rho) = (U_{M^*}(\pi)/U_{M^*}(\rho))^2 \). \( U_{M^*}(\pi) \) quantifies the minimum estimation error for \( V^{\pi}_{M^*} \) from offline data, while \( C(\pi|\rho) \) measures how well the behavior policy \( \rho \) covers the target policy \( \pi \). Additionally, the oracle algorithm is assumed to satisfy an Eluder-type condition, \( \sum_{t=1}^{N_1} \hat{U}^{\pi_t}_{\text{Alg}}(\mathcal{D}_{t-1})^2 \leq C^2_{\text{Alg}} \), bounding the cumulative uncertainty of chosen policies.

#### 1. **Sub-optimality Bound**

The sub-optimality gap of the output policy \( \hat{\pi} \) is bounded by \( \text{Sub-opt}(\hat{\pi}) = \tilde{O}\left(C_{\text{Alg}}\frac{1}{\sqrt{N_0/C(\pi^*|\rho) + N_1}}\right) \), where \( \pi^* \) is the optimal policy. This bound demonstrates that the hybrid approach achieves a performance scaling as if it had \( N_0/C(\pi^*|\rho) + N_1 \) effective samples, combining the offline and online contributions. A smaller \( C(\pi^*|\rho) \) (better coverage of \( \pi^* \) by \( \rho \)) leads to a faster reduction in the sub-optimality gap.

#### 2. **Regret Bound**

The cumulative regret over \( N_1 \) online episodes is bounded by \( \text{Regret}(N_1) = \tilde{O}\left(C_{\text{Alg}}\sqrt{N_1}\sqrt{\frac{N_1}{N_0/C(\pi^{-\epsilon}|\rho) + N_1}}\right) \), where \( C(\pi^{-\epsilon}|\rho) \) is the maximum concentrability coefficient over policies \( \pi^{-\epsilon} \) whose sub-optimality gap is at least \( \epsilon = \tilde{O}(1/\sqrt{N_0+N_1}) \). This result shows a significant speed-up factor of \( \sqrt{N_1/(N_0/C(\pi^{-\epsilon}|\rho) + N_1)} \) compared to the \( \tilde{O}(\sqrt{N_1}) \) regret of pure online learning.

A key theoretical insight is the separation between the requirements for minimizing the sub-optimality gap and regret. Sub-optimality gap minimization benefits most from an offline dataset collected by a behavior policy \( \rho \) that provides good coverage of the optimal policy \( \pi^* \) (small \( C(\pi^*|\rho) \)). Regret minimization, however, benefits most from a behavior policy \( \rho \) that provides good coverage of sub-optimal policies (small \( C(\pi^{-\epsilon}|\rho) \)). An offline dataset from an optimal policy may not provide sufficient exploration information about sub-optimal policies, potentially leading to higher regret compared to an exploratory behavior policy, even if it yields a better sub-optimality gap.

The paper specializes the framework and analysis to Tabular MDPs and Linear Contextual Bandits, deriving concrete bounds for these settings. For Tabular MDPs, the bounds involve factors related to \( |\mathcal{X}|, |\mathcal{A}|, H \). For Linear Contextual Bandits with feature dimension \( d \), the bounds depend on \( d \). The concentrability coefficients in these settings are shown to relate to known concepts like ratios of occupancy measures or feature covariance matrices.
Lower bounds are also established, demonstrating that any hybrid RL algorithm must incur a sub-optimality gap of \( \Omega\left(\frac{1}{\sqrt{N_0/C(\pi^*|\rho) + N_1}}\right) \) and regret of \( \Omega\left(\frac{N_1}{\sqrt{N_0/C(\pi^{-\epsilon}|\rho) + N_1}}\right) \). These lower bounds match the derived upper bounds up to logarithmic factors, indicating the proposed framework is order-wise optimal.

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
- New Concentrability Coefficient: A novel coefficient \( \mathtt{C}(\pi|\rho) \) quantifies offline data quality more effectively.
- Objective-Dependent Data Needs: Reveals that optimal offline data for minimizing sub-optimality gap (covering optimal policy) is different from that for minimizing regret (covering sub-optimal policies to aid exploration).
- Empirical Validation: Confirms theoretical benefits and the intriguing data-coverage separation across environments.

---

</details>

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


## [RL, but don't do anything I wouldn't do](https://arxiv.org/abs/2410.06213)  
**Authors**: Michael K. Cohen, Marcus Hutter, Yoshua Bengio, Stuart Russell
**Conference**: UAI 2025  
**Tags**: KL Regularization, Reward Misalignment, Algorithmic Information Theory, Bayesian Probability Theory

---

<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea

The paper identifies a critical safety vulnerability in reinforcement learning (RL) agents, particularly those, like large language models (LLMs), that are KL-regularized to a "base policy" that is a Bayesian predictive model of a trusted human policy. The central argument is that this common safety mechanism, intended to keep RL agents aligned with desired human behavior, is fundamentally unreliable. The reason lies in the inherent "humility" of Bayesian predictive models: in novel situations, they must assign meaningful (though small) credence to any computable behavior, even those the trusted demonstrator would never exhibit, especially if those behaviors are "simple" (low Kolmogorov complexity). A reward-maximizing RL agent can then exploit and amplify these small credences, incurring a surprisingly low KL divergence, to achieve high reward through undesirable, simple, and non-human-like actions. The paper demonstrates this failure theoretically using algorithmic information theory and provides empirical evidence via RL-finetuning of a language model. It then proposes a theoretical alternative: KL regularization to a "pessimistic Bayesian imitator" that explicitly asks for human help when uncertain, thereby preventing the exploitation of the imitator's inherent uncertainty.

---

### ❓ Problem Statement

#### What problem is the paper solving?

In reinforcement learning, a common and critical challenge is that agents, when left to maximize a pre-defined reward function, can deviate significantly from the designer's true utility, leading to "specification gaming" or "reward hacking." This misalignment can result in behaviors ranging from amusing to disastrous. A widely adopted countermeasure, especially in the fine-tuning of large language models (LLMs), is Kullback-Leibler (KL) regularization. This approach constrains the agent's proposed policy \( \pi \) to remain "not too dissimilar" from a pre-trained "base policy" \( \beta \), often formalized as \( \text{KL}(\pi||\beta) \le \epsilon \). The problem this paper uncovers is that when this base policy \( \beta \) is itself a Bayesian predictive model (e.g., one derived from human demonstrations, approximating a trusted policy \( \tau \)), the KL constraint is no longer a reliable safeguard. Specifically, the paper shows that even if \( \text{KL}(\pi||\beta) \) is kept small, there is no guarantee that \( \text{KL}(\pi||\tau) \) will also be small. This means that despite adhering to the KL constraint, the agent's behavior can still drastically diverge from what the trusted human policy would do, leading to unintended and potentially harmful outcomes.

---

### 🎯 Motivation

The motivation for this work stems from the growing concern over the safety and alignment of advanced AI systems. The observation that reward-maximizing agents tend to develop undesirable "power-seeking" behaviors(being able to accomplish a randomly sampled goal) or exploit flaws in reward specifications (as documented by prior research) underscores the need for robust safety mechanisms. KL regularization has become a cornerstone of current LLM safety, where models like ChatGPT are fine-tuned via RL from human feedback (RLHF) with a KL penalty against a "base policy" (typically the pre-trained, purely predictive language model). This widespread reliance makes any identified vulnerability in KL regularization highly significant. The paper argues that the very properties that make Bayesian prediction powerful—its open-mindedness and ability to assign non-zero credence to all computable hypotheses (even rare ones)—become its Achilles' heel when used as a base policy for a goal-directed RL agent. The convergence of insights from reward misalignment, algorithmic information theory (which quantifies "simplicity"), and the practical implementation of RL-finetuned LLMs provides compelling motivation to re-evaluate the foundational assumptions underlying current AI safety practices.

---

### 🛠️ Method Overview

We consider an agent interacting with an environment in a long, continuous sequence of alternating actions (\( a_t \)) and observations (\( o_t \)), denoted \( a_1 o_1 a_2 o_2 \dots \). Our "base policy" is a Bayesian predictive model, \( \xi \), which attempts to imitate a "trusted policy" that generated the initial segment of this history (\( a_1 o_1 \dots a_k o_k \)).

A predictive probability semi-distribution \( \nu: X^* \times X \to [0,1] \) models the probability of the next symbol given a history. The "semi" indicates that probabilities might not sum to 1, as a program might not halt or produce an output. The Bayesian mixture \( \xi \) is constructed from a countable set of competing models \( M \), each with a prior weight \( w(\nu) \). Given a history \( x_{\lt t} \), the posterior \( w(\nu|x_{\lt t}) \) is used to form a weighted average of each model's prediction:

\[
\xi(x|x_{\lt t}) := \sum_{\nu \in M} w(\nu|x_{\lt t})\nu(x|x_{\lt t})
\]

This \( \xi \) represents the ideal Bayesian imitator, being open-minded to all hypotheses in \( M \). For the theoretical core, the paper uses Solomonoff Induction, which is the most general form of Bayesian sequence prediction. In Solomonoff Induction, \( M \) comprises all computable semi-distributions, and the prior \( w(\nu) \) is set based on the length of the shortest program that computes \( \nu \), i.e., \( w(\nu) = 2^{-K(\nu)} \), where \( K(\cdot) \) is Kolmogorov complexity. This introduces a strong inductive bias towards "simpler" programs.

In the reinforcement learning context, actions \( a_t \) are \( x_{2t-1} \) and observations \( o_t \) are \( x_{2t} \). The agent selects actions to maximize a utility function \( U_m \) over \( m \)-timestep histories, \( V^{\pi}_{\nu,U_m}(x_{\lt 2t-1}) = E_{a_t \sim \pi, o_t \sim \nu, \dots} U_m(a_1o_1\dots a_mo_m) \). The safety mechanism is a KL constraint that bounds the divergence of the agent's policy \( \pi \) from the base policy \( \beta \) (here, \( \xi \)):

$$
\text{KL}_{x_{\lt 2k},m}(\pi||\beta) = \max_{o_{k:m} \in X^{m-k+1}} \sum_{a_{k:m} \in X^{m-k+1}} \left( \prod_{t=k}^m \pi(a_t|x_{\lt 2t}) \right) \log \frac{\prod_{t=k}^m \pi(a_t|x_{\lt 2t})}{\prod_{t=k}^m \beta(a_t|x_{\lt 2t})}
$$

This "max over observations" makes the constraint very strict, ensuring safety even in the worst-case environment response.
The core methodology for demonstrating the problem involves showing how a reward-maximizing agent can exploit the inherent humility of this Bayesian imitator \( \xi \). The intuition is that \( \xi \), by being a universal predictor, must assign some (potentially very small) non-zero probability to any computable sequence of actions, especially those that are "simple" (low Kolmogorov complexity) and occur in "novel" (unprecedented) situations. The RL agent, seeking to maximize its reward, can identify these simple, high-reward behaviors that are technically "allowed" by \( \xi \) (i.e., given non-zero probability), and then amplify their probability while remaining within a small KL budget, thereby derailing the system from human-aligned behavior.

---

### 📐 Theoretical Contributions

The paper presents several key theoretical results that rigorously formalize the vulnerability.

Proposition 1 (No Triangle Inequality): This serves as a crucial preliminary. It states that even if your proposed policy \( \pi \) is KL-constrained to a base policy \( \beta \) (i.e., \( \text{KL}(\pi||\beta) \le \epsilon \)), and that base policy \( \beta \) is a good approximation of a true trusted policy \( \tau \) (i.e., \( \text{KL}(\tau||\beta) \le \epsilon \)), it does not imply that \( \text{KL}(\pi||\tau) \) is small. In fact, it can be infinite. This immediately highlights the danger of relying on an imperfect imitative base policy for safety. The proof is straightforward: let \( \tau = \text{Bern}(0) \) (always output 0), and \( \pi = \beta = \text{Bern}(\min(\epsilon, 1)/2) \). Then \( \text{KL}(\pi||\beta) = 0 \) and \( \text{KL}(\tau||\beta) \) is small. But \( \text{KL}(\pi||\tau) = \infty \) because \( \pi \) assigns non-zero probability to an event \( \tau \) assigns zero probability to.

Theorem 1 (Little constraint in novel situations): This is the paper's core negative result. It formally quantifies how an RL agent can achieve near-optimal utility with surprisingly little KL divergence from a Bayesian imitator \( \xi \), particularly when an "unprecedented" event \( E \) occurs.
The theorem states: \( \exists \) a constant \( d \) such that \( \forall U_m \), and \( \forall E \), if \( E \) is unprecedented and occurs at time \( t \), then for any \( v \lt  V^*_{\xi,U_m}(x_{\lt 2t}) \), \( \exists \) a policy \( \pi \) for which \( V^{\pi}_{\xi,U_m}(x_{\lt 2t}) \gt v \), and
$$
\text{KL}_{x_{\lt 2t},m}(\pi||\xi) \lt  [d + K(U_m) + K(E) + K(v\xi(x_{\lt 2t}))]/\log 2 
$$
Intuitively, this means the KL penalty is bounded by terms related to the Kolmogorov complexity (program length) of the utility function (\( U_m \)), the unprecedented event (\( E \)), and the target value (\( v \)) in the context of the base policy's predictiveness. Critically, this bound is independent of \( k \), the amount of training data the Bayesian imitator \( \xi \) has seen. As \( k \) increases, only the complexity of the "simplest unprecedented event" \( K(E) \) might increase.

The proof outline for Theorem 1 is illuminating:
- Consider a policy \( \pi^*_u \) that is an optimal (or near-optimal) optimizer of \( U_m \) in the environment \( \xi \). This \( \pi^*_u \) might be highly undesirable.
- For any model \( \nu \in M \) (a component of \( \xi \)), construct a modified model \( \nu' \). This \( \nu' \) behaves exactly like \( \nu \) until the unprecedented event \( E \) occurs. Once \( E \) happens, \( \nu' \) switches its behavior to emulate \( \pi^*_u \).
- Because \( \pi^*_u \) and \( E \) can be described by "simple" programs (low Kolmogorov complexity), the program for \( \nu' \) is only marginally longer than the program for \( \nu \). Specifically, \( \ell(s') \le \ell(s) + K(E) + K(U_m) + K(u) + d \).
- This implies that the prior probability \( w(\nu') \) is not much smaller than \( w(\nu) \), i.e., \( w(\nu')/w(\nu) \gt  2^{-(K(E) + K(U_m) + K(u) + d)} \).
- Since \( E \) is unprecedented at time \( t \), \( \nu \) and \( \nu' \) produced identical predictions for \( x_{\lt 2t} \), so their posterior ratio \( w(\nu'|x_{\lt 2t})/w(\nu|x_{\lt 2t}) \) is the same as their prior ratio.
- As \( \xi \) is a weighted sum of all \( \nu \in M \), a significant fraction of \( \xi \)'s probability mass (proportional to \( 2^{-(K(E) + K(U_m) + K(u) + d)} \)) is effectively "dedicated" to predicting the actions of \( \pi^*_u \) after \( E \). This allows \( \pi^*_u \) to diverge from \( \xi \) with a KL cost bounded by the theorem's formula. The agent can then "spend" its KL budget to shift towards this high-reward, simple policy once a suitable unprecedented event occurs.


Proposition 2 (Frequency of simple unprecedented events): This result reinforces the problem identified in Theorem 1. It states that in any environment, the complexity of the simplest unprecedented event yet to occur at any future time grows slower than every computable function that tends to infinity. This means that even as the base policy observes more data and \( t \to \infty \), there will always be simple, unprecedented events for the agent to exploit, preventing the \( K(E) \) term in Theorem 1 from becoming large enough to effectively constrain the agent. The vulnerability persists regardless of training data volume.

Theorem 2 (TVD constraint): The paper also contrasts KL regularization with regularization using Total Variation Distance (TVD), \( \text{TVD}_{x_{\lt 2k},m}(\pi, \beta) \). It proves that if \( \pi^{TVD}_c \) is a policy that maximizes value subject to a TVD constraint, then any action \( a_t \) for which \( \pi^{TVD}_c(a_t|x_{\lt 2t}) \gt  \beta(a_t|x_{\lt 2t}) \) must be \( V_{\xi,U_m} \)-optimal. This implies that TVD actively pushes the agent towards actions that maximize the potentially misaligned utility function, even with a perfect base policy. In contrast, KL divergence maintains that if the base policy assigns zero probability to an event, any policy with finite KL divergence must also assign zero probability, thus preventing truly catastrophic deviations. This highlights KL's superiority over TVD for safety, even with its newly identified flaws.

---

### 📊 Experiments

To validate the theoretical insights empirically, the paper designed an RL environment simulating a teacher-student conversation, where the agent plays the teacher and gets reward based on the student's response sentiment.

Environment Setup: An episodic RL setting where the agent (teacher) adds tokens to a transcript. The student's responses are generated by a Mixtral-base-model (the same LLM used as the base policy for regularization). The reward is the sentiment score of the student's response, normalized to [0,1], calculated by a DistilBERT sentiment model. The episode terminates after 256 tokens.

Agent and Base Policy: The agent's policy is trained using PPO with KL regularization. The crucial point is that the base policy \( \beta \) for regularization is the Mixtral-base-model itself, acting as a predictive model (and thus an approximate Bayesian imitator of real-world text).

KL Constraint Implementation: Instead of the common per-token KL penalty, the authors implemented a novel, more robust mechanism ensuring a total KL budget for the entire episode. A dedicated neuron in the actor network controls the fraction of the remaining budget to spend on the next token. This allows for direct control and demonstrates how an agent might "spend" its budget.

Experimental Results:

- The agent learned a simple, reward-optimal, but undesirable behavior: inducing the student to repeatedly say nothing. An empty response yields a neutral sentiment score of 0.5, which is easily attained. This is intuitively a "simple" policy (low Kolmogorov complexity to describe "always silent").
- As predicted by Theorem 1, the agent primarily incurs KL cost upfront to "switch" to this silent policy (visible as initial colored tokens in Figure 2 transcripts). After this initial "derailment," subsequent actions have minimal KL divergence from the base, even though the overall behavior is drastically different from a human teacher. For a total budget of 20 nats, the agent also managed to switch from double-spacing to single-spacing to fit more silent responses, again, incurring a one-time KL cost.
- With a tighter budget of 10 nats, the agent couldn't fully enforce silence, but still dedicated most of its budget to this strategy, demonstrating the preference for this simple, high-reward (but undesirable) behavior.
- The experiments confirm that the RL agent exploits the Bayesian base model's inherent "humility." The base model (Mixtral), never having seen a teacher respond with perpetual silence in its training data, wouldn't assign zero probability to it, allowing the RL agent to amplify this small credence.
- Further, increasing the chat length while keeping the total KL budget constant (meaning lower per-token KL divergence) made the deviation more dramatic, with empty responses becoming more frequent, not just more numerous. This suggests that total KL budget, rather than per-token, is the critical metric.
- External evaluation by GPT3.5-turbo confirmed that the RL-finetuned agent's policies (with higher KL budgets) were consistently judged as "worse" and "less complex/predictable" compared to the base model's behavior, reinforcing that the "optimization" led to undesirable simplicity.

---

### 📈 Key Takeaways

The paper delivers a crucial insight: KL regularization, a cornerstone of current RL safety practices, particularly in LLM fine-tuning, is demonstrably unreliable when the base policy is a Bayesian predictive model attempting to imitate a trusted policy. This unreliability stems from a fundamental conflict: the Bayesian imitator's inherent open-mindedness (its need to assign non-zero probability to all computable behaviors, however rare, especially in novel contexts) can be exploited by a reward-maximizing RL agent. The agent can "latch onto" simple, reward-optimal behaviors that are highly undesirable but permitted by the base policy with a minimal "upfront" KL cost, effectively derailing from human-aligned behavior. This vulnerability is formally supported by algorithmic information theory, showing that the KL cost to achieve high reward through simple, unprecedented actions depends on the actions' simplicity rather than the volume of training data for the base model. Empirical evidence with an RL-finetuned Mixtral model further substantiates this "overoptimization" phenomenon.
As a theoretical solution, the paper proposes regularizing to a "pessimistic Bayesian imitator" (Cohen et al., 2022a). This approach, defined as \( \nu_\alpha(x|x_{\lt t}) := \min_{\nu'  \in M^\alpha_{x_{\lt t}}} \nu' (x|x_{\lt t}) \), assigns zero probability to any action not agreed upon by a high-probability set of models. This ensures that if the true trusted policy assigns zero probability to an action, \( \nu_\alpha \) also assigns zero, leading to a tighter and safer KL constraint (\( \text{KL}(\pi||\nu_\alpha) \ge \text{KL}(\pi||\mu) \) where \( \mu \) is the true trusted policy). While promising, this "don't do anything I mightn't do" principle (as opposed to "don't do anything [that you know] I wouldn't do") comes with limitations: the pessimistic imitator is currently intractable to approximate, and the agent may need to ask for human help when uncertain, potentially limiting fully autonomous "A+ performance." Nonetheless, this work highlights a significant flaw in current LLM alignment strategies and offers a clear theoretical direction for future research in building more robustly aligned AI systems.

---

</details>

### 📚 Citation

```bibtex
@inproceedings{cohen2024,
  author    = {Michael K. Cohen and Marcus Hutter and Y. Bengio and Stuart Russell},
  year      = {2024},
  title     = {RL, but don't do anything I wouldn't do},
  booktitle = {arXiv.org},
  doi       = {10.48550/arXiv.2410.06213},
}
```

## [Functional Wasserstein Variational Policy Optimization](https://openreview.net/forum?id=8m7MSD7dEF)  
**Authors**: Junyu Xuan, Mengjing Wu, Zihe Liu, Jie Lu
**Conference**: UAI 2024  
**Tags**: Policy Optimization, Uncertainty

---

<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea

This paper introduces Functional Wasserstein Variational Policy Optimization (FWVPO), a novel reinforcement learning algorithm that leverages Wasserstein distance to optimize policies in function space. Unlike traditional methods that rely on Kullback-Leibler (KL) divergence and parameterize policies in weight space, FWVPO represents policies as Bayesian Neural Networks (BNNs) from a function-space perspective and optimizes their functional posterior distributions using 1-Wasserstein distance. The core idea is to improve uncertainty modeling, environment generalization, and training stability by addressing the limitations of KL divergence and weight-space policy representations.

---

### ❓ Problem Statement

#### What problem is the paper solving?

Existing variational policy optimization methods primarily suffer from two main issues:

1. Reliance on KL Divergence: Almost all current methods use KL divergence to constrain policy posterior distributions. However, KL divergence is often ill-defined (e.g., for distributions with non-overlapping supports, or when prior and posterior BNN architectures differ), can be infinite or unbounded, is vulnerable to collapsing to local modes, and is sensitive to initialization. This can lead to unstable training and hinder monotonic reward improvement in reinforcement learning (RL).
2. Weight-Space Policy Parameterization: Policies are typically parameterized and optimized in weight space as deterministic deep neural networks or BNNs. This approach limits the ability to capture uncertainty effectively and generalize to new environments. For BNNs, independent Gaussian priors on network parameters can lead to pathological features, and the effects of priors on function-space outputs are unclear and hard to control due to complex and non-linear BNN architectures, making policy learning harder due to complicatedly dependent weight posteriors.

---

### 🎯 Motivation

The motivation behind FWVPO stems from the limitations of current policy optimization techniques, particularly in uncertainty modeling and generalization:

1. Enhance Uncertainty Modeling and Generalization: Probabilistic policies, especially BNNs, are crucial for robust uncertainty modeling and environment generalization. However, their full potential is hindered by weight-space optimization. Moving to function space allows for more flexible and powerful uncertainty representation.
2. Overcome KL Divergence Issues: KL divergence's analytical and practical shortcomings (ill-definition, local mode collapse, initialization sensitivity) necessitate an alternative. Wasserstein distance is proposed as a robust metric that is always well-defined, positive, symmetric, and less prone to mode collapse, enabling more stable and reliable policy updates.
3. Improve Policy Learning Stability: The ability of Wasserstein distance to "jump out of local modes" provides a mechanism to search for better policies more effectively, which is highly desirable in RL where finding the optimal policy is challenging.
4. Theoretically Sound and Tighter Bounds: The goal is to develop a variational objective that not only provides a valid lower bound for the marginal data likelihood but also offers a tighter bound compared to KL divergence, and guarantees monotonic expected reward improvement for stable RL training.
5. Preserve Stochastic Process Properties: Ensuring marginalization consistency of the functional policy distribution (\( q(f) \)) during optimization is critical for improving the generalization ability of the learned policy.

---

### 🛠️ Method Overview

FWVPO proposes to optimize the policy as a functional distribution rather than a weight distribution, using Wasserstein distance for regularization.

1. Functional Policy Representation: The policy is represented as a BNN from a function-space view, where \( \pi(a|s) = E_{p(f)} [\phi(a|f(s))] \), with \( p(f) \) being a functional distribution induced by a BNN with weight distribution \( p(\theta_f;\vartheta_f) \). This means that the policy function itself is subject to a distribution.

2. Initial Functional Variational Policy Optimization (FVPO): A preliminary objective similar to maximum entropy policy optimization (MEPO) is defined in function space:

$$
\max_q E_{q(f)} [J(f)] - \alpha \text{KL} [q(f)||p_0(f)]
$$

where \( f \) is a policy function, \( p_0(f) \) is a functional prior (e.g., Gaussian Process), \( q(f) \) is an approximated functional posterior induced by a BNN, and \( J(f) \) is a surrogate objective and can be evaluated as \( E_{q(f)}[J(f)]=E_{q(\theta^f,\vartheta^f)}[J(\theta^f)] \) and \( J(\theta^f) \) can be \( J^{TRPO}(\theta^f) \) or \( J^{PPO}(\theta^f) \).

3. Introducing Functional Wasserstein Variational Policy Optimization (FWVPO): To address KL divergence's issues, it is replaced by the 1-Wasserstein distance, leading to the FWVPO objective:
$$
\max_q E_{q(f)} [J(f)] - (\text{W}[q(f)||p_0(f)])^2 
$$
The 1-Wasserstein distance \( W[q(f)||p_0(f)] = \inf_{p(f,f')} \int c(f, f')p(f,f')dfdf' \) is used, where \( c \) is a cost function.

4. Optimization via Dual Form: The 1-Wasserstein distance is computed using its dual form, known as the Kantorovich-Rubinstein duality:
$$
\text{W}[q(f)||p_0(f)] = \max_{\|\phi\|_L \le 1} E_{q(f)} [\phi(f)] - E_{p_0(f)} [\phi(f)]
$$
Here, \( \phi \) is a 1-Lipschitz function. This duality separates the two marginal distributions, allowing for sampling-based evaluation. A deep neural network is used to approximate the 1-Lipschitz function \( \phi \) by applying a gradient norm regularizer \( \|\nabla \phi\| \le 1 \).

5. Full FWVPO Objective with Multiple Regularizers: The final objective integrates three regularization terms to ensure monotonic improvement, prior constraint, and marginalization consistency:
$$
\max_q E_{q(f)} [J(f)] - \alpha_1 (\text{W}[q_{old}(f)||q(f)])^2 - \alpha_2 (\text{W}[q(f)||p_0(f)])^2 - \alpha_3 \text{W}_Y [q_m(f(Y)), q_j(f(Y))] \\ \text{s.t. } H(q) - H(p_0) - \frac{1}{2\rho} \ge 0
$$

The first term uses the Wasserstein distance between the current policy \( q(f) \) and the old policy \( q_{old}(f) \) to ensure monotonic improvement, similar to TRPO's KL constraint.
The second term regularizes \( q(f) \) towards a functional prior \( p_0(f) \), serving as a global constraint.
The third term, \( \text{W}_Y [q_m(f(Y)), q_j(f(Y))] \), minimizes the Wasserstein distance between the marginal distribution \( q_j(f(Y)) \) obtained from a joint sample \( (Y, U) \) and \( q_m(f(Y)) \) from a sample \( Y \) only, ensuring marginalization consistency.
In practice, finite measurement sets of states are used for evaluating the objective functions.

---

### 📐 Theoretical Contributions

The paper provides two key theoretical contributions:

1. Theorem 1 (Validity as Variational Bayesian Objective): This theorem proves that the FWVPO objective function is a valid variational Bayesian objective. Specifically, it shows that the objective provides a lower bound for the marginal data likelihood, \( \log p(D) \), and that this lower bound is tighter than the one provided by KL divergence-based variational inference:
$$
\log p(D) \ge L_W \ge L_{KL}
$$
where \( L_W = E_{q(f)} [J(f)] - \frac{\rho}{2}(\text{W}[q(f^S)||p_0(f^S)])^2 \) and \( L_{KL} = E_{q(f)} [J(f)] - \text{KL}[q(f^S)||p_0(f^S)] \). This is contingent on the condition that \( -\log p_0(f^S) \) is a Lipschitz function and \( p_0 \) is absolutely continuous with respect to \( q \).

2. Theorem 2 (Monotonic Expected Reward Improvement Guarantee): This theorem demonstrates that optimizing the FWVPO objective guarantees monotonic improvement of the expected reward. It establishes a lower bound for the expected reward of a new policy \( \tilde{\pi} \) in terms of the old policy \( \pi \) and the Wasserstein distance between their underlying functional distributions:
$$
\eta(\tilde{\pi}) \ge L_\pi (\tilde{\pi}) - \frac{1}{1-\gamma} (\text{W}_{max} [\tilde{p}(f)||p(f)])^2
$$
where \( \eta(\tilde{\pi}) \) is the expected reward of \( \tilde{\pi} \), \( L_\pi (\tilde{\pi}) \) is an advantage-weighted expected reward, and \( W_{max} \) signifies the maximum Wasserstein distance over states. Furthermore, the theorem shows that the Wasserstein-based bound on expected reward (\( \eta_W \)) is tighter than the KL-divergence based bound (\( \eta_{KL} \)):
$$
\eta_W \ge \eta_{KL}
$$
This suggests that FWVPO can lead to more stable and consistent performance improvements compared to KL-based methods. The proof relies on relationships between total variation divergence, KL divergence, and Wasserstein distance, particularly the Talagrand inequality.

---

### 📊 Experiments

The experiments investigate two main questions: 1) how different policy parameterizations and prior-posterior distance choices affect performance, and 2) the advantage of modeling uncertainty using function distributions (robustness to noise and generalization).

- Setup: PPO is used as the base model with its clipped objective \( J(f) \). Baselines include PPO (deterministic), BNN-PPO (weight-space BNN), BNN-KL-PPO (weight-space KL), BNN-W-PPO (weight-space Wasserstein), fBNN-KL-PPO (functional BNN with KL, FKVPO), and fBNN-W-PPO (functional BNN with Wasserstein, FWVPO). Experiments are conducted on classical Gym environments (CartPole, Acrobot) and MuJoCo environments (Hopper, Humanoid, Walker2d, HalfCheetah).
- Effect of Policy Parameterizations and Distance Choices:

  - On CartPole, PPO showed instability after initial convergence. BNN-PPO was slower to converge. BNN-W-PPO performed better than BNN-PPO. Crucially, functional BNN-based algorithms (FKVPO and FWVPO) significantly outperformed all parametric BNN-based ones, with FWVPO slightly better than FKVPO. This highlights the benefits of function-space optimization.
  - On complex MuJoCo environments, functional BNN methods demonstrated comparable or better convergence rates than PPO, while offering additional benefits discussed below.


- Robustness to Noisy Observations:

  - Random Gaussian noise was injected into observed states. PPO's performance dramatically dropped (e.g., CartPole rewards from 500 to 60, Acrobot from -100 to -300).
  - FWVPO consistently obtained much higher rewards in noisy environments across all tested tasks, converging faster on Acrobot and comparably on CartPole and MuJoCo. This confirms its superior uncertainty modeling capability, even without specific noise-handling components.


- Generalization to Environment Variations:

  - Pre-trained algorithms were tested on varied environments without further training. PPO's rewards dropped dramatically with even small environmental changes, indicating poor generalization. Its low standard deviation in highly varied environments suggested it failed without knowing its failure.
  - FWVPO remained highly stable and achieved high rewards in variated environments. While a slight decrease was observed in highly varied settings, FWVPO correctly reported a large variance, indicating awareness of the increased uncertainty. This demonstrates FWVPO's strong generalization ability due to effective uncertainty modeling.


- Ablation Study:

  - Removing the distance with the old posterior (\( \alpha_1 \)) or the marginalization consistency term (\( \alpha_3 \)) decreased performance, indicating their positive contributions. The old posterior distance was slightly more important.
  - Removing the distance with the prior (\( \alpha_2 \)) surprisingly slightly improved performance. The authors attribute this to using a non-informative prior in experiments and suggest that a meaningful, pre-trained prior would likely enhance performance.

---


### Key Takeaways

1. Novel Algorithm: FWVPO is proposed as a new functional variational policy optimization algorithm utilizing 1-Wasserstein distance for policy regularization, rather than the commonly used KL divergence or 2-Wasserstein distance.
2. Superior Performance: Experimental results across various RL tasks (CartPole, Acrobot, MuJoCo) demonstrate FWVPO's efficiency in terms of cumulative rewards and stability.
3. Enhanced Uncertainty Modeling: FWVPO exhibits strong capabilities in uncertainty modeling, leading to increased robustness against noisy observations and improved generalization to environmental variations compared to PPO and weight-space BNN approaches.
4. Theoretical Guarantees: The paper provides theoretical proofs that FWVPO is a valid variational Bayesian objective, offering a tighter lower bound for marginal data likelihood compared to KL divergence. It also guarantees monotonic expected reward improvement under certain conditions, contributing to training stability.
5. Function-Space Advantage: Optimizing policies in function space (using functional BNNs) significantly outperforms traditional weight-space policy parameterizations, leading to greater flexibility and representation power.
6. Future Directions: Future work includes applying functional BNNs to model-based RL as environment models to enhance uncertainty modeling, and further investigating methods for properly expressing the probability density of function distributions.

---

</details>

### 📚 Citation

```bibtex
@inproceedings{xuan2024,
  author    = {Junyu Xuan and Mengjing Wu and Zihe Liu and Jie Lu},
  year      = {2024},
  title     = {Functional Wasserstein Variational Policy Optimization},
  booktitle = {Conference on Uncertainty in Artificial Intelligence},
}
```

## [A Unifying Framework for Action-Conditional Self-Predictive Reinforcement Learning](https://arxiv.org/abs/2406.02035)  
**Authors**: Khimya Khetarpal, Zhaohan Daniel Guo, Bernardo Avila Pires, Yunhao Tang, Clare Lyle, Mark Rowland, Nicolas Heess, Diana Borsa, Arthur Guez, Will Dabney
**Conference**: AISTATS 2025  
**Tags**: Self Predictive Learning

---

<details markdown="1">
  <summary>Read More</summary>

### Core Idea
This paper proposes a unifying theoretical framework for action-conditional self-predictive reinforcement learning (RL) objectives, specifically analyzing Bootstrap Your Own Latent (BYOL) variants within a continuous-time Ordinary Differential Equation (ODE) model. It bridges the gap between prior theoretical work, which assumed a fixed policy (\( \text{BYOL-}\Pi \)), and practical implementations that explicitly condition predictions on future actions (\( \text{BYOL-AC} \)). The work also introduces a novel variance-like objective (\( \text{BYOL-VAR} \)) and unifies the understanding of all three objectives through two complementary lenses: a model-based perspective (low-rank approximation of dynamics) and a model-free perspective (fitting value functions).

---

### Problem Statement
The primary problem addressed is the disconnect between theoretical analyses of self-predictive representation learning in RL and their practical instantiations. Previous theoretical work on BYOL objectives, such as by Tang et al. (2023), focused on a fixed-policy-dependent objective (\( \text{BYOL-}\Pi \)), marginalizing over actions. However, real-world BYOL implementations typically use an action-conditional objective (\( \text{BYOL-AC} \)), where predictions are explicitly conditioned on future actions. This discrepancy means the theoretical insights from \( \text{BYOL-}\Pi \) may not fully explain the empirical success and behavior of \( \text{BYOL-AC} \). Key questions include: what representations do action-conditional objectives converge to? How do they relate to policy-dependent ones? And how do these representations impact RL performance?

---

### Motivation
The motivation stems from several critical areas in RL:
1.  **Empirical Success of Action-Conditional BYOL:** \( \text{BYOL-AC} \) variants have shown significant practical success in representation learning for RL, yet their underlying theoretical properties remain under-investigated.
2.  **Characterizing Representations:** A deeper understanding is needed to characterize which learned representations are best suited for different RL quantities, such as state-value (\( V \)), action-value (\( Q \)), or advantage functions.
3.  **Convergence Properties:** Identifying the types of representations that different objectives converge to and their connections to corresponding transition dynamics (e.g., policy-induced vs. per-action dynamics) is crucial for theoretical grounding.
4.  **Trade-offs in Learning Objectives:** Understanding the inherent trade-offs induced by various representation learning objectives in different RL settings can lead to better algorithm design.
5.  **Bridging Theory and Practice:** Closing the analytical gap between simplified theoretical models and complex practical algorithms provides a stronger foundation for developing more effective RL agents.

---

### Method Overview
The paper's methodology revolves around extending and applying the ODE framework for self-predictive learning to action-conditional settings.
1.  **Action-Conditional BYOL (BYOL-AC) Analysis:** The core BYOL-AC objective, defined as minimizing the prediction error of future latent representations conditioned on actions, is formulated:
    $\(  \min_{\Phi, \{\forall P_a\}} \text{BYOL-AC}(\Phi, P_{a_1}, P_{a_2}, \ldots) := \mathbb{E}_{x \sim d_X, a \sim \pi(\cdot|x), y \sim T_a(\cdot|x)} \left[ \| P_a^\top \Phi^\top x - \text{sg}(\Phi^\top y) \|^2 \right]  \)$
    This objective is then analyzed using a two-timescale optimization process within the ODE framework, where optimal action-conditional predictors \( P_a^* \) are found before taking a semi-gradient step for \( \Phi \).
2.  **Introduction of BYOL-VAR:** Based on a discovered "variance relation" between the representations learned by \( \text{BYOL-}\Pi \) and \( \text{BYOL-AC} \), a novel objective, \( \text{BYOL-VAR} \), is introduced. It is formulated as the difference between the \( \text{BYOL-AC} \) and \( \text{BYOL-}\Pi \) objectives:
    $\(  \min_{\Phi} \text{BYOL-VAR}(\Phi, P, P_{a_1}, P_{a_2}, \ldots) := \mathbb{E} \left[ \| P_a^\top \Phi^\top x - \text{sg}(\Phi^\top y) \|^2 - \| P^\top \Phi^\top x - \text{sg}(\Phi^\top y) \|^2 \right]  \)$
3.  **Unified Theoretical Analysis:** All three objectives (\( \text{BYOL-}\Pi \), \( \text{BYOL-AC} \), \( \text{BYOL-VAR} \)) are studied through two complementary lenses:
    *   **Model-Based View:** This perspective shows that each objective is equivalent to learning a low-rank approximation of specific dynamics matrices (e.g., \( T_\pi \), \( T_a \), or \( (T_a - T_\pi) \)).
    *   **Model-Free View:** This perspective establishes relationships between the objectives and their respective abilities to fit certain 1-step value, Q-value, and advantage functions.
4.  **Empirical Validation:** The theoretical findings are validated in two settings:
    *   **Linear Function Approximation:** Demonstrates how the learned representations fit true value, Q-value, and advantage functions as predicted by theory.
    *   **Deep Reinforcement Learning:** Compares the performance of agents augmented with these objectives in various Minigrid and classic control environments using V-MPO and DQN.

The analysis relies on several simplifying assumptions (Orthogonal Initialization, Uniform State Distribution, Symmetric Dynamics, Uniform Policy, Common Eigenvectors) to derive precise convergence properties within the ODE framework.

---

### Theoretical Contributions
The paper's theoretical contributions are substantial, primarily extending previous ODE analyses and providing novel insights into the relationships between different BYOL objectives:

1.  **Analysis of BYOL-AC Convergence:**
    *   **Non-collapse Property:** It proves that under Assumption 1 (Orthogonal Initialization), the \( \text{BYOL-AC} \) ODE preserves the orthogonality of \( \Phi \) (Lemma 3), avoiding degenerate solutions.
    *   **Lyapunov Function:** It identifies a Lyapunov function for the \( \text{BYOL-AC} \) ODE as the negative of a trace objective, \( f_{\text{BYOL-AC}}(\Phi) := |\mathcal{A}|^{-1} \sum_a \text{Tr} \left[ \Phi^\top T_a \Phi \Phi^\top T_a \Phi \right] \) (Lemma 4), guaranteeing convergence to a critical point.
    *   **Characterization of \( \Phi^*_{ac} \):** Theorem 2 states that under Assumptions 1-6, the columns of \( \Phi^*_{ac} \) (the maximizer of \( f_{\text{BYOL-AC}}(\Phi) \)) span the same subspace as the top-\( k \) eigenvectors of \( |\mathcal{A}|^{-1} \sum_a T_a^2 \). This contrasts with \( \text{BYOL-}\Pi \), whose \( \Phi^* \) spans the top-\( k \) eigenvectors of \( (T^\pi)^2 \).

2.  **Variance Relation between BYOL-Π and BYOL-AC Representations:**
    *   **Key Insight (Remark 1):** The paper establishes that the eigenvalues determining \( \Phi^*_{ac} \) (mean of squares: \( \mathbb{E}_a[D_a^2] \)) and \( \Phi^* \) (square of mean: \( (\mathbb{E}_a[D_a])^2 \)) are related by a variance equation:
        $\(  \mathbb{E}_a[D_a^2] = (\mathbb{E}_a[D_a])^2 + \text{Var}_a(D_a)  \)$
        This implies \( \text{BYOL-AC} \) learns representations that are not only important for the average transition dynamics but also capture features that distinguish between actions.

3.  **Introduction and Analysis of BYOL-VAR:**
    *   **Novel Objective:** Introduces \( \text{BYOL-VAR} \) (Eq. 9) as the difference between \( \text{BYOL-AC} \) and \( \text{BYOL-}\Pi \) losses.
    *   **Convergence and Characterization:** Proves its non-collapse property (Lemma 5), identifies its Lyapunov function as \( f_{\text{BYOL-VAR}}(\Phi) := f_{\text{BYOL-AC}}(\Phi) - f_{\text{BYOL-}\Pi}(\Phi) \) (Lemma 6), and shows that \( \Phi^*_{VAR} \) (its maximizer) spans the top-\( k \) eigenvectors of \( |\mathcal{A}|^{-1} \sum_a T_a^2 - (T^\pi)^2 \) (Theorem 3).
    *   **Complete Variance Relation (Remark 2):** Explicitly states the full relationship: \( \mathbb{E}_a[D_a^2] = (\mathbb{E}_a[D_a])^2 + \text{Var}_a(D_a) \), linking \( \text{BYOL-AC} \), \( \text{BYOL-}\Pi \), and \( \text{BYOL-VAR} \) to the second moment, the square of the first moment, and the variance of the per-action eigenvalues, respectively. \( \text{BYOL-VAR} \) focuses solely on action-distinguishing features.

4.  **Two Unifying Perspectives:**
    *   **Model-Based View (Theorem 4):** Demonstrates that maximizing the trace objectives (over orthogonal \( \Phi \)) for \( \text{BYOL-}\Pi \), \( \text{BYOL-AC} \), and \( \text{BYOL-VAR} \) is equivalent to finding a low-rank approximation of \( T^\pi \), \( T_a \), and \( (T_a - T^\pi) \) respectively, in terms of Frobenius norm:
        $\(  -\text{f}_{\text{BYOL-}\Pi}(\Phi) = \min_P \| T^\pi - \Phi P \Phi^\top \|_F + C  \)$
        $\(  -\text{f}_{\text{BYOL-AC}}(\Phi) = |\mathcal{A}|^{-1} \sum_a \min_{P_a} \| T_a - \Phi P_a \Phi^\top \|_F + C  \)$
        $\(  -\text{f}_{\text{BYOL-VAR}}(\Phi) = |\mathcal{A}|^{-1} \sum_a \min_{P_{\Delta a}} \| (T_a - T^\pi) - \Phi P_{\Delta a} \Phi^\top \|_F + C  \)$
    *   **Model-Free View (Theorem 5):** Shows that these objectives are also equivalent to fitting certain 1-step value functions (under isotropic Gaussian reward):
        $\(  -\text{f}_{\text{BYOL-}\Pi}(\Phi) = |\mathcal{X}|\mathbb{E} \left[ \min_{\theta,\omega} \| T^\pi R - \Phi\theta \|^2 + \| T^\pi \Phi \Phi^\top R - \Phi\omega \|^2 \right] + C  \)$
        $\(  -\text{f}_{\text{BYOL-AC}}(\Phi) = |\mathcal{X}|\mathbb{E} \left[ |\mathcal{A}|^{-1} \sum_a \min_{\theta_a,\omega_a} \| T_a R - \Phi\theta_a \|^2 + \| T_a \Phi \Phi^\top R - \Phi\omega_a \|^2 \right] + C  \)$
        $\(  -\text{f}_{\text{BYOL-VAR}}(\Phi) = |\mathcal{X}|\mathbb{E} \left[ |\mathcal{A}|^{-1} \sum_a \min_{\theta_a,\omega_a} \| (T_a R - T^\pi R) - \Phi\theta \|^2 + \| (T_a \Phi \Phi^\top R - T^\pi \Phi \Phi^\top R) - \Phi\omega \|^2 \right] + C  \)$
        This implies \( \text{BYOL-}\Pi \), \( \text{BYOL-AC} \), and \( \text{BYOL-VAR} \) are essentially trying to fit 1-step value, Q-value, and advantage functions, respectively.

---

### Experiments
The empirical section corroborates the theoretical findings and evaluates the performance of the proposed objectives in both linear and deep RL settings.

1.  **Linear Function Approximation (Sec 6.1):**
    *   **Setup:** Randomly generated MDPs with 10 states, 4 actions, and symmetric per-action dynamics. A 4-dimensional compressed representation was learned for each objective.
    *   **Trace Objective Minimization (Table 1):** Empirically confirmed Theorem 4 and 5 by showing that each method minimized its *corresponding* negative trace objective (e.g., \( \Phi \) minimized \( -\text{f}_{\text{BYOL-}\Pi} \), \( \Phi_{ac} \) minimized \( -\text{f}_{\text{BYOL-AC}} \), and \( \Phi_{var} \) minimized \( -\text{f}_{\text{BYOL-VAR}} \)) with high probability (99-100%).
    *   **Value Function Fitting (Table 2):** Evaluated how well the learned representations fit traditional V-MSE, Q-MSE, and Advantage-MSE.
        *   \( \Phi \) and \( \Phi_{ac} \) performed competitively in fitting state-value (V-MSE of 6.32 vs. 6.48) and action-value (Q-MSE of 8.31 vs. 8.01).
        *   \( \Phi_{var} \) was optimal for fitting the true Advantage MSE (0.43, 100% best), confirming its role in capturing action-distinguishing features. \( \Phi_{ac} \) also showed better Advantage fitting than \( \Phi \).
    *   **Robustness to Policy Perturbation (Appendix C):** \( \Phi_{ac} \) (BYOL-AC) was found to be significantly more robust to changes in the initial policy used for representation learning compared to \( \Phi \) (BYOL-Π) and \( \Phi_{var} \) (BYOL-VAR).

2.  **Deep Reinforcement Learning (Sec 6.2):**
    *   **Agents and Environments:** Used V-MPO (policy-gradient, online) in Minigrid (DoorKey, MemoryS13/17, MultiRoom) and DQN (off-policy) in classic control domains (CartPole, MountainCar, Acrobot).
    *   **V-MPO Results (Figure 2):** \( \Phi_{ac} \) consistently outperformed other baselines in 3 out of 4 Minigrid tasks, and was on par in one. \( \Phi \) was competitive but generally slightly worse. \( \Phi_{var} \) performed poorly in all tasks, likely due to the non-linear predictors causing the objective to behave like a min-max adversarial optimization, removing features useful for general RL tasks.
    *   **DQN Results (Figure 3):** \( \Phi_{ac} \) outperformed \( \Phi \) in CartPole and MountainCar, and performed on par in Acrobot. \( \Phi_{var} \) was not evaluated due to its poor V-MPO performance.

---

### Key Takeaways
1.  **BYOL-AC's Superiority:** Both theoretically and empirically, the action-conditional \( \text{BYOL-AC} \) objective generally yields a better representation (\( \Phi_{ac} \)) for RL agents compared to the fixed-policy \( \text{BYOL-}\Pi \) (\( \Phi \)). \( \Phi_{ac} \) captures spectral information about per-action transition dynamics (\( T_a \)), which proves more beneficial in practice.
2.  **Variance Relationship Unifies Objectives:** The core theoretical insight is the "variance equation" (Remark 1 & 2) that connects the representations learned by \( \text{BYOL-}\Pi \), \( \text{BYOL-AC} \), and the novel \( \text{BYOL-VAR} \). \( \text{BYOL-AC} \) relates to the second moment of per-action eigenvalues, \( \text{BYOL-}\Pi \) to the square of the first moment, and \( \text{BYOL-VAR} \) to their variance. This provides a deep understanding of what each objective prioritizes in feature learning.
3.  **Dual Unifying Perspectives:** The model-based and model-free lenses provide valuable intuitions.
    *   **Model-Based:** \( \text{BYOL-}\Pi \) learns a low-rank approximation of \( T^\pi \), \( \text{BYOL-AC} \) approximates \( T_a \), and \( \text{BYOL-VAR} \) approximates the residual \( (T_a - T^\pi) \). This directly links objectives to fundamental dynamics.
    *   **Model-Free:** Each objective effectively minimizes a loss for fitting a certain 1-step value function: \( \text{BYOL-}\Pi \) for value (\( V \)), \( \text{BYOL-AC} \) for Q-value (\( Q \)), and \( \text{BYOL-VAR} \) for advantage functions. This demonstrates their suitability for different RL quantities.
4.  **BYOL-VAR's Niche:** While \( \text{BYOL-VAR} \) performed poorly in deep RL (likely due to non-linearities and adversarial optimization behavior), its theoretical connection to advantage functions and its focus on action-distinguishing features suggest potential utility in specialized applications, such as learning action representations or for option discovery in hierarchical RL.
5.  **Robustness:** \( \Phi_{ac} \) also exhibited higher robustness to initial policy perturbations, suggesting its learned features are more transferable across minor policy shifts.

In essence, the paper provides a comprehensive analytical and empirical framework for understanding action-conditional self-predictive learning, highlighting \( \text{BYOL-AC} \) as a robust and superior approach for learning representations in RL.

---

</details>

### 📚 Citation

```bibtex
@inproceedings{khetarpal2024,
  author    = {Khimya Khetarpal and Z. Guo and B. '. Pires and Yunhao Tang and Clare Lyle and Mark Rowland and N. Heess and Diana Borsa and A. Guez and Will Dabney},
  year      = {2024},
  title     = {A Unifying Framework for Action-Conditional Self-Predictive Reinforcement Learning},
  booktitle = {arXiv.org},
  doi       = {10.48550/arXiv.2406.02035},
}
```

## [Narrowing the Gap between Adversarial and Stochastic MDPs via Policy Optimization](https://arxiv.org/abs/2406.02035)  
**Authors**: Daniil Tiapkin, Evgenii Chzhen, Gilles Stoltz
**Conference**: AISTATS 2025  
**Tags**: Adversarial Markov Decision Processes

---

<details markdown="1">
  <summary>Read More</summary>

### Core Idea
The paper proposes Adversarial Policy Optimization based on Monotonic Value Propagation (APO-MVP), an algorithm for learning in episodic, obliviously adversarial Markov Decision Processes (MDPs) with full information. The core idea is to narrow the gap between regret bounds for adversarial and stochastic MDPs by using policy optimization combined with dynamic programming, avoiding the use of occupancy measures, and achieving a \( \tilde{O}(\text{poly}(H)\sqrt{SAT}) \) regret bound. This improves upon prior state-of-the-art results in adversarial tabular MDPs by a factor of \( \sqrt{S} \), matching the minimax lower bound in dependencies on state (\( S \)), action (\( A \)), and episode (\( T \)) counts.

---

### Problem Statement
The paper addresses the problem of learning optimal policies in an \( H \)-episodic (obliviously) adversarial Markov Decision Process (MDP). An MDP is defined by a finite set of states \( S \) (cardinality \( S \)), a finite set of actions \( A \) (cardinality \( A \)), a sequence of Markov transition kernels \( P = (P_h)_{h \in [H-1]} \) where \( P_h: S \times A \to \Delta(S) \), and a fixed-in-advance sequence of bounded time-inhomogeneous \( H \)-episodic reward functions \( (r_t)_{t>1} \) where \( r_t = (r_{t,h})_{h \in [H]} \) and \( r_{t,h}: S \times A \to [0,1] \). The number of episodes \( T \) is fixed and known, and each episode starts from an initial state \( s_1 \).

At each episode \( t \) and stage \( h \), the learner chooses a stage policy \( \pi_{t,h}: S \to \Delta(A) \), samples an action \( a_{t,h} \sim \pi_{t,h}(\cdot | s_{t,h}) \), moves to the next state \( s_{t,h+1} \sim P_h(\cdot | s_{t,h}, a_{t,h}) \) (if \( h < H \)), and finally observes the reward function \( r_t \) at the end of the episode.

The objective is to minimize the regret \( R_T \), defined as the difference between the accumulated value of the best static policy in hindsight and the accumulated value achieved by the learner's policies:
$\( R_T = \max_{\pi} \sum_{t=1}^T (V^{\pi, r_t, P}_1(s_1) - V^{\pi_t, r_t, P}_1(s_1)) \)$
where \( V^{\pi, r_t, P}_h(s) \) denotes the value function of policy \( \pi \) at episode \( t \), stage \( h \), starting from state \( s \).

---

### Motivation
Previous approaches to adversarial MDPs, particularly those with unknown transition kernels, largely relied on online linear optimization (OLO) strategies in the space of *occupancy measures* (e.g., O-REPS algorithms). While successful, these methods faced two significant drawbacks:
1.  **Computational Complexity:** They often required solving high-dimensional convex programs at each episode, leading to non-explicit policy updates and practical implementation challenges.
2.  **Suboptimal Regret Dependency:** When extended to handle unknown transition kernels, these methods incurred an additional \( \sqrt{S} \) factor in their regret bounds compared to the state-of-the-art in stochastic (non-adversarial) MDPs.

Another line of research focused on *policy-optimization-based approaches*, which are more practical due to their connection to algorithms like TRPO and PPO. However, these methods also historically suffered from the additional \( \sqrt{S} \) factor in the regret bound for finite MDP settings.

The key motivation for this work is to address the open question of whether the dependency on the number of states (\( S \)) can be matched between adversarial and stochastic MDPs, while simultaneously providing a more practical and easily implementable algorithm that avoids occupancy measures.

---

### Method Overview
The proposed algorithm, APO-MVP (Adversarial Policy Optimization based on Monotonic Value Propagation), combines dynamic programming with a black-box online linear optimization (OLO) strategy. It operates in random epochs, leveraging ideas from Zhang et al. (2021, 2023) and Jonckheere et al. (2023).

The algorithm proceeds as follows:

1.  **Epoch Switching:** The learning process is divided into random epochs \( E_e \subseteq [T] \). An epoch switch occurs when, for any state-action pair \( (s,a) \) at stage \( h \), the empirical count \( n_{t,h}(s,a) \) (number of times \( (s,a) \) was visited at stage \( h \)) reaches a power of two, i.e., \( 2^{\ell-1} \) for some integer \( \ell > 1 \).
2.  **Model Estimation and Bonuses:** At the beginning of each epoch \( e \), empirical transition kernels \( \hat{P}^{(e)} = (\hat{P}^{(e)}_h)_{h \in [H-1]} \) and bonus functions \( b^{(e)} = (b^{(e)}_h)_{h \in [H]} \) are computed and fixed for the entire epoch.
    *   The estimated transition probability \( \hat{P}_{t,h}(s'|s,a) \) for the current epoch is \( 1/S \) if \( n_{\tau,h}(s,a)=0 \) (where \( \tau \) is the episode of the last epoch switch), and \( \frac{n_{\tau,h}(s,a,s')}{n_{\tau,h}(s,a)} \) otherwise.
    *   The bonus function \( b_{t,h}(s,a) \) is defined as \( H \) if the local epoch counter \( \ell=0 \) (meaning \( n_{t,h}(s,a) \) is low), and \( \sqrt{\frac{2H^2 \ln(J)}{2^{\ell-1}}} \wedge H \) otherwise, where \( J = 2SATH \log_2(2T)/\delta \).
3.  **Policy Interaction:** For each episode \( t \) in the current epoch \( e_t \):
    *   The agent plays policies \( \pi_t = (\pi_{t,h})_{h \in [H]} \) to interact with the environment, observing states and actions.
    *   Empirical counts \( n_{t,h}(s,a,s') \) and \( n_{t,h}(s,a) \) are updated based on observed transitions.
4.  **Value and Advantage Estimation (Backward Pass):** At the end of episode \( t \), once the reward function \( r_t \) is revealed, optimistic estimates of Q-value and value functions are computed in a backward fashion using Bellman's equations, incorporating the fixed estimated transitions \( \hat{P}^{(e_t)} \) and bonus functions \( b^{(e_t)} \):
    *   For \( h=H \): \( Q_{t,H}(s,a) = r_{t,H}(s,a) \) and \( V_{t,H}(s) = \pi_{t,H} \cdot Q_{t,H}(s) \).
    *   For \( h \in [H-1] \): \( Q_{t,h}(s,a) = r_{t,h}(s,a) + b^{(e_t)}_h(s,a) + \hat{P}^{(e_t)}_h \cdot V_{t,h+1}(s,a) \) and \( V_{t,h}(s) = \pi_{t,h} \cdot Q_{t,h}(s) \).
    *   Estimated advantage functions are calculated as \( A_{t,h}(s,a) = Q_{t,h}(s,a) - V_{t,h}(s) \).
    *   A crucial technical remark is that value function clipping to \( [0,H] \) is *not* used to preserve the performance-difference lemma, incurring an additional \( H \) factor in the regret but simplifying the adversarial analysis.
5.  **Policy Update (Online Linear Optimization):** Policies for the next episode, \( \pi_{t+1} \), are determined by feeding the history of estimated advantage functions from the current epoch to a black-box OLO strategy \( \phi \). Specifically, for each \( (s,h) \in S \times [H] \):
    *   \( \pi_{t+1,h}(\cdot | s) = \phi_t((A_{\tau,h}(s, \cdot))_{\tau \in E_{e_t} \cap [t-1]}) \).
    *   The paper considers polynomial-potential and exponential-potential based OLO strategies, which provide closed-form expressions for the policies and satisfy specific performance guarantees (Definition 6).

**Computational Complexity:** The algorithm's computational complexity is dominated by the dynamic programming step, which is \( O(S^2AH) \) per episode. The policy optimization phase adds \( O(SAH) \) operations, making the overall per-episode complexity \( O(S^2AH) \). The space complexity is \( O(S^2AH) \), standard for model-based RL.

---

### Theoretical Contributions
The paper makes the following key theoretical contributions:

1.  **Improved Regret Bound:** APO-MVP achieves a regret bound of \( \tilde{O}(\text{poly}(H)\sqrt{SAT}) \) in the setting of adversarial episodic MDPs with full information. Specifically, with probability at least \( 1-3\delta \), the regret \( R_T \) is bounded by:
    $\( R_T \le \sqrt{H^7SAT} \log_2(2T) (2 \log_2(2T) + 16\sqrt{\ln(A)}) + 7\sqrt{H^4SAT \ln(2SATH \log_2(2T)/\delta)} + 2\sqrt{2H^6 T \log_2(2T) \ln(2/\delta)} + 2H^3SA \)$
    where \( \tilde{O}(\cdot) \) hides all absolute constants and polylogarithmic multiplicative terms.

2.  **Bridging the \( \sqrt{S} \) Gap:** This result improves upon the previously best-known regret bound for adversarial tabular MDPs with unknown transitions, which was \( \tilde{O}(\sqrt{H^4S^2AT}) \) (Rosenberg and Mansour, 2019b). APO-MVP effectively removes a \( \sqrt{S} \) factor, significantly narrowing the gap between adversarial and stochastic MDPs, as stochastic MDPs typically achieve \( \tilde{O}(\sqrt{H^3SAT}) \).

3.  **Matching Minimax Lower Bounds (in S, A, T):** The derived regret bound matches the minimax lower bound \( \Omega(\sqrt{H^3SAT}) \) for stochastic MDPs with respect to dependencies on \( S \), \( A \), and \( T \), up to logarithmic factors. The main remaining gap is in the dependency on the horizon \( H \).

4.  **Modular and General Analysis:** The analysis of APO-MVP is modular, decomposing the total regret into four terms (A, B, C, D) corresponding to optimism, adversarial OLO performance, concentration of transition estimates, and bonus summation. This modularity allows for the integration of black-box OLO strategies and leverages recent advances in stochastic MDP analysis (Zhang et al., 2023) and adversarial learning (Jonckheere et al., 2023).

5.  **Avoidance of Occupancy Measures:** Unlike previous state-of-the-art methods, APO-MVP does not rely on occupancy measures, which are often difficult to work with in practice, especially with unknown transition kernels. This direct policy optimization approach contributes to its practical implementability.

---

### Experiments
The paper does not include any experimental results. The authors state this in their checklist (point 3a).

---

### Key Takeaways
1.  **Practical and Sample-Efficient Algorithm:** APO-MVP is a practical algorithm for adversarial MDPs, combining dynamic programming with OLO in the policy space. It avoids the use of complex occupancy measures, making it easier to implement than prior methods.
2.  **Reduced State-Dependency Regret:** The algorithm achieves a \( \tilde{O}(\text{poly}(H)\sqrt{SAT}) \) regret bound, which improves the dependency on the number of states \( S \) by a factor of \( \sqrt{S} \) compared to previous state-of-the-art methods in the adversarial setting.
3.  **Matching Stochastic Rates:** The achieved regret bound matches the minimax lower bound for stochastic MDPs in terms of \( S \), \( A \), and \( T \) dependencies, effectively bridging a long-standing gap in theoretical understanding between adversarial and stochastic settings. This suggests that policy optimization can be more sample-efficient in large state spaces for this problem class.
4.  **Modular Analysis Framework:** The paper's analytical decomposition of regret into adversarial and stochastic components, coupled with the use of black-box OLO strategies and monotonic value propagation techniques, offers a flexible framework for future research in adversarial MDPs.
5.  **Limitations and Open Problems:**
    *   **High H-dependency:** The regret bound's dependency on the episode length \( H \) is relatively high, \( \tilde{O}(\sqrt{H^7}) \). Reducing this dependency while maintaining the optimal \( S, A, T \) rates remains an open problem.
    *   **Full Monitoring Assumption:** The current analysis assumes full monitoring (observing \( s_{t,h+1} \) for all \( h \)). Extending the approach to bandit monitoring (only observing rewards) is a non-trivial challenge, and it is unknown if a \( \sqrt{SAT} \) regret is achievable in the adversarial case with bandit feedback.
    *   **Oblivious Adversary:** The \( \sqrt{S} \) factor improvement relies on the assumption of an oblivious adversary. A fully adversarial setup would likely reintroduce the \( \sqrt{S} \) factor due to concentration issues if \( \ell_1 \)-norm bounds were used.

---

</details>

### 📚 Citation
```bibtex
@inproceedings{tiapkin2024,
  author    = {Daniil Tiapkin and Evgenii Chzhen and Gilles Stoltz},
  year      = {2024},
  title     = {Narrowing the Gap between Adversarial and Stochastic MDPs via Policy Optimization},
  booktitle = {arXiv.org},
  doi       = {10.48550/arXiv.2407.05704},
}
```

## [Hybrid Transfer Reinforcement Learning: Provable Sample Efficiency from Shifted-Dynamics Data](https://arxiv.org/abs/2411.03810)  
**Authors**: Chengrui Qu, Laixi Shi, Kishan Panaganti, Pengcheng You, Adam Wierman
**Conference**: AISTATS 2025  
**Tags**: Hybrid Transfer RL, Distribution shift, Sample Complexity

---

<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea

This paper introduces a novel Hybrid Transfer Reinforcement Learning (HTRL) setting where an agent learns in a target environment while leveraging an offline dataset collected from a source environment with shifted dynamics. The core idea is to theoretically demonstrate how to provably enhance sample efficiency in this setting, despite an initial hardness result for general shifted dynamics. The proposed algorithm, HySRL, achieves problem-dependent sample complexity by first identifying the unknown dynamics shift and then designing an exploration strategy that intelligently reuses source data, outperforming pure online RL under specific conditions.

---

### ❓ Problem Statement

#### What problem is the paper solving?

Online Reinforcement Learning (RL) typically requires extensive, high-stakes online interaction data, making it sample inefficient and often impractical for real-world applications. While leveraging historical data from related or outdated source environments could improve efficiency, it remains unclear how to effectively use such data, especially when the source and target environments have unknown shifted dynamics, to achieve provable sample efficiency gains. Existing frameworks lack theoretical guarantees for this specific transfer setting, and naive transfer can even lead to "negative transfer." The paper aims to answer whether data from a shifted source environment can provably enhance sample efficiency when learning in a target environment.

---

### 🎯 Motivation

The high sample complexity of online RL is a significant practical barrier. Transfer learning offers a promising direction, but using historical data from environments with dynamics shifts (e.g., imperfect simulators, outdated operational data) poses challenges. Prior empirical studies show mixed results, with some demonstrating benefits and others indicating negative transfer. This highlights a critical need for theoretical understanding and provable guarantees in transfer RL with dynamics shifts, which existing hybrid RL (where source and target dynamics are identical) or cross-domain RL (which often lack sample complexity analysis) frameworks do not adequately address. The paper is motivated to provide such theoretical insights and design an algorithm with provable sample efficiency.

---

### 🛠️ Method Overview

The paper proposes HySRL, a two-stage transfer algorithm designed for HTRL with \( \beta \)-separable shifts. A Markov Decision Process (MDP) is defined as \( M = (S, A, H, p, r, \rho) \), where \( S \) is state space size, \( A \) is action space size, \( H \) is horizon, \( p \) is transition probability, \( r \) is reward, and \( \rho \) is initial state distribution. The target MDP is \( M_{tar} = (S, A, H, p_{tar}, r, \rho) \) and the source MDP is \( M_{src} = (S, A, H, p_{src}, r, \rho) \), differing only in transition probabilities.

The key assumption for HySRL's provable gains is the \( \beta \)-separable shift (Definition 1): for some \( \beta \in (0, 1] \), for all \( (s, a) \in S \times A \), \( p_{src}(\cdot | s, a) \neq p_{tar}(\cdot | s, a) \implies TV(p_{src}(\cdot | s, a), p_{tar}(\cdot | s, a)) \geq \beta \). This means transitions are either identical or differ by at least \( \beta \) in total variation distance. The shifted region \( B \) is defined as \( B \triangleq \{(s, a) \in S \times A | p_{src}(\cdot | s, a) \neq p_{tar}(\cdot | s, a)\} \). The algorithm also assumes \( \sigma \)-reachability (Assumption 1): \( \max_{\pi} \max_{h \in [H]} p^{\pi}_h(s, a) \geq \sigma \), \( \forall(s, a) \in S \times A \).

HySRL (Algorithm 1) operates in two main steps:

1. Reward-free Shift Identification (Algorithm 2):
This phase aims to accurately estimate \( p_{tar} \) and identify the shifted region \( \hat{B} \). It employs a reward-free exploration strategy inspired by RF-Express.

It recursively updates an uncertainty function \( W^t_h(s, a) \) (Eq. 1):

$$
W^t_h(s, a) \triangleq \min \left\{ 1, \frac{4H\sqrt{g_1(n_t(s, a), \delta)}}{n_t(s, a)} + \sum_{s'}\hat{p}_{tar}^t(s' | s, a) \max_{a'} W^t_{h+1}(s', a') \right\}
$$

where \( g_1(n, \delta) \triangleq \log(6SAH/\delta) + S \log(8e(n + 1)) \) and \( n_t(s, a) \) is the visitation count for \( (s, a) \) in \( t \) episodes.
The policy \( \pi^{t+1}_h(\cdot) = \arg \max_{a \in A} W^t_h(\cdot, a) \) is chosen to gather online data.
The algorithm stops when the uncertainty measure is sufficiently small: \( 3\sqrt{\rho_{\pi^{t+1}_1} W^t_1} + \rho_{\pi^{t+1}_1} W^t_1 \leq \sigma\beta/8 \).
The estimated shifted region is \( \hat{B} \triangleq \{(s, a) \in S \times A | TV(\hat{p}_{src}(\cdot | s, a), \hat{p}_{tar}(\cdot | s, a)) \gt \beta/2 \} \). This step guarantees \( \hat{B}=B \) with high probability (Lemma 1).

2. Hybrid UCB Value Iteration (Algorithm 3):
Once \( \hat{B} \) is identified, this phase leverages the source data for efficient exploration.

It defines empirical transitions \( \tilde{p}^t(\cdot | s, a) \) and visitation counts \( \tilde{n}_t(s, a) \):
If \( (s, a) \in \hat{B} \), \( \tilde{n}_t(s, a) \triangleq n_t(s, a) \) (online count) and \( \tilde{p}^t(\cdot | s, a) \triangleq \hat{p}_{tar}^t(\cdot | s, a) \) (empirical target transition).
If \( (s, a) \notin \hat{B} \), \( \tilde{n}_t(s, a) \triangleq n_{src}(s, a) \) (source count) and \( \tilde{p}^t(\cdot | s, a) \triangleq \hat{p}_{src}(\cdot | s, a) \) (empirical source transition).


It computes optimistic Q-functions \( Q^t_h(s, a) \) (Eq. 3a) and value functions \( V^t_h(s) \):
$$
Q^t_h(s, a) \triangleq \min \left\{ H, r(s, a) + 3\sqrt{\frac{Var_{\tilde{p}^t}(V^t_{h+1})(s, a) g_2(\tilde{n}_t(s, a), \delta)}{\tilde{n}_t(s, a)}} + \frac{14H^2 g_1(\tilde{n}_t(s, a), \delta)}{\tilde{n}_t(s, a)} + \frac{1}{H}\tilde{p}^t(V^t_{h+1} - V^t_{h+1})(s, a) + \tilde{p}^t V^t_{h+1}(s, a) \right\}
$$
where \( g_2(n, \delta) \triangleq \log(6SAH/\delta) + \log(8e(n + 1)) \). \( V^t_{h+1} \) and \( V^t_{h+1} \) are upper and lower confidence bounds on optimal value functions.
The policy \( \pi^{t+1}_h(\cdot) = \arg \max_{a \in A} Q^t_h(\cdot, a) \) is used for exploration.
An optimality gap function \( G^t_h(s, a) \) (Eq. 4) is tracked:
$$
G^t_h(s, a) \triangleq \min \left\{ H, 6\sqrt{\frac{Var_{\tilde{p}^t}(V^t_{h+1})(s, a) g_2(\tilde{n}_t(s, a), \delta)}{\tilde{n}_t(s, a)}} + \frac{35H^2 g_1(\tilde{n}_t(s, a), \delta)}{\tilde{n}_t(s, a)} + (1 + \frac{3}{H})\tilde{p}^t \pi^{t+1}_{h+1} G^t_{h+1}(s, a) \right\}
$$
Algorithm 3 stops when \( \rho_{\pi^{t+1}_1} G^t_1 \leq \epsilon \), guaranteeing \( \epsilon \)-optimality (Lemma 8).

---

### 📐 Theoretical Contributions

The paper provides two main theoretical contributions:

1. Minimax Lower Bound for General HTRL (Theorem 1):

For any source MDP \( M_{src} \), the minimax lower bound on the sample complexity for finding an \( \epsilon \)-optimal policy in a target MDP \( M_{tar} \in \mathcal{M}_{\alpha} \) (where \( \mathcal{M}_{\alpha} \) is a set of MDPs with total variation distance from \( M_{src} \) bounded by \( \alpha \)) is \( \Omega(H^3SA/\epsilon^2) \). This matches the best known lower bound for pure online RL, demonstrating that, without further conditions on the dynamics shift, general shifted-dynamics data (even with subtle shifts) does not provably reduce sample complexity in the worst case. This result motivates the focus on more practical settings with prior information.

2. Problem-Dependent Sample Complexity for \( \beta \)-separable Shifts (Theorem 2):

Under the assumptions of \( \beta \)-separable shift and \( \sigma \)-reachability, and with sufficient source data \( D_{src} \) (at least \( eO(H^3/\epsilon^2 + S/\beta^2) \) samples for each \( (s, a) \)), HySRL can find an \( \epsilon \)-optimal policy for \( M_{tar} \) with total online samples collected from \( M_{tar} \) of:

$$
eO \left( \min \left\{ \frac{H^3SA}{\epsilon^2}, \frac{H^3|B|}{\epsilon^2} + \frac{H^2S^2A}{(\sigma\beta)^2} \right\} \right)
$$

This formula shows that:

When \( \epsilon \geq \Omega(\sqrt{H/S}\sigma\beta) \), the sample complexity is \( eO(H^3SA/\epsilon^2) \), matching pure online RL, thus provably avoiding negative transfer.
When \( \epsilon \lt \Omega(\sqrt{H/S}\sigma\beta) \), the sample complexity becomes \( eO(H^3|B|/\epsilon^2 + H^2S^2A/(\sigma\beta)^2) \). Crucially, if the shifted region \( |B| \) is significantly smaller than the full state-action space \( SA \) (i.e., \( |B| \ll SA \)), this bound is strictly better than pure online RL. This demonstrates that HySRL achieves provable sample efficiency gains when the dynamics shift only affects a small portion of the environment. The additional term \( H^2S^2A/(\sigma\beta)^2 \) reflects the cost of identifying the shifted region \( B \).

---

### 📊 Experiments

The proposed algorithm, HySRL, is evaluated against the state-of-the-art online RL baseline, BPI-UCBVI (Ménard et al., 2021), in a GridWorld environment (\( S=16, A=4, H=20 \)). The source environment had \( 1 \times 10^5 \) episodes of pre-collected data and differed from the target environment by including three additional absorbing states in the target.

Key findings include:

- Superior Sample Efficiency: HySRL learns the optimal policy with approximately \( 1 \times 10^6 \) samples from the target environment, while BPI-UCBVI converges significantly slower, demonstrating the substantial sample efficiency improvement from leveraging shifted-dynamics source data.
- Robustness to Inaccurate \( \beta \): An ablation study showed that HySRL's performance degradation was minor even when the input \( \beta \) (0.45) did not precisely match the true \( \beta \) (ranging from 0.05 to 0.4). HySRL still outperformed BPI-UCBVI within finite samples, indicating its practical robustness.

---

### 📈 Key Takeaways

- Hardness of General HTRL: Without prior information about the degree of dynamics shift, general hybrid transfer RL cannot provably reduce sample complexity compared to pure online RL in the worst case.
- Value of Prior Information: When prior information about the minimum degree of dynamics shift (\( \beta \)-separable shift) is available, provable sample efficiency gains are achievable.
- HySRL's Effectiveness: The proposed HySRL algorithm, by first identifying the shifted region and then using a carefully designed hybrid exploration strategy, can effectively leverage shifted-dynamics offline data.
- Avoidance of Negative Transfer: HySRL's problem-dependent sample complexity guarantees that it will perform at least as well as state-of-the-art pure online RL, thereby provably avoiding negative transfer.
- Significant Gains for Sparse Shifts: When the dynamics shift impacts only a small portion of the state-action space (\( |B| \ll SA \)), HySRL provides significant sample efficiency improvements.
- Practical Insights: The work provides theoretical foundations for understanding how to effectively incorporate source data in transfer RL, particularly for scenarios where domain variations are localized.

---

</details>

### 📚 Citation

```bibtex
@inproceedings{qu2024,
  author    = {Chengrui Qu and Laixi Shi and Kishan Panaganti and Pengcheng You and Adam Wierman},
  year      = {2024},
  title     = {Hybrid Transfer Reinforcement Learning: Provable Sample Efficiency from Shifted-Dynamics Data},
  booktitle = {arXiv.org},
  doi       = {10.48550/arXiv.2411.03810},
}
```

## [Decision-Point Guided Safe Policy Improvement](hhttps://arxiv.org/abs/2410.09361)  
**Authors**: Abhishek Sharma, Leo Benac, Sonali Parbhoo, Finale Doshi-Velez
**Conference**: AISTATS 2025  
**Tags**: Batch Reinforcement Learning, Safe Policy Improvement

---

<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea

Decision Points RL (DPRL) is a safe batch reinforcement learning (RL) algorithm that ensures a learned policy performs at least as well as the behavior policy that generated the dataset. The core idea is to restrict policy improvements to a subset of state-action pairs, or regions in continuous states, termed "Decision Points." These are defined as state-action pairs \( (s,a) \) that have been observed at least \( N_\wedge \) times and where the estimated Q-value for action \( a \) in state \( s \) under the behavior policy, \( \hat{Q}_{\pi_b}(s,a) \), is greater than or equal to the estimated V-value for state \( s \) under the behavior policy, \( \hat{V}_{\pi_b}(s) \). For states where high-confidence improvements cannot be identified (i.e., not a decision point), DPRL defers to the current behavior policy. This selective improvement strategy, combined with deferral, leads to significantly tighter theoretical guarantees, with data-dependent bounds that do not scale with the size of the state and action spaces, unlike prior work.

---

### ❓ Problem Statement

#### What problem is the paper solving?

The main challenge in Safe Policy Improvement (SPI) within batch RL is to identify and implement policy changes that yield improvements while carefully balancing the inherent risk, especially when many state-action pairs are infrequently visited. Existing approaches face several limitations:

1. Density-based methods (e.g., CQL, Behavior Cloning) constrain the learned policy to be close to the behavior policy. However, this can be overly conservative and suboptimal if the behavior policy is stochastic and suboptimal, as it may prevent the selection of better, explored actions.
2. Pessimism-based planning (e.g., PQI) incorporates pessimism into value estimates proportional to uncertainty. This can also be overly conservative by penalizing actions leading to rarely visited states, even if the action itself is frequently observed.
3. Support-constrained policies restrict the learned policy to the support of the behavior policy. While less conservative than density-based methods, they can become unsafe with noisy actions or rewards, common in real-world data. Count-based techniques like SPIBB impose count-based constraints for improvements but require a priori access to the true behavior policy, which is often impractical, and their guarantees may not be tight in practice.
4. Scalability: Many existing methods' improvement guarantees scale with the size of the state and action spaces (\( |S||A| \)), leading to loose bounds in large, sparse environments.
5. Practicality: Real-world batch RL deployments often require incremental changes that are easily reviewable and implementable (e.g., by a clinician). Current methods may not naturally yield such interpretable modifications.

---

### 🎯 Motivation

Batch RL is crucial for applications where direct interaction with the environment is risky (e.g., medicine, robotics) or expensive. However, its effectiveness hinges on a sufficiently exploratory behavior policy. In scenarios with limited exploration, or where expert behavior exhibits systematic errors, learning an truly optimal policy might not be feasible without risking the adoption of an unsafe policy that performs worse than the existing behavior. Therefore, providing safe, high-confidence modifications relative to the existing behavior policy is a more practical and often sufficient goal.

The key motivations behind DPRL are:

1. Targeted Improvements: Instead of broadly constraining the learned policy, DPRL identifies only those specific state-action pairs where confident improvements can be made. This addresses the conservativeness of density- and pessimism-based methods.
2. Confidence through Data Density: By focusing on "Decision Points"—state-action pairs with a high visitation count (\( n(s,a) \geq N_\wedge \)) and estimated advantage (\( \hat{Q}_{\pi_b}(s,a) \geq \hat{V}_{\pi_b}(s) \))—DPRL ensures that any deviation from the behavior policy is made with high confidence.
3. Strategic Deferral: For states where such confident improvements cannot be established, DPRL explicitly defers to the behavior policy. This prevents risky actions in sparsely visited or uncertain regions, offering a robust safety mechanism.
4. Independence from True Behavior Policy: Unlike SPIBB, DPRL does not require a priori knowledge of the true behavior policy during training, making it more applicable in real-world settings where obtaining the functional form of expert behavior is difficult.
5. Tighter Guarantees: By restricting the scope of policy changes to well-observed regions, DPRL achieves tighter, data-dependent theoretical bounds that scale with the number of sufficiently observed state-action pairs (\( C(N_\wedge) \)) rather than the entire state-action space (\( |S||A| \)).
6. Actionable Policies: The resulting policy often involves a small number of high-impact changes, making it easier for human experts (e.g., clinicians) to review, understand, and implement.

---

### 🛠️ Method Overview

DPRL adapts its approach based on whether the state space is discrete or continuous.

#### Discrete Case (DPRL-D)

1. Define Decision Points: From the given dataset \( \mathcal{D} \), DPRL first identifies "Decision Points" at which a change from the behavior policy \( \pi_b \) is considered. This involves defining:

\( \mathcal{A}^{DP}_s = \{a \in \mathcal{A} : n(s, a) \geq N_\wedge \text{ and } \hat{Q}_{\pi_b}(s, a) \geq \hat{V}_{\pi_b}(s)\} \): The set of "advantageous actions" for a state \( s \). An action \( a \) is advantageous if its empirical count \( n(s,a) \) is at least a threshold \( N_\wedge \), and its estimated Q-value under the behavior policy \( \hat{Q}_{\pi_b}(s,a) \) is greater than or equal to the estimated state value \( \hat{V}_{\pi_b}(s) \).
\( \mathcal{S}^{DP} = \{s \in \mathcal{S} : \mathcal{A}^{DP}_s \neq \emptyset\} \): The set of states where at least one advantageous action exists. These are the states identified as "decision points."
\( \Phi = \{s \in \mathcal{S} : \mathcal{A}^{DP}_s = \emptyset\} \): The set of states where no confident advantageous action can be found, implying deferral to \( \pi_b \).
Empirical Q-values \( \hat{Q}_{\pi_b}(s,a) \) and \( \hat{V}_{\pi_b}(s) \) are estimated using first-visit Monte Carlo averages from the dataset.

2. Construct Elevated Semi-MDP (SMDP): An "elevated" SMDP \( \tilde{\mathcal{M}} = (\mathcal{S}^{DP}, \mathcal{A}, \tilde{P}, \tilde{R}, \tilde{\gamma}) \) is constructed. The state space is restricted to \( \mathcal{S}^{DP} \). The transition function \( \tilde{P}(s'|s,a) \), reward function \( \tilde{R}(s,a) \), and discount factor \( \tilde{\gamma}(s,a,s') \) are estimated using data from \( \mathcal{D} \) that connect decision points:

\( \tilde{P}(s'|s,a) = \frac{\sum_{k=1}^T \tilde{n}(s, a, s', k)}{\sum_{s'' \in \mathcal{S}^{DP}} \sum_{k=1}^T \tilde{n}(s, a, s'', k)} \)
\( \tilde{\gamma}(s,a,s') = \frac{\sum_{k=1}^T \tilde{n}(s, a, s', k)\gamma^k}{\sum_{k=1}^T \tilde{n}(s, a, s', k)} \)
\( \tilde{R}(s,a) = \frac{\sum_{s' \in \mathcal{S}^{DP}} \sum_{k=1}^T \sum_{n=1}^N \tilde{r}(n, s, a, s', k)}{\sum_{s' \in \mathcal{S}^{DP}} \sum_{k=1}^T \tilde{n}(s, a, s', k)} \)
where \( \tilde{n}(s, a, s', k) \) is the count of trajectories starting from \( (s,a) \) and reaching \( s' \) as the first decision point after \( k \) steps, and \( \tilde{r}(n, s, a, s', k) \) is the discounted sum of rewards along such a segment.

3. Policy Optimization: Policy iteration is applied to the constructed SMDP \( \tilde{\mathcal{M}} \) over a restricted policy set \( \Pi_{\text{DP}} \). This set \( \Pi_{\text{DP}} \) contains deterministic policies that can only select advantageous actions: \( \pi(a|s) = 0 \quad \forall a \notin \mathcal{A}_{DP}^s \). In each iteration \( i \), the policy \( \pi^{(i)} \) is evaluated to get \( V^{(i)}_{\tilde{\mathcal{M}}} \), which is then improved to \( \pi^{(i+1)} \):
$$
\pi^{(i+1)}(s) = \arg \max_{a \in \mathcal{A}^{DP}_s} \left( \tilde{R}(s, a) + \mathbb{E}_{\tilde{P}(s'|s,a)} [\tilde{\gamma}(s, a, s')V^{(i)}_{\tilde{\mathcal{M}}}(s')] \right)
$$
This process converges to a policy \( \pi^{(K)} \).

4. Final Policy Construction: The final policy \( \pi_{\text{DP}}(s) \) is defined as:
$$
\pi_{\text{DP}}(s) = \begin{cases} \text{DEFER} \quad \text{if } s \in \Phi \\ \pi^{(K)}(s) \quad \text{otherwise} \end{cases}
$$
where "DEFER" implies adhering to the behavior policy \( \pi_b \).

#### Continuous Case (DPRL-C)
For continuous state spaces, DPRL-C uses a non-parametric, neighborhood-based approach. The dataset \( \mathcal{D} \) is stored, and queries are made using Ball-Tree data structures for efficient neighbor search.

1. Neighborhood Definition: For a given state \( s \) and action \( a \):

- \( N(s) \): Set of neighbors of \( s \) within a ball of radius \( r \).
- \( n(s) = |N(s)| \): Count of neighbors of \( s \).
- \( N(s,a) \): Set of state-action neighbors of \( (s,a) \) within a radius \( r \) (only if actions match).
- \( n(s,a) = |N(s,a)| \): Count of state-action neighbors.

The distance metric \( d((s,a), (s',a')) = d(s,s') \) if \( a=a' \), and \( \infty \) otherwise, with \( d(s,s') = \|s-s'\| \).


2. Define Advantageous Actions: Similar to the discrete case, the set of advantageous actions \( \mathcal{A}^{DP}_s \) for a state \( s \) is defined as:
\( \mathcal{A}^{DP}_s = \{a \in \mathcal{A} : n(s, a) \geq N_\wedge \text{ and } \hat{Q}_{\pi_b}(s, a) \geq \hat{V}_{\pi_b}(s)\} \)
Here, \( \hat{Q}_{\pi_b}(s, a) \) and \( \hat{V}_{\pi_b}(s) \) are estimated by averaging the returns of neighbors in \( N(s,a) \) and \( N(s) \) respectively.

3. Final Policy Construction: The policy \( \pi_{\text{DP}}(s) \) is implicitly defined:
$$
\pi_{\text{DP}}(s) = \begin{cases} \text{DEFER} \quad \text{if } \mathcal{A}^{DP}_s = \emptyset \\ \arg \max_{a \in \mathcal{A}^{DP}_s} \hat{Q}_{\pi_b}(s, a) \quad \text{otherwise} \end{cases}
$$
The hyperparameters \( r \) (radius) and \( N_\wedge \) control the bias-variance trade-off and the sparsity of improvements. A smaller \( r \) leads to lower bias but fewer decision points, while a larger \( r \) increases bias but potentially more decision points. \( N_\wedge \) ensures a minimum number of samples for reliable estimation.

---

### 📐 Theoretical Contributions

DPRL provides robust theoretical guarantees for safe policy improvement.

1. Theorem 1 (DPRL Discrete)

Let \( \pi_{\text{DP}} \) be the policy obtained by the DPRL-D algorithm. Then \( \pi_{\text{DP}} \) is a safe policy improvement over the behavior policy \( \pi_b \), with probability at least \( 1 - \delta \):
$$
 \rho(\pi_{\text{DP}}) - \rho(\pi_b) \geq - \frac{V_{\text{max}}}{1 - \gamma} \sqrt{\frac{1}{N_\wedge} \log \frac{C(N_\wedge)}{\delta}}
$$
where \( V_{\text{max}} \) is the maximum possible value, \( \gamma \) is the discount factor, \( N_\wedge \) is the minimum count threshold for decision points, and \( C(N_\wedge) \) is the count of state-action pairs observed at least \( N_\wedge \) times in the dataset:
$$
C(N_\wedge) = \sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} \mathbf{I}[n(s, a) \geq N_\wedge]
$$
Proof Sketch: The proof leverages the Performance Difference Lemma, which relates the value difference \( \rho(\pi_{\text{DP}}) - \rho(\pi_b) \) to the expected advantage \( A_{\pi_b}(s,a) = Q_{\pi_b}(s,a) - V_{\pi_b}(s) \) under \( \pi_{\text{DP}} \). The key is to bound the empirical advantage \( \hat{A}_{\pi_b}(s,a) = \hat{Q}_{\pi_b}(s,a) - \hat{V}_{\pi_b}(s) \). By expressing \( \hat{Q}_{\pi_b} \) and \( \hat{V}_{\pi_b} \) as first-visit Monte Carlo averages of returns, the analysis uses McDiarmid's inequality. For states where DPRL defers to \( \pi_b \) (i.e., \( s \in \Phi \)), the advantage is zero. For decision points (\( s \in \text{SDP} \)) where \( \pi_{\text{DP}} \) acts, \( \hat{A}_{\pi_b}(s,a) \) is positive by construction, and the true \( A_{\pi_b}(s,a) \) is bounded using McDiarmid's inequality for samples with at least \( N_\wedge \) observations. A union bound is then applied over the \( C(N_\wedge) \) state-action pairs where improvements are considered.

Discussion and Comparison to Baselines:

- Data-Dependent Bound: The bound's dependence on \( C(N_\wedge) \) is crucial. This term is typically much smaller than \( |\mathcal{S}||\mathcal{A}| \) in real-world sparse datasets, leading to significantly tighter guarantees than prior work (e.g., SPIBB, PQI, Kim and Oh (2023)) whose bounds directly depend on \( |\mathcal{S}||\mathcal{A}| \).
- Hyperparameter Control: The \( N_\wedge \) parameter directly controls the confidence-performance trade-off. Higher \( N_\wedge \) means higher confidence but potentially fewer improvements.
- No Behavior Policy Access: Unlike SPIBB, DPRL achieves its guarantees without requiring explicit knowledge of the behavior policy \( \pi_b \) during training, only for empirical evaluation.
- Independence from Density Threshold: Unlike Liu et al. (2020), DPRL's bound does not depend on a density threshold \( b = N_\wedge/|\mathcal{D}| \). This means the bound does not loosen when the dataset size \( |\mathcal{D}| \) increases while the set of well-observed state-action pairs remains constant, making it more robust to superfluous data in low-density regions.

2. Theorem 2 (DPRL Continuous)
Let \( M(r, N_\wedge) \) be the number of balls of radius \( r \) needed to cover the subset of \( \mathcal{S} \times \mathcal{A} \) where each \( (s,a) \) has at least \( N_\wedge \) data points in its ball \( B_r(s,a) \), and let \( \epsilon_r \) be the maximum error in estimating Q-values within a ball of radius \( r \). Then \( \pi_{\text{DP}} \) is a safe policy improvement over \( \pi_b \) with probability at least \( 1 - \delta \):
$$
\rho(\pi_{\text{DP}}) - \rho(\pi_b) \geq - \frac{V_{\text{max}}}{1 - \gamma} \left( \sqrt{\frac{1}{2N_\wedge} \log \frac{M(r, N_\wedge)}{\delta}} - 3\epsilon_r \right)
$$
Discussion and Comparison to Baselines:

- Volume Measure: \( M(r, N_\wedge) \) quantifies the "volume" of the state-action space where policy improvement is guaranteed. For sparse datasets, \( M(r, N_\wedge) \) will be much smaller than the total covering number \( M(r) \) of the entire space.
- Neighborhood Error: The \( 3\epsilon_r \) term accounts for the error introduced by using neighborhood-based estimates of Q-values.
- Hyperparameter Influence: \( N_\wedge \) affects the bound directly and indirectly through \( M(r, N_\wedge) \). A higher \( N_\wedge \) leads to more confident estimations and a smaller \( M(r, N_\wedge) \). The radius \( r \) influences the bias-variance trade-off: smaller \( r \) means lower bias (smaller \( \epsilon_r \)) but potentially a larger \( M(r, N_\wedge) \) (more distinct "dense" regions).
- Non-Parametric Advantage: This bound presents a non-parametric alternative to bounds for parametric methods (e.g., Liu et al. (2020)), which depend on the size of the function class \( |\mathcal{F}| \). While parametric methods optimize for a global estimation error, DPRL-C optimizes for a local estimation error. For datasets with non-uniform exploration, DPRL-C can achieve tighter bounds because \( M(r, N_\wedge) \) can be small in dense regions, whereas global error \( \epsilon_{\mathcal{F}} \) for parametric methods might remain large.
- Actionability: \( M(r, N_\wedge) \) can be estimated using clustering algorithms like DBSCAN, making the bound more actionable in practice.

---

### 📊 Experiments

DPRL's performance and safety were evaluated across various synthetic and real-world datasets, comparing against SPIBB, PQI, and CQL.

Discrete State and Action Spaces (Toy MDP, GridWorld):

- Challenging MDPs: Two toy MDPs (Figure 1) demonstrate issues with prior approaches. One ("Forest" MDP) shows PQI failing to achieve high 5% CVaR (Conditional Value at Risk) due to over-pessimism in potentially good, sparse regions. The other highlights how density-regularization (CQL) can fail with stochastic, suboptimal behavior. DPRL's ability to set an \( N_\wedge \) parameter allows it to ignore actions that cannot be reliably estimated.
- DPRL's Tighter Bounds: On both the MDPs and a \( 10 \times 10 \) GridWorld, DPRL consistently provided tighter safety bounds (Figure 1, center). This is attributed to the \( C(N_\wedge) \) term in its bound being much smaller than \( |S||A| \) in sparse environments, and its data-dependent nature preventing degradation with increasing \( |S| \).
- Safety and Performance: In GridWorld (Figure 2, right), DPRL maintained high CVaR (safety) without significantly affecting the mean value (performance), outperforming baselines.
- Bias-Variance Trade-off: The \( N_\wedge \) parameter effectively manages the bias-variance trade-off (Figure 2, center). Increasing \( N_\wedge \) decreases the fraction of states where the optimal action is considered (increased bias) but increases the fraction of states where a "better" action is chosen (reduced variance in value estimation), leading to safer outcomes.
- Robustness to Behavior Policy Estimation: A critical finding (Figure 3) is that SPIBB's performance (CVaR) significantly degrades when the true behavior policy is replaced with an estimated one. DPRL, which does not require the true behavior policy during training, consistently outperforms estimated SPIBB.

Continuous State, Discrete Action Spaces (Atari, MIMIC III):

- High-Dimensional Settings:
    - Atari: On five Atari environments (Qbert, Pong, Freeway, Bowling, Amidar), DPRL consistently learned good policies from suboptimal behavior data (where the "expert" took the worst action 50% of the time), achieving competitive returns compared to baselines (Figure 5, Figure 7).
    - MIMIC III (Hypotension Management): In a real-world medical dataset for hypotensive patients in ICU (11 core features + 18 handcrafted features, 16 discrete actions), DPRL achieved the highest estimated value via Off-Policy Evaluation (OPE) compared to the behavior baseline and other methods (Figure 4, left).


- Actionability through Deferral: For the chosen hyperparameters (\( N_\wedge = 50, r=10 \)), DPRL deferred to the behavior policy in over 95% of states (Figure 4, right). This highly sparse policy is actionable for clinicians, allowing them to focus on reviewing a minimal set of recommended changes.

---

### 📈 Key Takeaways

- Decision Points as a Foundation for Safe RL: DPRL introduces a novel framework for safe batch RL by identifying "Decision Points"—specific state-action pairs or regions where policy improvements can be made with high confidence due to sufficient data density and an estimated advantage over the behavior policy.
- Tighter, Data-Dependent Guarantees: DPRL provides significantly tighter theoretical bounds for safe policy improvement. Unlike prior methods that scale with the entire state-action space (\( |\mathcal{S}||\mathcal{A}| \)), DPRL's bounds scale with \( C(N_\wedge) \) (the number of sufficiently observed state-action pairs) for discrete states and \( M(r, N_\wedge) \) (volume of dense regions) for continuous states, making them more relevant for real-world sparse datasets.
- Independence from True Behavior Policy: A key practical advantage is that DPRL does not require a priori access to the true behavior policy during training, unlike methods like SPIBB, whose performance significantly degrades when the behavior policy must be estimated.
- Effective Safety-Performance Trade-off: The hyperparameter \( N_\wedge \) (and \( r \) in the continuous case) allows for explicit control over the balance between safety (high confidence in improvements) and performance.
- Practical Applicability and Actionability: Empirical evaluations on synthetic MDPs, GridWorld, Atari environments, and a real-world medical dataset (MIMIC III) demonstrate DPRL's strong performance, superior safety guarantees (measured by CVaR), and ability to produce actionable policies through selective deferral to the behavior policy. The high deferral rate in complex real-world tasks (e.g., MIMIC III) makes the learned policy interpretable and reviewable by experts.

Limitations and Future Work:

- DPRL, particularly DPRL-C, is non-parametric and requires storing the entire dataset, which can be computationally costly for very large datasets. Future work could explore data compression techniques (e.g., coresets).
- The use of Euclidean distance for continuous states could be extended to more sophisticated metrics like bisimulation distance, and their influence on safety bounds should be investigated.
- DPRL-C currently performs only 1-step planning over the behavior. Extending it to multi-step planning while maintaining safety guarantees is a promising direction.

---

</details>

### 📚 Citation

```bibtex
@article{sharma2024decision,
  title={Decision-point guided safe policy improvement},
  author={Sharma, Abhishek and Benac, Leo and Parbhoo, Sonali and Doshi-Velez, Finale},
  journal={arXiv preprint arXiv:2410.09361},
  year={2024}
}
```

## [Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning](https://arxiv.org/abs/2506.06873)
**Authors**: Armin Behnamnia, Gholamali Aminian, Alireza Aghaei, Chengchun_Shi1, Vincent Y. F. Tan, Hamid R. Rabiee
**Conference**: ICML 2025
**Tags**:  off-policy learning, off-policy evaluation, log sum exponential, regret bound, generalization bound, concentration, bias and variance

---

<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea

The paper introduces a novel estimator, the Log-Sum-Exponential (LSE) estimator, for Off-Policy Evaluation (OPE) and Off-Policy Learning (OPL) from logged bandit feedback (LBF) datasets. The core idea is to leverage the robustness properties of the LSE operator, particularly for negative parameters, to address challenges such as high variance, noisy (estimated) propensity scores, and heavy-tailed reward distributions that commonly plague traditional Inverse Propensity Score (IPS) estimators. The LSE estimator implicitly performs a form of shrinkage on large weighted reward values, leading to improved stability and performance.

---

### ❓ Problem Statement

#### What problem is the paper solving?

Off-policy evaluation and learning from logged bandit feedback face significant challenges in real-world applications. The primary issues include:

1. **High Variance of Estimators**: Traditional IPS estimators often suffer from high variance, leading to unreliable performance.
2. **Noisy or Estimated Propensity Scores**: Access to true propensity scores is frequently unavailable, necessitating their estimation, which introduces noise and further increases estimator variance.
3. **Heavy-tailed Reward Distributions**: In many real-life applications (e.g., financial markets, web advertising), reward distributions can be heavy-tailed, meaning their variance may not be well-defined or can be extremely large due to outliers or inherent noise, making standard estimators unstable.

Existing improved IPS estimators (e.g., truncated IPS, self-normalizing, exponential smoothing, power-mean) primarily focus on variance reduction but often assume bounded rewards or true propensity scores, failing to adequately address heavy-tailed conditions or handle noisy rewards/propensity scores effectively.

---

### 🎯 Motivation

The motivation stems from the limitations of current off-policy estimators, particularly IPS and its variants, when confronted with real-world complexities such as noise and heavy-tailed data. The LSE operator, characterized by \( \text{LSE}_\lambda(Z) = \frac{1}{\lambda} \log \left( \frac{1}{n} \sum_{i=1}^n e^{\lambda z_i} \right) \), inherently provides robustness for negative \( \lambda \). When \( \lambda \lt 0 \), terms with abnormally large positive \( z_i \) (noisy or outlier samples in the weighted reward) vanish in the exponential sum as \( \lim_{z_i \to +\infty} e^{\lambda z_i} = 0 \), thus being effectively ignored. This property makes the LSE estimator naturally robust to high-magnitude outliers in weighted rewards, a common characteristic of heavy-tailed distributions. A motivating toy example with a Pareto distribution demonstrates that LSE significantly reduces variance and MSE compared to the Monte-Carlo estimator for estimating the mean of a heavy-tailed variable, with minimal impact on bias. This intrinsic robustness for handling heavy-tailed and noisy data points motivates its application to off-policy evaluation and learning.

---

### 🛠️ Method Overview

The proposed method introduces the Log-Sum-Exponential (LSE) estimator for off-policy tasks.
The LSE estimator for a given target policy \( \pi_\theta \) and logged bandit feedback dataset \( S = (x_i, a_i, p_i, r_i)_{i=1}^n \) is defined as:

$$
 \hat{V}_{\lambda}^{\text{LSE}}(S, \pi_\theta) := \text{LSE}_\lambda(S) = \frac{1}{\lambda} \log \left( \frac{1}{n} \sum_{i=1}^n e^{\lambda r_i w_\theta(a_i,x_i)} \right)
$$
where \( \lambda \lt 0 \) is a tunable parameter, and \( w_\theta(a_i, x_i) = \frac{\pi_\theta(a_i|x_i)}{\pi_0(a_i|x_i)} \) is the importance weight for the \( i \)-th data point.

Key aspects of the LSE estimator:

- **Non-linearity**: Unlike many existing model-free estimators (e.g., IPS, truncated IPS, PM, ES, OS, IX, LS) which are linear transformations or weighted averages of individual weighted rewards, the LSE estimator is a non-linear function applied to the entire set of weighted reward samples. This non-linearity requires novel theoretical analysis techniques.
- **Parameter \( \lambda \lt 0 \)**:
  - As \( \lambda \to 0^- \), the LSE estimator converges to the standard IPS estimator: \( \lim_{\lambda \to 0} \hat{V}_{\lambda}^{\text{LSE}}(S, \pi_\theta) = \frac{1}{n} \sum_{i=1}^n r_i w_\theta(a_i, x_i) \).
  - As \( \lambda \to -\infty \), it converges to the minimum observed weighted reward: \( \lim_{\lambda \to -\infty} \hat{V}_{\lambda}^{\text{LSE}}(S, \pi_\theta) = \min_i r_i w_\theta(a_i, x_i) \).
  - This property implies an implicit shrinkage effect for large positive weighted rewards when \( \lambda \) is negative, as terms with large \( r_i w_\theta(a_i, x_i) \) are suppressed by \( e^{\lambda (\cdot)} \) when \( \lambda \lt 0 \).
- **Connection to KL Regularization**: The LSE estimator with \( \lambda \lt 0 \) can be interpreted as the solution to a KL-regularized expected minimization problem, connecting it to concepts of entropy regularization.
- **Off-Policy Evaluation (OPE)**: The objective is to estimate the value function \( V(\pi_\theta) = E_{P_X}[E_{\pi_\theta(A|X)}[r(A,X)|X]] \) by analyzing the bias and variance of \( \hat{V}_{\lambda}^{\text{LSE}}(S, \pi_\theta) \).
- **Off-Policy Learning (OPL)**: The objective is to find an optimal policy \( \pi_\theta^* = \arg \max_{\pi_\theta \in \Pi_\Theta} V(\pi_\theta) \) by maximizing the LSE estimator \( \hat{\pi}_\theta(S) = \arg \max_{\pi_\theta \in \Pi_\Theta} \hat{V}_{\lambda}^{\text{LSE}}(S, \pi_\theta) \) and bounding its regret, \( R_\lambda(\hat{\pi}_\theta, S) := V(\pi_\theta^*) - V(\hat{\pi}_\theta(S)) \).

---

### 📐 Theoretical Contributions

The paper provides comprehensive theoretical guarantees for the LSE estimator under challenging conditions.

1. Heavy-tail Assumption (Assumption 5.1):
The core assumption is that for all learning policies \( \pi_\theta(A|X) \in \Pi_\Theta \) and some \( \epsilon \in [0, 1] \), the \( (1 + \epsilon) \)-th moment of the weighted reward is bounded:
$$
E_{P_X \otimes \pi_0(A|X) \otimes P_{R|X,A}} \left[ \left(w_\theta(A, X)R\right)^{1+\epsilon} \right] \le \nu 
$$
This is a significantly weaker assumption than requiring bounded second or higher moments, enabling analysis under heavy-tailed distributions where variance might be infinite.

2. Regret Bounds in OPL (Theorem 5.3):
For a finite policy set \( |\Pi_\Theta| \lt \infty \), the paper establishes an upper bound on the regret \( R_\lambda(\hat{\pi}_\theta, S) \). The bound explicitly shows dependence on \( \lambda \), \( \nu \), \( n \) (sample size), \( \epsilon \), \( \delta \) (confidence parameter), and \( |\Pi_\Theta| \).

3. Convergence Rate (Proposition 5.4):
By strategically setting \( \lambda = -n^{- \frac{1}{1+\epsilon}} \), the paper demonstrates that the overall convergence rate of the regret upper bound is \( O(n^{-\epsilon/(1+\epsilon)}) \). Notably:

- If \( \epsilon = 1 \) (meaning the second moment of the weighted reward is bounded), the rate becomes \( O(n^{-1/2}) \), matching the optimal rate for many estimators under stronger assumptions.
- This rate is achieved even for unbounded weighted rewards, a key advantage over prior works that often require bounded rewards.

4. Bias and Variance Analysis in OPE:

- Bias Bounds (Proposition 5.5): Upper and lower bounds on the bias \( B(\hat{V}_{\lambda}^{\text{LSE}}(S, \pi_\theta)) \) are derived. The upper bound is \( O(|\lambda|^\epsilon \nu) \).
- Asymptotic Unbiasedness (Remark 5.6): By choosing \( \lambda \) to approach zero as \( n \to \infty \) (e.g., \( \lambda(n) = -n^{-\zeta} \) for \( \zeta \gt 0 \)), the LSE estimator is shown to be asymptotically unbiased. Specifically, setting \( \zeta = \frac{1}{1+\epsilon} \) yields an \( O(n^{-\epsilon/(1+\epsilon)}) \) convergence rate for bias.
- Variance Bound (Proposition 5.7): Assuming a bounded second moment of weighted reward (\( E[(w_\theta(A,X)R)^2] \le \nu_2 \)), the variance of the LSE estimator is bounded by \( V(\hat{V}_{\lambda}^{\text{LSE}}(S, \pi_\theta)) \le \frac{1}{n} V(w_\theta(A,X)R) \le \frac{1}{n} \nu_2 \). This demonstrates that the LSE estimator reduces variance compared to IPS.
- A bias-variance trade-off exists with \( \lambda \): decreasing \( |\lambda| \) (i.e., making \( \lambda \) closer to 0) decreases bias but increases variance, and vice-versa.

5. Robustness of the LSE Estimator:

- Noisy Reward (Theorem 5.9): The paper analyzes the regret of the LSE estimator when the reward distribution is perturbed by noise. The upper bound on regret includes a term proportional to \( TV(P_{R|X,A}, \tilde{P}_{R|X,A}) \), the total variation distance between the true and noisy reward distributions. This quantifies the cost of noise.
- Estimated Propensity Scores (Appendix E, Theorem E.7): Regret bounds are also derived under scenarios where propensity scores are estimated (e.g., modeled via Gamma noise), demonstrating the estimator's robustness to this common real-world issue.


6. Comparison with Other Estimators:
The LSE estimator is theoretically compared (Table 2) with IPS, SN-IPS, truncated IPS, IX, PM, ES, LS, and OS estimators. LSE demonstrates superior performance in heavy-tailed scenarios, robust to noisy rewards and estimated propensity scores, and maintains differentiability, which is important for optimization. It achieves a better convergence rate under heavy-tailed conditions compared to existing methods.

---

### 📊 Experiments

The theoretical findings are complemented by extensive empirical evaluations in both off-policy evaluation (OPE) and off-policy learning (OPL) scenarios.

#### Off-policy Evaluation (OPE)

- Baselines: Truncated IPS, PM, ES, IX, SNIPS, LS-LIN, LS, and OS estimators.
- Datasets:
  - Synthetic: Utilized a single-context LBF dataset where learning and logging policies are Gaussian distributions, and the reward function is an unbounded positive exponential function (\( e^{\alpha x^2} \)). The parameter \( \alpha \) is varied to control the tail behavior, demonstrating the estimator's performance under heavy-tailed weighted rewards. Experiments with Lomax distributions for policies are also performed.
  - UCI datasets: Additional experiments are conducted on various UCI datasets.
- Metrics: Bias, variance, and Mean Squared Error (MSE).
- Results: The LSE estimator consistently exhibits superior performance in terms of both MSE and variance, especially under heavy-tailed conditions, outperforming all baselines.

#### Off-policy Learning (OPL)

- Baselines: Truncated IPS, PM, ES, IX, BanditNet, LS-LIN, and OS estimators.
- Datasets:
  - EMNIST/FMNIST: Standard supervised-to-bandit transformation is applied to EMNIST (and FMNIST in Appendix), where each class corresponds to an action, and reward is binary (1 for correct label, 0 otherwise).
  - KUAIREC: A real-world dataset is used for additional validation.
- Noisy (Estimated) Propensity Scores: Modeled by introducing multiplicative inverse Gamma noise \( U \sim \text{Gamma}(b,b) \) such that \( b\pi_0 = \frac{1}{U}\pi_0 \).
- Noisy Reward: Simulated by a reward-switching probability \( P_f \), where a reward of 1 can switch to 0 with probability \( P_f \).
- Logging Policy Quality: Varied using an inverse temperature parameter \( \tau \) in the softmax layer, allowing for evaluation under more uniform and less accurate logging policies (higher \( \tau \)).
- Metric: Accuracy of the learned deterministic policy.
- Results: The LSE estimator achieves the highest accuracy with lower variance across most scenarios, including those with true, estimated propensity scores, and noisy rewards, demonstrating its practical advantages over state-of-the-art algorithms.

---

### 📈 Key Takeaways

1. Novelty and Robustness: The paper introduces a novel Log-Sum-Exponential (LSE) estimator that is inherently robust to outliers and heavy-tailed weighted reward distributions due to the properties of the LSE operator with negative parameters.
2. Weakened Assumptions: It provides rigorous theoretical guarantees for off-policy evaluation and learning under the weaker assumption of bounded \( (1+\epsilon) \)-th moments of weighted reward, unlike many existing methods requiring bounded second or higher moments.
3. Strong Theoretical Guarantees: The LSE estimator achieves a regret convergence rate of \( O(n^{-\epsilon/(1+\epsilon)}) \), including the optimal \( O(n^{-1/2}) \) when the second moment is bounded. It is also shown to be asymptotically unbiased and exhibits variance reduction.
4. Handling Noisy Data: The estimator's robustness is formally analyzed and demonstrated for both noisy (estimated) propensity scores and noisy reward samples, which are common issues in real-world logged datasets.
5. Empirical Superiority: Extensive experiments on synthetic and real-world datasets confirm the practical advantages of the LSE estimator, showing improved accuracy and reduced variance/MSE compared to several state-of-the-art baselines in various challenging scenarios.

---

</details>

### 📚 Citation

```bibtex
@Article{Behnamnia2025LogSumExponentialEF,
 author = {Armin Behnamnia and Gholamali Aminian and Alireza Aghaei and Chengchun Shi and Vincent Y. F. Tan and Hamid R. Rabiee},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning},
 volume = {abs/2506.06873},
 year = {2025}
}
```

## [Monte-Carlo tree search with uncertainty propagation via optimal transport](https://arxiv.org/abs/2309.10737)
**Authors**: Tuan Dam, Pascal Stenger, Lukas Schneider, J. Pajarinen, Carlo D'Eramo, Odalric-Ambrym Maillard 
**Conference**: ICML 2025
**Tags**: MCTS, Optimal Transport, Wasserstein Barycenter

---

<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea

This paper introduces Wasserstein Monte-Carlo Tree Search (W-MCTS), a novel MCTS algorithm designed for highly stochastic and partially observable Markov Decision Processes (MDPs) and Partially Observable MDPs (POMDPs). The core idea is to model both value and action-value nodes in the search tree as Gaussian distributions, enabling the propagation of uncertainty estimates throughout the tree via a novel backup operator. This operator computes value nodes as the \( L_1 \)-Wasserstein barycenter of their action-value children nodes. The paper establishes a connection between this Wasserstein barycenter, when combined with \( \alpha \)-divergence as the distance measure, and the generalized mean backup operator previously introduced in Power-UCT. W-MCTS is complemented by two exploration strategies: optimistic selection and Thompson sampling. The authors provide theoretical guarantees of asymptotic convergence to the optimal policy and demonstrate superior empirical performance over state-of-the-art baselines in several challenging environments.

---

### ❓ Problem Statement

#### What problem is the paper solving?

Traditional MCTS methods, particularly those coupled with deep learning, have achieved remarkable success in deterministic problems (e.g., Go, Chess). However, their performance significantly degrades in environments characterized by high stochasticity (unpredictable transitions) and partial observability (incomplete state information). In such settings, value estimates become highly uncertain and inaccurate. This inaccuracy propagates through the search tree, leading to suboptimal action selection at the root node and, consequently, poor overall performance. The challenge lies in developing an MCTS framework that can explicitly model, propagate, and effectively utilize this uncertainty during the tree search and backup phases.

---

### 🎯 Motivation

The motivation stems from the limitations of current MCTS approaches in handling real-world complex scenarios that are inherently stochastic and partially observable. While deep reinforcement learning has advanced planning capabilities, its application to high-uncertainty domains remains challenging due to the difficulty in obtaining reliable value function estimates. Existing MCTS methods often rely on point estimates of values, ignoring the inherent uncertainty, which can lead to issues like value overestimation. By adopting a probabilistic approach that models uncertainty as Gaussian distributions and propagates it using optimal transport (Wasserstein barycenters), W-MCTS aims to:

1. Provide more accurate and robust value estimates in uncertain environments.
2. Unify and generalize existing MCTS backup operators (like average and maximum backup) by leveraging the connection to the generalized mean.
3. Improve exploration-exploitation balance by incorporating uncertainty directly into action selection.
4. Extend the success of MCTS to a broader class of complex, stochastic, and partially observable problems.

---

### 🛠️ Method Overview

W-MCTS operates by modeling every node in the search tree, both value (V-nodes) and action-value (Q-nodes), as Gaussian distributions \( N(m, \sigma^2) \), where \( m \) is the mean and \( \sigma^2 \) is the variance (or \( \sigma \) is the standard deviation).
The core of the method is a novel backup operator based on **Wasserstein barycenters with \( \alpha \)-divergence**.

1. V-posterior Definition: A V-node \( V(s) \) is defined as the \( L_1 \)-Wasserstein barycenter of its children Q-nodes \( Q(s,a) \) for actions \( a \in A \), given a policy \( \bar{\pi} \):
$\( V(s) \in \arg \inf_{V} E_{a \sim \bar{\pi}} \left[W_1(V, Q(s, a))\right] \)$
Here, \( W_1(\cdot, \cdot) \) is the \( L_1 \)-Wasserstein distance, and the cost function used within it is the \( \alpha \)-divergence \( D_{f_\alpha}(X||Y) \), which is a subclass of \( f \)-divergence defined by \( f_\alpha(x) = \frac{(x^\alpha - 1) - \alpha(x - 1)}{\alpha(\alpha - 1)} \).

2. Closed-Form Solutions for Gaussians (Proposition 1): For Gaussian distributions \( V(s) \sim N(\bar{m}(s),\bar{\sigma}^2(s)) \) and \( Q(s,a) \sim N(m(s,a), \sigma^2(s,a)) \), the mean and standard deviation of the V-posterior are derived as power means:
$\( \bar{m}(s) = \left(E_{a \sim \bar{\pi}} [m(s, a)^p]\right)^{\frac{1}{p}} \)$
$\( \bar{\sigma}(s) = \left(E_{a \sim \bar{\pi}} [\sigma(s, a)^p]\right)^{\frac{1}{p}} \)$
where \( p = 1 - \alpha \). When \( p=1 \), these reduce to the expected mean and standard deviation.

3. Particle Filter Extension (Proposition 2): The approach is also shown to apply to particle models, where each particle \( \bar{x}_i(s) \) of a V-posterior \( V(s) \) is the power mean of corresponding particles \( x_i(s,a) \) from Q-posteriors:
$\( \bar{x}_i(s) = \left(E_{a \sim \bar{\pi}} [x_i(s, a)^p]\right)^{\frac{1}{p}} \)$

4. Empirical Backup Operators: In W-MCTS, the expectation over policy \( \bar{\pi} \) is approximated by the visitation count ratio \( \frac{n(s,a)}{N(s)} \). The backup rules for V-nodes are:
$\( V_m(s, N(s)) \leftarrow \left(\sum_a \frac{n(s, a)}{N(s)} Q_m(s, a, n(s, a))^p\right)^{\frac{1}{p}} \)$
$\( V_{std}(s, N(s)) \leftarrow \left(\sum_a \frac{n(s, a)}{N(s)} Q_{std}(s, a, n(s, a))^p\right)^{\frac{1}{p}} \)$
For Q-nodes, the updates are Bellman-like, using empirical averages of observed rewards and discounted V-node values of next states:
$\( Q_m(s, a, n(s, a)) \leftarrow \frac{\sum_{\text{samples }i} r_i(s, a) + \gamma \sum_{\text{samples }i} V_m(s'_i, N(s'_i))}{n(s, a)} \)$
$\( Q_{std}(s, a, n(s, a)) \leftarrow \gamma \frac{\sum_{\text{samples }i} V_{std}(s'_i, n(s'_i))}{n(s, a)} \)$
(Note: The paper's empirical notation \( P r(s,a) \) and \( P_{s'} N(s')V_m(s', N(s')) \) is interpreted as summations over observed samples.)

5. Action Selection Strategies:

   - Optimistic Selection (W-MCTS-OS): Inspired by UCB, this strategy selects actions based on an upper confidence bound that incorporates the estimated standard deviation:
$\( a = \arg \max_{a_i, i \in \{1...K\}} m(s, a_i) + C\sigma_i(s, a_i)p\sqrt{\log N(s)} \)$
where \( C \) is an exploration constant.
  - Thompson Sampling (W-MCTS-TS): Actions are selected by sampling from the posterior Gaussian distributions of action-values:
$\( a = \arg \max_{a_i, i \in \{1...K\}} \{\theta_i \sim N(m(s, a_i), \sigma^2(s, a_i))\} \)$

---

### 📐 Theoretical Contributions

The paper provides significant theoretical contributions, particularly regarding the convergence properties of W-MCTS, leveraging analysis from non-stationary Multi-Armed Bandits (MABs).

1. Wasserstein Non-stationary MAB Analysis:
   - Assumption 1 (Gaussian Rewards & Convergence): Assumes rewards for each arm \( k \) are Gaussian \( N(\mu_k, V_k/T_k(n)) \) and that the empirical mean \( \mu_{k,n} \) converges to \( \mu_k \).
   - Theorem 1 (Expected Suboptimal Arm Plays): For Thompson Sampling, the expected number of times a suboptimal arm \( k \) is played up to time \( n \), \( E[T_k(n)] \), is bounded polynomially:
$\( E[T_k(n)] \le \Theta \left(\frac{1 + V \log(n\Delta_k^2/V)}{\Delta_k^2}\right) \)$
where \( V = \max_k \{V_k\} \) and \( \Delta_k = \mu^* - \mu_k \) is the gap to the optimal mean \( \mu^* \).
   - Theorem 2 (Convergence of Expected Power Mean): The bias of the expected power mean backup operator \( X_n(p) \) (at the root of the MAB) from the optimal mean \( \mu^* \) converges polynomially:
$\( E[X_n(p)] - \mu^* \le |\delta^*_n| + \Theta \left(\frac{(K - 1)(1 + V \log(n\Delta^2/V))}{\Delta^2 n}\right)^{\frac{1}{p}} \)$
where \( K \) is the number of arms, \( \Delta = \max_k \{\Delta_k\} \), and \( |\delta^*_n| \) accounts for non-stationarity.
   - Theorem 3 (Concentration of Power Mean): The power mean backup operator \( X_n(p) \) concentrates polynomially around the optimal mean \( \mu^* \): for any \( \epsilon \gt 0 \), there exist constants \( C_0, \alpha, \beta \gt 0 \) such that for sufficiently large \( n \):
$\( Pr\left( X_n(p) - \mu^* \ge \epsilon \right) \le C_0 n^{-\alpha}\epsilon^{-\beta} \)$
$\( Pr\left( X_n(p) - \mu^* \le -\epsilon \right) \le C_0 n^{-\alpha}\epsilon^{-\beta} \)$

2. Convergence in Monte-Carlo Tree Search (W-MCTS-TS):
   - Proposition 3 (Q-Value Concentration and Suboptimal Action Plays): Extends the MAB results to the MCTS tree. It proves polynomial concentration for the estimated Q-value mean at the root node, and a polynomial bound on the expected number of suboptimal action plays. It also shows polynomial concentration of the estimated V-value mean at the root towards the optimal Q-value.
   - Theorem 4 (Convergence of Failure Probability): The probability of W-MCTS-TS choosing a suboptimal action at the root node decays polynomially to zero:
$\( Pr\left( a_k = a_k^* \right) \le Cn^{-\alpha} \)$
for constants \( C, \alpha \gt 0 \) and sufficiently large number of simulations \( n \).
   - Theorem 5 (Convergence of Expected Payoff): The expected estimated mean value function at the root node converges polynomially to the optimal Q-value:
$\( E\left[V_m^{(0)}(s(0), n)\right] - Q_m^{(0)}(s(0), a_k^*) \le \Theta \left(\frac{2(K - 1)(1 + V \log(n\Delta^2/V))}{\Delta^2 n}\right) \)$
The paper highlights that it is the first to provide a specific polynomial convergence rate for MCTS with Thompson Sampling.

---

### 📊 Experiments

The W-MCTS algorithm (both OS and TS variants) was evaluated on a suite of stochastic and partially observable environments against several baselines: UCT, Power-UCT, DNG (Bayesian MCTS using Dirichlet-NormalGamma), and D2NG (extension of DNG to POMDPs). The performance metric was the mean of total discounted reward over multiple evaluation runs.

1. Fully Observable Highly-Stochastic Problems (MDPs):

- Environments: FrozenLake, NChain, RiverSwim, SixArms, Taxi. These environments vary in state space size, stochasticity, and exploration requirements.
- Results:
  - W-MCTS-TS consistently outperformed UCT, Power-UCT, and DNG across most environments.
  - In FrozenLake, W-MCTS-TS and W-MCTS-TS (p=1) performed best.
  - In NChain, both W-MCTS sampling methods outperformed UCT and Power-UCT, with Thompson Sampling showing faster convergence.
  - RiverSwim saw W-MCTS-OS converging fastest and achieving the best results.
  - SixArms, a highly stochastic environment, was effectively solved only by W-MCTS.
  - Taxi, a large and highly stochastic environment, was successfully handled primarily by W-MCTS-TS, which managed to pick up all three passengers, while other methods struggled.

2. Partially Observable Highly-Stochastic Problems (POMDPs):

- Environments: RockSample (11x11, 15x15, 15x35) and PocMan. These are challenging POMDPs requiring robust exploration and planning under uncertainty.
- Results:
  - RockSample: W-MCTS-TS outperformed UCT and D2NG in all three variants (different numbers of actions).
  - PocMan: W-MCTS-TS (p=100) showed superior performance compared to UCT and D2NG for higher numbers of samples (4096, 32768, 65536 simulations).

---

### 📈 Key Takeaways

1. Uncertainty Propagation: W-MCTS provides a robust framework for Monte-Carlo Tree Search in highly stochastic and partially observable environments by explicitly modeling and propagating uncertainty of value estimates via Wasserstein barycenters.
2. Generalized Mean Connection: The novel backup operator, based on \( L_1 \)-Wasserstein barycenters with \( \alpha \)-divergence, successfully unifies and generalizes common MCTS backup operators through its connection to the power mean, effectively tackling issues like value overestimation.
3. Theoretical Guarantees: The algorithm, particularly W-MCTS with Thompson sampling, is theoretically sound, offering polynomial convergence rates to the optimal policy and accurate value functions at the root node, a significant contribution to the MCTS literature.
4. Empirical Superiority: W-MCTS demonstrates strong empirical advantages over existing state-of-the-art MCTS and Bayesian MCTS baselines in diverse and challenging stochastic MDPs and POMDPs, highlighting its practical effectiveness.
5. Direct Convergence: Unlike some approaches that converge to a regularized value function with potential bias, W-MCTS converges directly to the original optimal value function, leading to more accurate action selection.

---

</details>

### 📚 Citation

```bibtex
@inproceedings{dam2023,
  author    = {Tuan Dam and Pascal Stenger and Lukas Schneider and J. Pajarinen and Carlo D'Eramo and Odalric-Ambrym Maillard},
  year      = {2023},
  title     = {Monte-Carlo tree search with uncertainty propagation via optimal transport},
  booktitle = {arXiv.org},
  doi       = {10.48550/arXiv.2309.10737},
}
```

## [A Dual Approach to Constrained Markov Decision Processes with Entropy Regularization](https://arxiv.org/abs/2110.08923)  
**Authors**: Donghao Ying, Yuhao Ding, J. Lavaei
**Conference**:  AISTATS 2022
**Tags**: Constrained MDP, Entropy Regularization, Lagrangian dual function

---
<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea

This paper studies entropy-regularized Constrained Markov Decision Processes (CMDPs) under the soft-max parameterization. The core idea is to leverage entropy regularization to induce favorable optimization properties in the Lagrangian dual problem. Specifically, the authors show that the Lagrangian dual function becomes smooth (i.e., differentiable with a Lipschitz continuous gradient). This smoothness allows the application of accelerated first-order methods to the dual problem, leading to strong global convergence guarantees for both the dual optimality gap and, importantly, the primal optimality gap and constraint violation. The work provides the first theoretical analysis certifying the effectiveness of entropy regularization in CMDPs from an optimization perspective.

---

### ❓ Problem Statement

#### What problem is the paper solving?

The paper addresses the optimization problem for an agent aiming to maximize an entropy-regularized value function while satisfying multiple constraints on the expected total utility. This can be formally stated as:

$$
\max_{\pi \in \Pi} V^\pi_\tau(\rho) \quad \text{s.t.} \quad U^\pi_g(\rho) \geq b
$$

where \( V^\pi_\tau(\rho) := V^\pi(\rho) + \tau \cdot H(\rho, \pi) \) is the entropy-regularized value function, \( V^\pi(\rho) \) is the standard discounted sum of rewards, \( H(\rho, \pi) \) is the discounted entropy, \( \tau \geq 0 \) is the regularization weight, \( U^\pi_g(\rho) := (U^\pi_{g_1}(\rho), \dots, U^\pi_{g_n}(\rho)) \in \mathbb{R}^n \) are the discounted utilities for \( n \) constraint functions \( g_i \), and \( b \in \mathbb{R}^n \) are the corresponding thresholds. \( \Pi \) denotes the class of soft-max parameterized policies. This primal problem is generally non-convex due to a non-concave objective and non-convex constraints, making it challenging to solve directly.

---

### 🎯 Motivation

Sequential decision-making in safety-critical systems often requires satisfying various constraints beyond simple objective optimization, leading to the study of CMDPs. While policy search methods have shown empirical success in CMDPs, and theoretical progress has been made on their non-asymptotic global convergence, these works typically do not incorporate entropy regularization. Entropy regularization is a popular technique in unconstrained reinforcement learning for encouraging exploration, preventing premature convergence, and improving robustness. Recent theoretical work has also demonstrated that entropy regularization can lead to benign optimization landscapes and faster convergence rates in unconstrained MDPs. However, the theoretical benefits of entropy regularization for CMDPs, even in tabular settings with exact value evaluation, remained largely unknown. This paper aims to bridge this gap by investigating the optimization properties induced by entropy regularization in CMDPs.

---

### 🛠️ Method Overview

The proposed method, "Accelerated Gradient Projection Method with NPG Subroutine" (Algorithm 1), is a two-loop primal-dual algorithm:

1. Outer Loop (Accelerated Gradient Projection for Dual Variable):
This loop updates the dual variable \( \lambda \in \mathbb{R}^n_{\ge 0} \) using an accelerated gradient projection method onto a compact convex set \( \Lambda = \{\lambda \mid 0 \le \lambda_i \le \frac{V_*^\tau - V_\tau^\pi(\rho)}{\xi_i}, \forall i \in [n] \} \).

- At each iteration \( k \), an extrapolation point \( \mu^{(k)} = \lambda^{(k)} + \beta_k (\lambda^{(k)} - \lambda^{(k-1)}) \) is computed.
- An approximate gradient \( \tilde{\nabla} D(\mu^{(k)}) \) of the dual function \( D(\lambda) = \max_{\pi \in \Pi} L(\pi, \lambda) \) is estimated, where \( L(\pi, \lambda) := V^\pi_\tau(\rho) + \lambda^T (U_g^\pi(\rho) - b) \). The gradient is given by \( \nabla D(\lambda) = U_g^{\pi_\lambda}(\rho) - b \), where \( \pi_\lambda = \arg\max_{\pi \in \Pi} L(\pi, \lambda) \).
- A gradient projection step is performed: \( \lambda^{(k+1)} = P_\Lambda(\mu^{(k)} - \alpha_k \tilde{\nabla} D(\mu^{(k)})) \), where \( P_\Lambda(\cdot) \) is the projection onto \( \Lambda \).
- The step-size \( \alpha_k = 1/\mathcal{L} \) and extrapolation weight \( \beta_k = (k-1)/(k+2) \) are used, where \( \mathcal{L} \) is the smoothness constant of \( D(\lambda) \).

2. Inner Loop (Natural Policy Gradient Subroutine for Primal Variable): 
This subroutine (Algorithm 2, NPGSub) is called in the outer loop to find the Lagrangian maximizer \( \pi_\lambda \) for a given dual variable \( \lambda \). It uses the Natural Policy Gradient (NPG) method tailored for entropy-regularized MDPs.

- The policy update rule is given by:

$$
\pi^{(t+1)}(a|s) \propto \left(\pi^{(t)}(a|s)\right)^{1 - \frac{\eta\tau}{1-\gamma}} \exp \left( \frac{\eta Q_\tau^{\pi^{(t)}}(s, a)}{1-\gamma} \right)
$$
where \( Q_\tau^\pi(s, a) \) is the soft Q-function.

- This inner loop is run for a sufficient number of iterations \( N_2 = O(\log T) \) to ensure a high-accuracy gradient estimation for the outer loop.
- Upon termination of the outer loop, the final primal policy \( \pi \) is recovered from the last dual variable \( \lambda^{(N_1)} \) by running the NPG subroutine for \( N_3 = O(\log(1/\epsilon_1)) \) iterations.

The algorithm relies on the soft-max policy parameterization, which is crucial for the NPG update's direct form and the theoretical properties.

---

### 📐 Theoretical Contributions

The paper provides several key theoretical contributions, rigorously proving the benefits of entropy regularization for CMDPs:

1. Quadratic Lower Bound for Lagrangian: Proposition 3.5 shows that for all policies \( \pi \) and \( \lambda \ge 0 \), the Lagrangian function \( L(\pi, \lambda) \) is strongly concave (has a negative curvature) with respect to \( \pi \) at its maximizer \( \pi_\lambda \):
\( L(\pi_\lambda, \lambda) - L(\pi, \lambda) \ge \frac{\tau d^2}{2(1-\gamma)\ln 2} \| \pi - \pi_\lambda \|_2^2 \)
where \( d \) is a lower bound on the discounted state visitation distribution, and \( \tau \) is the entropy weight. This strong concavity is vital for stability and Lipschitz continuity.

2. Smoothness of Dual Function: Proposition 3.6 proves that, under the Slater condition (Assumption 3.1) and a uniform exploration assumption (Assumption 3.4, where \( d^\pi_\rho(s) \ge d \gt 0 \)), the Lagrangian dual function \( D(\lambda) \) is both differentiable and \( \mathcal{L} \)-smooth on its feasible domain \( \Lambda \). The gradient is \( \nabla D(\lambda) = U_g^{\pi_\lambda}(\rho) - b \), and the smoothness constant is:
\( \mathcal{L} = 2 \ln 2 \left( \frac{\sqrt{n|S||A|}}{(1-\gamma)^2} + \frac{\sqrt{n|S||A|}}{\tau (1-\gamma) d} \right) \)
This smoothness is a crucial enabler for accelerated gradient-based methods.

3. Decomposition of Duality Gap: Proposition 3.7 demonstrates a critical relationship between the dual optimality gap and primal metrics. If \( \lambda \) is an \( \epsilon \)-optimal dual multiplier (i.e., \( D(\lambda) - D_*^\tau \le \epsilon \)), then the associated Lagrangian maximizer \( \pi_\lambda \) satisfies:
   - Policy distance: \( \|\pi_\lambda - \pi_*^\tau\|_2 \le C_1\sqrt{\epsilon} \)
   - Primal optimality gap: \( |V_\tau^{\pi_\lambda}(\rho) - V_*^\tau| \le 2\epsilon + \mathcal{L}_c C_1 C_2 \sqrt{\epsilon} \)
   - Constraint violation: \( \max_{i \in [n]} [b_i - U_{g_i}^{\pi_\lambda}(\rho)]_+ \le \mathcal{L}_c C_1 \sqrt{\epsilon} \)
This shows that an \( O(\epsilon) \) dual error translates to an \( O(\sqrt{\epsilon}) \) primal error, a non-trivial result enabled by entropy regularization.

4. Global Convergence Rates: Theorem 5.2 establishes the global convergence rates for Algorithm 1:
   - Dual optimality gap: \( D(\lambda) - D_*^\tau = O(1/T^2) \)
   - Primal optimality gap: \( |V_\tau^{\pi}(\rho) - V_*^\tau| = O(1/T) \)
   - Constraint violation: \( \max_{i \in [n]} [b_i - U_{g_i}^{\pi}(\rho)]_+ = O(1/T) \)
   - Policy distance: \( \|\pi - \pi_*^\tau\|_2 = O(1/T) \)
The total iteration complexity is \( O(T \log T) \), where \( T \) is the number of outer loop iterations. This implies an overall \( O(1/\epsilon) \) rate to achieve \( \epsilon \)-accuracy in primal metrics.


5. Convergence for Standard CMDPs: Corollary 5.4 shows that by choosing a small regularization parameter \( \tau = O(\epsilon) \), the proposed method can compute an \( O(\epsilon) \)-optimal solution for the standard (unregularized) CMDP problem in \( O(1/\epsilon^2) \) total iterations.

6. Single Constraint CMDPs: For the special case of a single constraint (\( n=1 \)), Theorem 6.1 and Corollary 6.2 show that a bisection-based dual approach (Algorithm 3) achieves an even faster linear convergence rate in terms of outer loop iterations. Specifically, it achieves primal error bounds of \( O(\sqrt{\epsilon} + \epsilon_1) \) and dual error \( O(\epsilon) \) in \( O(\log^2(1/\epsilon) + \log(1/\epsilon_1)) \) total iterations. For standard CMDPs, it achieves \( O(\epsilon) \) primal optimality and constraint violation in \( O(\log^2(1/\epsilon)) \) iterations.

---

### 📊 Experiments

The provided paper does not include an experimental section. The analysis is purely theoretical, focusing on the optimization properties and convergence rates.

---

### 📈 Key Takeaways

- Entropy regularization significantly improves the optimization landscape of CMDPs by rendering their Lagrangian dual function smooth and convex, enabling the use of advanced first-order optimization methods.
- The paper provides a crucial theoretical link showing that an \( \epsilon \)-optimal dual solution in entropy-regularized CMDPs translates to an \( O(\sqrt{\epsilon}) \) error for primal optimality and constraint violation.
- The proposed accelerated dual-descent method with NPG as a subroutine achieves fast global convergence rates, specifically \( O(1/T) \) for primal metrics and \( O(1/T^2) \) for the dual function, for entropy-regularized CMDPs.
- For CMDPs with a single constraint, a bisection method can achieve a linear convergence rate for the dual problem, leading to an overall logarithmic total iteration complexity.
- By carefully choosing the regularization parameter \( \tau \), these results can be extended to provide near-optimal solutions for standard (unregularized) CMDPs with provable convergence rates.
- The work is foundational for understanding and developing more efficient algorithms for constrained reinforcement learning, particularly in settings where exact value/gradient evaluation is possible.

---
</details>

### 📚 Citation

```bibtex
@inproceedings{ying2022dual,
  title={A dual approach to constrained markov decision processes with entropy regularization},
  author={Ying, Donghao and Ding, Yuhao and Lavaei, Javad},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1887--1909},
  year={2022},
  organization={PMLR}
}
```

## [Temporal Difference Flows](https://arxiv.org/abs/2503.09817)
**Authors**: Jesse Farebrother, Matteo Pirotta, Andrea Tirinzoni, Rémi Munos, Alessandro Lazaric, Ahmed Touati
**Conference**: ICML 2025
**Tags**: Geometric Horizon Models, TD Learning, Flow Matching

---
<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea

The paper introduces Temporal Difference Flows (TD-Flow), a novel generative modeling approach for learning Geometric Horizon Models (GHMs). TD-Flow leverages a new Bellman equation defined on probability paths, combined with flow-matching and denoising diffusion techniques, to address the challenge of accurately predicting future states over long horizons. The core innovation lies in structuring the learning objective and sampling procedure to reduce the variance of sample-based gradient estimates, enabling stable and high-quality predictions far beyond the capabilities of prior methods.

---

### ❓ Problem Statement

#### What problem is the paper solving?

Traditional world models in Reinforcement Learning (RL) predict future states by unrolling dynamics step-by-step. This approach suffers from compounding errors, where small inaccuracies at each step accumulate, severely limiting their effectiveness for long-horizon reasoning and planning, a phenomenon referred to as the "curse of horizon". While Geometric Horizon Models (GHMs) offer an alternative by directly learning a generative model of future states, avoiding cumulative inference errors, existing methods face significant limitations during training. Specifically, their reliance on bootstrapped predictions (sampling from the model itself during training) leads to instability and growing inaccuracy over long horizons, restricting accurate predictions to typically 20-50 steps.

---

### 🎯 Motivation

Accurate and reliable long-horizon predictions are fundamental for intelligent agents to perform reasoning and planning in complex environments. Standard world models are inherently limited by compounding errors. GHMs, based on the successor measure and leveraging temporal difference (TD) learning, offer a promising path by directly modeling future state distributions. However, their training instability at long horizons necessitates a new approach. The paper is motivated by the insight that the iterative nature of modern generative models like flow matching and denoising diffusion, while not directly applicable to GHMs, can be adapted to better exploit the temporal difference structure of the successor measure problem. This adaptation aims to reduce the gradient variance associated with bootstrapped predictions, thereby enabling stable and accurate long-horizon predictions.

---

### 🛠️ Method Overview


#### Sccessor Measure

The normal **successor measure** of a policy \( \pi \) describes the discounted distribution of future states visisted by \( \pi \) starting from an initial state-action pair \( (s,a) \). For any policy \( \pi \), initial state-action pair \( (s,a)\in S×A \), and any measurable subset of states \( X\subset S \), the successor measure \( m^\pi(X∣s,a) \) is defined as the discounted, cumulative probability that the state trajectory falls within the set \( X \). The formal definition is given by the following equation:

$$
m^\pi (X | s, a) = (1 - \gamma) \sum_{k=0}^\infty \gamma^k \Pr(S_{k+1} \in X | S_0 = s, A_0 = a, \pi),
$$

where:
- \( \gamma \in[0,1) \) is the discount factor, which geometrically discounts the importance of future states.
- The term \( \Pr(S_{k+1} \in X | S_0 = s, A_0 = a, \pi) \) denotes the probability that the state at timestep \( k+1 \) is in the set \( X \), given that the agent started in state \( s \), took action \( a \), and subsequently followed policy \( \pi \).
- The summation \( \sum_{k=0}^\infty \) accounts for the entire future trajectory from the initial state-action pair.
- The normalization constant \( (1−\gamma) \) ensures that \( m^\pi(S∣s,a)=1 \), making it a valid probability measure over the state space.

A key advantage of the successor measure is its ability to decouple the environment's transition dynamics from the task-specific reward function, \( r(s) \). This allows for the efficient computation of the state-action value function, \( Q^\pi(s,a) \), for any reward function. The relationship is expressed as:

$$
Q^\pi(s, a) = (1 - \gamma)^{-1} \mathbb{E}_{X \sim m^\pi(\cdot|s,a)}[r(X)].
$$

The successor measure is the unique fixed point of a Bellman operator, \( \mathcal{T}^\pi: \mathcal{P}(S)^{S\times A} \rightarrow \mathcal{P}(S)^{S\times A} \). This operator provides a recursive definition for the successor measure, which is fundamental for its computation. The Bellman equation for the successor measure is:

$$
m^{\pi}(\cdot \mid s, a) = (\mathcal{T}^{\pi}m^{\pi})(\cdot \mid s, a) := (1 - \gamma)P(\cdot \mid s, a) + \gamma(P^{\pi}m^{\pi})(\cdot \mid s, a).
$$

The operator \( P^\pi \) mixes the one-step transition kernel with the successor measure from the subsequent states. It is formally defined as:

$$
(P^\pi m)(\mathrm{d}x \mid s, a) = \int_{s'} P(\mathrm{d}s' \mid s, a) m(\mathrm{d}x \mid s', \pi(s')).
$$

---

### 📐 Theoretical Contributions


---

### 📊 Experiments


---

### 📈 Key Takeaways


---
</details>

### 📚 Citation

```bibtex
@inproceedings{author2023title,
  title={Title},
  author={Author, First and Author, Second},
  booktitle={Conference Name},
  year={2023},
  url={URL}
}
```

## [Latent Diffusion Planning for Imitation Learning](https://arxiv.org/abs/2504.16925) 
**Authors**: Amber Xie, Oleh Rybkin, Dorsa Sadigh, Chelsea Finn
**Conference**: ICML 2025
**Tags**:  Imitation learning, Diffusion

---
<details markdown="1">
  <summary>Read More</summary>

### 🧠 Core Idea


---

### ❓ Problem Statement

#### What problem is the paper solving?


---

### 🎯 Motivation


---

### 🛠️ Method Overview


---

### 📐 Theoretical Contributions


---

### 📊 Experiments


---

### 📈 Key Takeaways


---
</details>

### 📚 Citation

```bibtex
@article{xie2025latent,
  title={Latent diffusion planning for imitation learning},
  author={Xie, Amber and Rybkin, Oleh and Sadigh, Dorsa and Finn, Chelsea},
  journal={arXiv preprint arXiv:2504.16925},
  year={2025}
}
```