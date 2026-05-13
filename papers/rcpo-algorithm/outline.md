# Extracted Paper Outline Relevant to RCPO

Source: `papers/Constrained Reinforcement Learning Under Model Mismatch.pdf` converted with `pdftotext -layout` to `papers/rcpo-algorithm/paper.txt`.

Relevant sections:

1. **Introduction**
   - Motivation: constrained RL policies trained in simulators can violate constraints under model mismatch.
   - Contribution: RCPO for robust constrained RL with large/continuous state spaces and per-iteration reward/constraint guarantees.
   - Algorithm overview: robust policy improvement followed by projection.

2. **Section 3.1: Constrained MDP**
   - CMDP tuple `(S,A,p,r,c)`.
   - Reward value/action-value and utility value/action-value functions.
   - Discounted occupancy measure.
   - Non-robust CMDP objective: maximize discounted reward subject to utility threshold.

3. **Section 3.2: Robust MDP**
   - Robust MDP tuple `(S,A,P,r)`.
   - `(s,a)`-rectangular uncertainty set `P = \otimes_{s,a} P_s^a`.
   - Robust reward value and robust reward action-value as minima over transition kernels.
   - Worst-case transition kernel.

4. **Section 3.3: Problem Formulation**
   - Robust constrained MDP `(S,A,P,r,c)`.
   - Robust utility value as the minimum utility over `P`.
   - Main objective: `max_pi V_r^pi(rho)` subject to `V_c^pi(rho) >= d`.

5. **Section 4: Robust Constrained Policy Optimization**
   - Robust performance-difference lemma (Lemma 4.1).
   - Local approximation of robust value improvement using current policy's worst-case transition kernel.
   - RCPO design: robust policy improvement plus projection.

6. **Section 4.1: Algorithm Design**
   - Estimate worst-case reward kernel `p_r^k` with projected gradient descent.
   - Parameterized transition kernels for large/continuous spaces.
   - Robust policy improvement optimization, paper Eq. (13).
   - Estimate worst-case utility kernel `p_c^k` with projected gradient descent.
   - Projection optimization, paper Eq. (16).
   - Algorithm 1 pseudocode.

7. **Section 4.2: Theoretical Results**
   - Assumption 4.2: approximate worst-case kernels within `epsilon`.
   - Theorem 4.3: guarantees when current policy is feasible.
   - Theorem 4.4: guarantees when current policy is infeasible.

8. **Section 5: Practical Implementation**
   - Defines gradients `g`, `h`, and KL Hessian `H`.
   - First-order reward objective and second-order KL approximation.
   - Practical robust policy-improvement subproblem, Eq. (23).
   - Practical projection subproblem, Eq. (25).
   - Closed-form/natural-gradient update, Eq. (26).

9. **Appendices C--E**
   - Appendix C: first-order correctness of the approximate robust loss.
   - Appendix D: proof of feasible-case theorem and Bregman/KL projection property.
   - Appendix E: proof of infeasible-case theorem and enlarged KL bound.
