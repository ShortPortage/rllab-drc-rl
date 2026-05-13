# Extracted Paper Outline Relevant to RRPO

Source: `papers/Ma et al. 2025 - Rectified Robust Policy Optimization for model-uncertain constrained reinforcement learning without strong duality.pdf`, converted with `pdftotext -layout` to `papers/rrpo-algorithm/paper.txt`.

Relevant sections:

1. **Introduction**
   - Motivation: robust constrained RL must optimize worst-case reward while satisfying worst-case constraints under model uncertainty.
   - Key negative result: strong duality generally fails in robust constrained RL, so primal-dual methods can fail.
   - Contribution: Rectified Robust Policy Optimization (RRPO), a primal-only CRPO-style algorithm with a rectification mechanism and convergence guarantees.

2. **Section 2.1: Robust MDPs**
   - Robust MDP tuple `(S,A,P,r,gamma)`.
   - General uncertainty sets, including s-rectangular and `(s,a)`-rectangular sets.
   - Robust value function as worst-case expected discounted reward.

3. **Section 2.2: Robust Constrained MDPs**
   - Multiple reward/constraint reward functions `r_i`, `i=0,...,I`.
   - Robust values `V_i^pi`; `V_0` is objective, `V_i`, `i>=1`, are constraint values.
   - Objective: maximize `V_0^pi(mu)` subject to `V_i^pi(mu) >= d_i`.

4. **Section 2.3 and Section 3: Duality Gap**
   - Lagrangian formulation and duality gap definition.
   - Counterexample showing strictly positive duality gap.
   - Implication: algorithms relying on strong duality are not generally justified.

5. **Section 4.1: Algorithm Design**
   - Reformulate with auxiliary objective threshold `d_0`:
     `max_{d_0,pi} d_0` subject to `V_0^pi(mu) >= d_0` and `V_i^pi(mu) >= d_i`.
   - RRPO categories:
     - threshold update when feasible and objective meets current threshold;
     - constraint rectification when a constraint violates tolerance;
     - objective rectification when objective falls below tracked threshold.
   - Algorithm 1 pseudocode.

6. **Section 4.2: Handling Uncertainty**
   - Robust policy evaluation module estimates robust Q-functions.
   - Robust natural policy gradient (NPG) update:
     `pi_{t+1}(a|s) proportional to pi_t(a|s) exp(eta Q_i^{pi_t}(s,a)/(1-gamma))`.
   - Under softmax parameterization, equivalent to `theta_{t+1}(s,a)=theta_t(s,a)+eta Q_i^{pi_t}(s,a)`.
   - p-norm and IPM uncertainty examples.

7. **Section 4.3--4.4: Guarantees and Complexity**
   - Assumptions: robust policy evaluation accuracy, worst-case exploration, bounded uncertainty diameter.
   - Theorem 4.5: approximate optimality and constraint violation guarantees with `O(1/sqrt(T)) + O(c)` value error and `delta` constraint violation.
   - Sample complexity discussion: `O(epsilon^-4)` in the p-norm example when each robust Q-evaluation costs `O(epsilon^-2)` and `T=O(epsilon^-2)` iterations are used.

8. **Appendix C.2--C.5**
   - Robust performance difference lemma.
   - NPG improvement lemmas and proof structure.
   - Robust policy evaluation options.
   - Compact Algorithm 3.

No repository code is available for RRPO, so the final algorithm document should rely on the paper and clearly label implementer choices where code would normally disambiguate details.
