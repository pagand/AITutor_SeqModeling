import numpy as np
import pandas as pd

# Interaction log
interaction_log = pd.DataFrame({
    "User ID": [1, 1, 2],
    "Skills": [["skill_1"], ["skill_1", "skill_3"], ["skill_2"]],
    "Correctness": [True, False, True],
    "Time Step": [1, 2, 1]
})

# Skill parameters
skills = {
    'skill_0': ([0.5, 0.3, 0.1, 0.1], []),
    'skill_1': ([0.6, 0.4, 0.2, 0.2], ['skill_0']),
    'skill_2': ([0.4, 0.3, 0.15, 0.15], ['skill_1']),
    'skill_3': ([0.7, 0.5, 0.25, 0.25], ['skill_0'])
}

# Initialize user-specific parameters
user_params = {}

def initialize_user_params(user_id, skills):
    user_params[user_id] = {}
    for skill in skills.keys():
        skill_params = skills[skill][0]
        user_params[user_id][skill] = {
            "P(L)": skill_params[0],
            "P(T)": skill_params[1],
            "P(G)": skill_params[2],
            "P(S)": skill_params[3]
        }

for user_id in interaction_log["User ID"].unique():
    initialize_user_params(user_id, skills)

# Function to compute P(C_t|L_t, G, S)
def compute_prob_correctness(P_L, P_G, P_S, correct):
    if correct:
        return (1 - P_S) * P_L + P_G * (1 - P_L)
    else:
        return P_S * P_L + (1 - P_G) * (1 - P_L)

# EM Optimization: E-Step
def expectation_step(user_data, user_params, skills):
    likelihoods = []
    for _, row in user_data.iterrows():
        skill_list = row["Skills"]
        correctness = row["Correctness"]

        for skill in skill_list:
            # Retrieve user and skill parameters
            P_L = user_params[skill]["P(L)"]
            P_G = user_params[skill]["P(G)"]
            P_S = user_params[skill]["P(S)"]

            # Compute likelihood
            likelihood = compute_prob_correctness(P_L, P_G, P_S, correctness)
            likelihoods.append(likelihood)

    return np.log(np.sum(likelihoods))

# EM Optimization: M-Step
def maximization_step(user_data, user_params, skills):
    for skill, skill_data in skills.items():
        total_correct = 0
        total_attempts = 0

        for _, row in user_data.iterrows():
            if skill in row["Skills"]:
                total_attempts += 1
                if row["Correctness"]:
                    total_correct += 1

        # Update probabilities
        user_params[skill]["P(L)"] = total_correct / total_attempts
        user_params[skill]["P(T)"] = np.random.uniform(0.1, 0.3)  # Random adjustment for transition
        user_params[skill]["P(G)"] = np.random.uniform(0.1, 0.3)  # Guessing parameter
        user_params[skill]["P(S)"] = np.random.uniform(0.1, 0.3)  # Slipping parameter

    return user_params

# Parent-Child Constraints
def enforce_constraints(user_params, skills):
    for skill, skill_data in skills.items():
        parents = skill_data[1]
        for parent in parents:
            for user in user_params:
                parent_prob = user_params[user][parent]["P(L)"]
                child_prob = user_params[user][skill]["P(L)"]
                if child_prob >= parent_prob:
                    user_params[user][skill]["P(L)"] = parent_prob - 0.01  # Apply heuristic

# Parallel Computation
def parallel_computation(interaction_log, skills, user_params):
    # Decompose into skill-specific and user-specific components
    skill_specific = {}
    user_specific = {}

    for skill, skill_data in skills.items():
        skill_specific[skill] = np.mean([user_params[user][skill]["P(L)"] for user in user_params])

    for user in user_params:
        user_specific[user] = {skill: user_params[user][skill]["P(L)"] - skill_specific[skill]
                               for skill in skills.keys()}

    return skill_specific, user_specific

# Full EM Algorithm
def run_em_algorithm(interaction_log, skills, user_params, max_iter=10):
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}")
        for user_id, user_data in interaction_log.groupby("User ID"):
            # E-Step
            likelihood = expectation_step(user_data, user_params[user_id], skills)
            print(f"  User {user_id} Likelihood: {likelihood:.4f}")

            # M-Step
            user_params[user_id] = maximization_step(user_data, user_params[user_id], skills)

        # Enforce Constraints
        enforce_constraints(user_params, skills)

    # Parallel Computation
    skill_specific, user_specific = parallel_computation(interaction_log, skills, user_params)
    return skill_specific, user_specific

# Run the EM Algorithm
skill_specific, user_specific = run_em_algorithm(interaction_log, skills, user_params)

# Output Results
print("Skill-Specific Components:")
for skill, prob in skill_specific.items():
    print(f"  {skill}: {prob:.4f}")

print("\nUser-Specific Deviations:")
for user, params in user_specific.items():
    print(f"User {user}:")
    for skill, prob in params.items():
        print(f"  {skill}: {prob:.4f}")
