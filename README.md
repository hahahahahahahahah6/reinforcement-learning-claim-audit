# Documentation claim auditor

Version 0.1.0. This offline audit tool compares documentation with Python source patterns for the checked capability. It uses the standard library at runtime, never executes inspected code, makes no network requests, and sends no telemetry. It reads files without modifying the inspected project.

## Installation and commands

```sh
python3 -m pip install .
reinforcement-learning-claim-audit --version
reinforcement-learning-claim-audit audit . --json
python3 -m reinforcement_learning_claim_audit audit . --verbose
reinforcement-learning-claim-audit audit --ignore vendor/ . --ignore generated/ --fail-on medium
python3 -m unittest discover
```

The repository argument defaults to the current directory. Each ignore option accepts a single relative path prefix and can be repeated. The threshold accepts high or medium, defaulting to high. Exit status is 0 when no finding meets the threshold, 2 when a finding meets it, and 1 for usage or file access errors. Findings remain visible below the threshold. JSON includes a verdict, findings with rule, severity and message fields, and production and test evidence paths. Verbose text output includes evidence paths.

## Inspection rules

The tool checks root README.md and Markdown files with a docs directory component. Fenced blocks are excluded; inline code counts as prose. Paragraphs are separated by blank lines. Matching lowercases text, expands negative contractions, replaces hyphens with spaces, and collapses whitespace. Exact phrases use word boundaries, longest first with alphabetical ties. Denial spans are removed before positive matching. Each family produces at most one finding.

```text
documented-reinforcement-learning-claim-false: HIGH without production evidence;
MEDIUM when evidence exists only in tests.
documented-no-reinforcement-learning-claim-false: HIGH for a denial contradicted
by production evidence.
Positive claim with no Python files: HIGH. Denial with no Python files: PASS.
```

Python files are parsed statically. Symbolic links are excluded. Files under fixtures, docs, or examples are excluded from evidence. Files under tests or named test_*.py or *_test.py provide test evidence only. Invalid syntax provides no evidence. Imported aliases are resolved; local function, method, and class definitions shadow route names. Each pair must occur in one module.

```text
Supported source patterns:
stable-baselines3: PPO / A2C / SAC / TD3 / DQN / DDPG construction plus .learn(),
including PPO(...).learn(...).
gym / gymnasium: make(...) plus .step(), including chains and imported make.
RLlib: PPOConfig / DQNConfig / SACConfig / TD3Config / A2CConfig / DDPGConfig
construction plus .build(), including .environment() / .training() chains.
Tianshou: PPOPolicy / A2CPolicy / DQNPolicy / SACPolicy / TD3Policy / DDPGPolicy
construction plus Collector(policy, ...) or Collector(policy=...), or
onpolicy_trainer / offpolicy_trainer with policy=...
Generic: 24 exact call names, as attributes on any base or imported bare names.
Examples include train_agent, run_episode, compute_reward, and train_ppo.
Bare imports, lone constructions, unbound calls, strings, comments, references,
subprocess commands, supervised .fit(), and plain reward arithmetic do not count.
```

## Limitations

Static matching does not establish runtime behavior or quality. Dead code counts. Unsupported alias chains can be missed. Exact phrases can misread unrelated meanings, and denial handling depends on the fixed list. Adjacent tasks such as categorization, forecasting, numeric estimation, grouping, outlier detection, recommendations, and supervised weight updates do not establish the checked capability.

```text
The short phrases ppo, dqn, and a2c can match unrelated acronyms or file formats.
"optimal action" and "value function" can occur in planning, control, or finance.
"reward function" can describe hand-written scoring without learning.
A gym.make + env.step pair counts even for random-action loops: quality is not checked.
Pure-torch RL, custom policy-gradient losses, and hand-rolled Q-learning loops
are not detected without a supported route.
An SB3 agent only loaded and .predict()ed without construction + .learn() does not count.
A Tianshou trainer without an explicit policy keyword does not count.
```

The test suite includes fixed phrase and operation-name oracles. The specification is not distributed. Distributed under the MIT license.
