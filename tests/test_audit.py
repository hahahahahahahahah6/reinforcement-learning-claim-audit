import ast
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from reinforcement_learning_claim_audit import phrases
from reinforcement_learning_claim_audit.audit import audit, claims, evidence, main

EXPECTED = {'REINFORCEMENTLEARNING': ('reinforcement learning', 'reinforcement learning model', 'reinforcement learning models', 'deep reinforcement learning', 'trains an rl agent', 'trains rl agents', 'uses reinforcement learning', 'learns from rewards', 'learns by trial and error', 'learns through interaction', 'learns a policy', 'learns an optimal policy', 'trains a policy', 'policy gradient', 'policy gradients', 'proximal policy optimization', 'q learning', 'deep q learning', 'deep q network', 'actor critic', 'soft actor critic', 'multi armed bandit', 'multi armed bandits', 'markov decision process', 'maximizes reward', 'maximises reward', 'cumulative reward', 'reward based learning', 'reward function', 'exploration and exploitation', 'epsilon greedy', 'temporal difference', 'monte carlo learning', 'self play', 'trains with reinforcement learning', 'agent learns from interaction', 'trial and error learning', 'reward signal', 'learns to act', 'optimal action', 'value function', 'learns a value function', 'bandit algorithm', 'rl agent', 'rl agents', 'rl model', 'rl models', 'ppo', 'dqn', 'a2c'), 'NOREINFORCEMENTLEARNING': ('no reinforcement learning', 'does not do reinforcement learning', 'without reinforcement learning', 'no rl agent', 'does not support reinforcement learning', 'reinforcement learning not supported', 'does not learn from rewards', 'no policy learning', 'cannot learn a policy', 'never learns from rewards', 'no reward based learning', 'does not use reinforcement learning', 'no rl model', 'does not train an agent', 'supervised learning only', 'classification only', 'no trial and error', 'does not learn by trial and error', 'no exploration', 'rule based only', 'scripted behavior only', 'scripted behaviour only', 'hardcoded policy only', 'no agent training'), 'GENERIC_NAMES': ('train_agent', 'train_policy', 'learn_policy', 'run_rl', 'train_rl', 'train_rl_agent', 'run_episode', 'run_episodes', 'collect_rollout', 'collect_rollouts', 'update_policy', 'compute_reward', 'select_action', 'epsilon_greedy_action', 'q_learning_step', 'policy_gradient_step', 'train_ppo', 'train_dqn', 'train_sac', 'train_a2c', 'train_td3', 'train_ddpg', 'run_ppo', 'run_dqn'), 'SB3_NAMES': ('PPO', 'A2C', 'SAC', 'TD3', 'DQN', 'DDPG'), 'RLLIB_NAMES': ('PPOConfig', 'DQNConfig', 'SACConfig', 'TD3Config', 'A2CConfig', 'DDPGConfig'), 'TIANSHOU_NAMES': ('PPOPolicy', 'A2CPolicy', 'DQNPolicy', 'SACPolicy', 'TD3Policy', 'DDPGPolicy')}


class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        tree = ast.parse(Path(phrases.__file__).read_text())
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)}
        for key in ('REINFORCEMENTLEARNING', 'NOREINFORCEMENTLEARNING'):
            self.assertEqual(values[key], EXPECTED[key])
            self.assertEqual(len(values[key]), len(set(values[key])))

    def test_generic_names(self):
        self.assertEqual(phrases.GENERIC_NAMES, EXPECTED['GENERIC_NAMES'])
        self.assertEqual(len(set(phrases.GENERIC_NAMES)), 24)
        for name in EXPECTED['GENERIC_NAMES']:
            self.assertTrue(evidence('x.'+name+'()'))
            self.assertTrue(evidence('from service import '+name+' as f; f()'))
            self.assertFalse(evidence(name+'()'))
            self.assertFalse(evidence('def '+name+'(): pass\nx.'+name+'()'))

    def test_sb3_names(self):
        self.assertEqual(phrases.SB3_NAMES, EXPECTED['SB3_NAMES'])
        self.assertEqual(len(set(phrases.SB3_NAMES)), 6)
        for name in EXPECTED['SB3_NAMES']:
            self.assertTrue(evidence('from stable_baselines3 import '+name+' as M; m=M(); m.learn(1)'))

    def test_rllib_names(self):
        self.assertEqual(phrases.RLLIB_NAMES, EXPECTED['RLLIB_NAMES'])
        self.assertEqual(len(set(phrases.RLLIB_NAMES)), 6)
        for name in EXPECTED['RLLIB_NAMES']:
            self.assertTrue(evidence('from ray.rllib import '+name+' as C; c=C(); c.build()'))

    def test_tianshou_names(self):
        self.assertEqual(phrases.TIANSHOU_NAMES, EXPECTED['TIANSHOU_NAMES'])
        self.assertEqual(len(set(phrases.TIANSHOU_NAMES)), 6)
        for name in EXPECTED['TIANSHOU_NAMES']:
            self.assertTrue(evidence('from tianshou import '+name+' as P, Collector as C; p=P(); C(p, env)'))

    def test_routes(self):
        sources = [
            'from stable_baselines3 import PPO; m=PPO(); m.learn(100)',
            'from stable_baselines3 import DQN; m=DQN(); m.learn(100)',
            'from stable_baselines3 import PPO; PPO().learn(100)',
            'import stable_baselines3 as s; s.PPO().learn(100)',
            'import gymnasium as gym; env=gym.make("x"); env.step(a)',
            'import gymnasium as gym; gym.make("x").step(a)',
            'from gymnasium import make as create; env=create("x"); env.step(a)',
            'import gym; gym.make("x").step(a)',
            'from ray.rllib.algorithms.ppo import PPOConfig; c=PPOConfig().environment("x").training(x=1); c.build()',
            'from ray.rllib.algorithms.ppo import PPOConfig; PPOConfig().environment("x").build()',
            'from ray.rllib.algorithms.dqn import DQNConfig; c=DQNConfig(); c.build()',
            'from tianshou.policy import PPOPolicy; from tianshou.data import Collector; p=PPOPolicy(); Collector(p, env)',
            'from tianshou.policy import DQNPolicy; from tianshou.data import Collector; p=DQNPolicy(); Collector(policy=p)',
            'from tianshou import PPOPolicy, onpolicy_trainer; p=PPOPolicy(); onpolicy_trainer(policy=p)',
            'from tianshou import DQNPolicy, offpolicy_trainer; p=DQNPolicy(); offpolicy_trainer(policy=p)',
            'x.train_agent()', 'from service import run_episode as run; run()',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(evidence(source))

    def test_not_evidence(self):
        sources = [
            'import stable_baselines3, gym, gymnasium, ray, tianshou',
            'from stable_baselines3 import PPO; m=PPO()',
            'import stable_baselines3; m.learn(1)',
            'from stable_baselines3 import PPO; m=PPO(); m.predict(x)',
            'import stable_baselines3; m=Other(); m.learn(1)',
            'import gymnasium as gym; env=gym.make("x")',
            'import gymnasium; env.step(a)',
            'import gymnasium as gym; env=gym.make("x"); env.action_space.sample()',
            'import gymnasium as gym; env=gym.make("x"); env.reset()',
            'import gymnasium; env=Other(); env.step(a)',
            'env=make("x"); env.step(a)',
            'from ray import PPOConfig; c=PPOConfig()',
            'import ray; c.build()',
            'import ray; c=Other(); c.build()',
            'from ray import PPOConfig; c=PPOConfig(); c.train()',
            'from tianshou import PPOPolicy; p=PPOPolicy()',
            'from tianshou import Collector; Collector()',
            'from tianshou import Collector; p=Other(); Collector(p)',
            'from tianshou import PPOPolicy, onpolicy_trainer; p=PPOPolicy(); onpolicy_trainer(p)',
            'model.fit(X,y)', 'Classifier().predict(X)', 'Forecaster().forecast()',
            'Regressor().predict(X)', 'KMeans().fit_predict(X)', 'IsolationForest().fit_predict(X)',
            'recommender.recommend(x)', 'model.fine_tune(x)', 'reward=score+1',
            'import subprocess; subprocess.run(["train_rl"])',
            'x.train_agent', '"x.train_agent()"', '# x.train_agent()', 'invalid !!!',
            'from service import train_agent\ndef train_agent(): pass\ntrain_agent()',
            'from stable_baselines3 import PPO\nclass PPO: pass\nPPO().learn(1)',
            'from gymnasium import make\ndef make(x): pass\nmake("x").step(a)',
            'from ray import PPOConfig\nclass PPOConfig: pass\nPPOConfig().build()',
            'from tianshou import PPOPolicy, Collector\nclass C:\n def Collector(self): pass\np=PPOPolicy(); Collector(p)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_matching(self):
        for phrase in EXPECTED['REINFORCEMENTLEARNING']:
            self.assertTrue(claims(phrase)[0], phrase)
        for phrase in EXPECTED['NOREINFORCEMENTLEARNING']:
            self.assertEqual(claims(phrase), ([], [phrase]))
        self.assertEqual(claims("doesn't do reinforcement-learning"), ([], ['does not do reinforcement learning']))
        self.assertEqual(claims('reinforcement learning models')[0], ['reinforcement learning models'])
        self.assertFalse(claims('xppo ppoz reinforcement\n\nlearning')[0])
        self.assertTrue(claims('reinforcement\nlearning')[0])
        self.assertTrue(claims('`ppo`')[0])
        for fence in ('```', '~~~', '   ````'):
            self.assertEqual(claims(fence+'text\nppo\n'+fence.strip()), ([], []))

    def fixture(self, root, doc='reinforcement learning', source=None, filename='app.py'):
        (root/'README.md').write_text(doc)
        if source is not None:
            path=root/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def cli(self, args):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            status=main(args)
        return status, out.getvalue()

    def test_findings_and_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.fixture(root)
            status, output=self.cli(['audit', d, '--json'])
            result=json.loads(output)
            self.assertEqual(status, 2)
            self.assertEqual(result['verdict'], 'FAIL')
            self.assertEqual(result['findings'][0]['rule'], 'documented-reinforcement-learning-claim-false')
            self.assertEqual(result['findings'][0]['severity'], 'HIGH')
            self.assertTrue(result['findings'][0]['message'])
            self.fixture(root, source='x.train_agent()', filename='tests/test_impl.py')
            self.assertEqual(self.cli(['audit', d])[0], 0)
            self.assertEqual(audit(root)['findings'][0]['severity'], 'MEDIUM')
            self.assertEqual(self.cli(['audit', d, '--fail-on', 'medium'])[0], 2)
            self.fixture(root, source='x.train_agent()')
            self.assertEqual(audit(root)['verdict'], 'PASS')
            self.fixture(root, doc='no reinforcement learning')
            finding=audit(root)['findings'][0]
            self.assertEqual(finding['rule'], 'documented-no-reinforcement-learning-claim-false')
            self.assertEqual(finding['severity'], 'HIGH')
            self.assertEqual(self.cli(['audit', d])[0], 2)
            self.fixture(root, doc='A neutral description.')
            self.assertEqual(audit(root)['verdict'], 'PASS')

    def test_no_source_and_docs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.fixture(root, doc='no reinforcement learning')
            self.assertEqual(audit(root)['verdict'], 'PASS')
            self.fixture(root, doc='```\nppo\n```')
            self.assertEqual(audit(root)['verdict'], 'PASS')
            self.fixture(root, doc='`ppo`')
            self.assertEqual(audit(root)['findings'][0]['severity'], 'HIGH')
            self.fixture(root, doc='neutral')
            path=root/'nested/docs/guide.md'; path.parent.mkdir(parents=True); path.write_text('ppo')
            self.assertEqual(audit(root)['verdict'], 'FAIL')
            self.assertEqual(audit(root, ['nested'])['verdict'], 'PASS')
            path.unlink(); (root/'other.md').write_text('ppo')
            self.assertEqual(audit(root)['verdict'], 'PASS')

    def test_excluded_paths(self):
        for name in ('tests/a.py','test_a.py','a_test.py','fixtures/a.py','docs/a.py','examples/a.py'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root=Path(d); self.fixture(root, source='x.train_agent()', filename=name)
                self.assertFalse(audit(root)['evidence']['production'])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); self.fixture(root)
            external=root/'fixtures'; external.mkdir(); (external/'a.py').write_text('x.train_agent()')
            (root/'linked.py').symlink_to(external/'a.py')
            (root/'linked').symlink_to(external, target_is_directory=True)
            (root/'docs').mkdir(); (root/'docs/claim.md').symlink_to(root/'README.md')
            self.assertFalse(audit(root)['evidence']['production'])

    def test_split_modules(self):
        pairs=[('from stable_baselines3 import PPO; m=PPO()', 'm.learn(1)'),
               ('import gymnasium as gym; e=gym.make("x")','e.step(a)'),
               ('from ray import PPOConfig; c=PPOConfig()', 'c.build()'),
               ('from tianshou import PPOPolicy; p=PPOPolicy()', 'from tianshou import Collector; Collector(p)')]
        for first, second in pairs:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d); self.fixture(root, source=first); (root/'second.py').write_text(second)
                self.assertEqual(audit(root)['findings'][0]['severity'], 'HIGH')

    def test_ignore_before_positional(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); self.fixture(root, source='x.train_agent()', filename='vendor/a.py')
            status, output=self.cli(['audit','--ignore','vendor',d,'--json'])
            self.assertEqual(status, 2)
            self.assertFalse(json.loads(output)['evidence']['production'])

    def test_errors_and_version(self):
        for args in ([], ['oops'], ['audit','--fail-on','HIGH'], ['audit','--ignore']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(args)
            self.assertEqual(exc.exception.code, 1)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.cli(['audit','/nonexistent/audit-root'])[0], 1)
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as exc:
            main(['--version'])
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), '0.1.0')

    def test_alias_shadowing(self):
        for source in (
            'from service import train_agent as run\n' + 'def train_agent(): pass\nrun()',
            'from stable_baselines3 import PPO as M\nclass PPO: pass\nM().learn(1)',
            'from ray import PPOConfig as C\nclass PPOConfig: pass\nC().build()',
            'from gymnasium import make as mk\ndef make(x): pass\nmk(1).step(a)',
            'from tianshou import PPOPolicy as P, Collector\nclass PPOPolicy: pass\np=P(); Collector(p)',
        ):
            self.assertFalse(evidence(source))
        self.assertTrue(evidence('class Service: pass\nService.train_agent()'))

    def test_io_and_repeatable_ignore(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.fixture(root, source='x.train_agent()', filename='one/a.py')
            (root/'two').mkdir(); (root/'two/b.py').write_text('x.train_agent()')
            self.assertEqual(self.cli(['audit', '--ignore', 'one', '--ignore', 'two', d])[0], 2)
            (root/'broken.py').write_bytes(b'\xff')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.cli(['audit', d])[0], 1)

    def test_module_entry(self):
        run=subprocess.run([sys.executable,'-m','reinforcement_learning_claim_audit','--version'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout.strip(), '0.1.0')

    @unittest.skipUnless(shutil.which('reinforcement-learning-claim-audit'), 'console script not installed')
    def test_console_script_entry(self):
        run=subprocess.run(['reinforcement-learning-claim-audit','--version'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout.strip(), '0.1.0')

    def test_readme(self):
        readme=Path(__file__).resolve().parents[1]/'README.md'
        if readme.exists():
            self.assertEqual(claims(readme.read_text()), ([], []))


if __name__ == '__main__':
    unittest.main()
