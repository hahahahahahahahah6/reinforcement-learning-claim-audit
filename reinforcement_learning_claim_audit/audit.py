"""Offline, non-executing documentation and Python source inspection."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import (REINFORCEMENTLEARNING, NOREINFORCEMENTLEARNING,
                      GENERIC_NAMES, SB3_NAMES, RLLIB_NAMES, TIANSHOU_NAMES)


def claims(text):
    """Return positive and denial matches after paragraph normalization."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            lines.append('')
        elif match:
            fence = match[1]
            lines.append('')
        else:
            lines.append(line)
    positive, negative = [], []
    for paragraph in re.split(r'\n\s*\n', '\n'.join(lines)):
        normalized = re.sub(r'\s+', ' ', paragraph.lower().replace("n't", ' not').replace('-', ' '))
        for phrases, found in ((NOREINFORCEMENTLEARNING, negative), (REINFORCEMENTLEARNING, positive)):
            for phrase in sorted(phrases, key=lambda x: (-len(x), x)):
                pattern = r'\b' + re.escape(phrase) + r'\b'
                if re.search(pattern, normalized):
                    found.append(phrase)
                    normalized = re.sub(pattern, lambda m: ' ' * len(m[0]), normalized)
    return positive, negative


def evidence(source):
    """Recognize supported call pairs within one parsed module."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    nodes = list(ast.walk(tree))
    shadows = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {}
    families = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
                families.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            families.add(node.module.split('.')[0])
            for alias in node.names:
                imports[alias.asname or alias.name] = node.module + '.' + alias.name
    def resolve(node):
        if isinstance(node, ast.Name):
            return '' if node.id in shadows else imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base and node.attr not in shadows else ''
        return ''
    def name(node):
        if isinstance(node, ast.Attribute):
            return node.attr if node.attr not in shadows else ''
        resolved = resolve(node).split('.')[-1]
        return resolved if resolved not in shadows else ''
    assignments = {}
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.value)
    def constructed(node, kind, seen=None):
        seen = set() if seen is None else seen
        if isinstance(node, ast.Name):
            if node.id in shadows or node.id in seen:
                return False
            return any(constructed(v, kind, seen | {node.id}) for v in assignments.get(node.id, []))
        if not isinstance(node, ast.Call):
            return False
        route = name(node.func)
        if kind == 'sb3':
            return 'stable_baselines3' in families and route in SB3_NAMES
        if kind == 'gym':
            return route == 'make' and resolve(node.func) in ('gym.make', 'gymnasium.make')
        if kind == 'ray':
            return 'ray' in families and (route in RLLIB_NAMES or
                isinstance(node.func, ast.Attribute) and constructed(node.func.value, kind, seen))
        return 'tianshou' in families and route in TIANSHOU_NAMES
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        route = name(node.func)
        if route in GENERIC_NAMES and (isinstance(node.func, ast.Attribute) or
                isinstance(node.func, ast.Name) and node.func.id in imports):
            return True
        if isinstance(node.func, ast.Attribute):
            kind = {'learn': 'sb3', 'step': 'gym', 'build': 'ray'}.get(route)
            if kind and constructed(node.func.value, kind):
                return True
        if route in ('Collector', 'onpolicy_trainer', 'offpolicy_trainer'):
            args = [k.value for k in node.keywords if k.arg == 'policy']
            if route == 'Collector' and node.args:
                args.append(node.args[0])
            if any(isinstance(a, ast.Name) and constructed(a, 'tianshou') for a in args):
                return True
    return False


def audit(root, ignore=()):
    root = Path(root)
    if not root.is_dir():
        raise OSError('repository must be a readable directory')
    positive, negative, production, tests = [], [], [], []
    prefixes = [Path(x).as_posix().strip('/') for x in ignore]
    def ignored(rel):
        return any(rel == p or rel.startswith(p + '/') for p in prefixes)
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, onerror=walk_error):
        base = Path(directory)
        dirs[:] = sorted(d for d in dirs if not (base / d).is_symlink() and
                         not ignored((base / d).relative_to(root).as_posix()))
        for filename in sorted(files):
            path = base / filename
            rel = path.relative_to(root)
            if path.is_symlink() or ignored(rel.as_posix()):
                continue
            if rel.as_posix() == 'README.md' or path.suffix == '.md' and 'docs' in rel.parts[:-1]:
                pos, neg = claims(path.read_text(encoding='utf-8'))
                positive.extend(pos)
                negative.extend(neg)
            if path.suffix != '.py' or set(rel.parts[:-1]) & {'fixtures', 'docs', 'examples'}:
                continue
            if evidence(path.read_text(encoding='utf-8')):
                target = tests if 'tests' in rel.parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py') else production
                target.append(rel.as_posix())
    findings = []
    if positive and not production:
        findings.append({'rule': 'documented-reinforcement-learning-claim-false',
                         'severity': 'MEDIUM' if tests else 'HIGH',
                         'message': 'Documented capability has test evidence only.' if tests else 'Documented capability has no production evidence.'})
    if negative and production:
        findings.append({'rule': 'documented-no-reinforcement-learning-claim-false', 'severity': 'HIGH',
                         'message': 'Production evidence contradicts a documented denial.'})
    return {'verdict': 'FAIL' if findings else 'PASS', 'findings': findings,
            'evidence': {'production': production, 'tests': tests}}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, '%s: error: %s\n' % (self.prog, message))


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('audit')
    check.add_argument('repo', nargs='?', default='.')
    check.add_argument('--json', action='store_true')
    check.add_argument('--verbose', action='store_true')
    check.add_argument('--ignore', action='append', default=[])
    check.add_argument('--fail-on', choices=('high', 'medium'), default='high')
    args = parser.parse_args(argv)
    try:
        result = audit(args.repo, args.ignore)
    except (OSError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result['verdict'])
        for finding in result['findings']:
            print('{rule} {severity}: {message}'.format(**finding))
        if args.verbose:
            print(json.dumps(result['evidence'], sort_keys=True))
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in result['findings']) else 0
