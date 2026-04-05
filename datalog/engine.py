"""
engine.py — Semi-naive Datalog evaluator with hash-join indexing.

No dependencies. Supports:
  - Facts: relation(a, b, c)
  - Rules: head(X, Y) :- body1(X, Z), body2(Z, Y)
  - Negation: head(X) :- body(X), not neg(X)  [stratified]
  - Arithmetic comparisons: X < Y, X != Y, X * 2 >= Y

Designed to be the monotonic layer that feeds Clingo's choice/optimization.
"""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Rule:
    head_relation: str
    head_vars: tuple  # variable names or constants
    body: list  # list of (relation, vars, negated)
    constraints: list  # list of (op, left, right) — arithmetic


class DatalogEngine:
    def __init__(self):
        self.facts: dict[str, set[tuple]] = defaultdict(set)
        self.rules: list[Rule] = []
        # Index: (relation, position) -> {value: set of tuples}
        self._index: dict[tuple[str, int], dict] = {}

    def add_fact(self, relation: str, *args):
        self.facts[relation].add(args)

    def _get_index(self, relation: str, pos: int) -> dict:
        key = (relation, pos)
        if key not in self._index:
            idx = defaultdict(set)
            for tup in self.facts.get(relation, set()):
                if pos < len(tup):
                    idx[tup[pos]].add(tup)
            self._index[key] = idx
        return self._index[key]

    def _invalidate_index(self, relation: str):
        keys = [k for k in self._index if k[0] == relation]
        for k in keys:
            del self._index[k]

    def query(self, relation: str) -> set[tuple]:
        return self.facts.get(relation, set())

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def solve(self, verbose=False) -> dict[str, set[tuple]]:
        """Semi-naive evaluation with stratification."""
        # Clear index cache — external code may have added facts directly
        self._index.clear()
        strata = self._stratify()
        for stratum in strata:
            self._evaluate_stratum(stratum, verbose)
        return dict(self.facts)

    def _stratify(self) -> list[list[Rule]]:
        """Topological stratification respecting both positive and negation deps.

        Rules are grouped so that:
        1. If rule A's head is used positively in rule B's body, A's stratum <= B's
        2. If rule A's head is used with negation in rule B's body, A's stratum < B's
        """
        # Build dependency graph: head_relation -> {(body_relation, negated)}
        # First, collect which relations are derived (appear as rule heads)
        derived = {rule.head_relation for rule in self.rules}

        # Assign strata to relations
        rel_stratum: dict[str, int] = {}
        for rel in derived:
            rel_stratum[rel] = 0

        # Iterate until stable
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                head = rule.head_relation
                for body_rel, _, negated in rule.body:
                    if body_rel not in derived:
                        continue
                    if negated:
                        # Negated: body must be strictly lower stratum
                        needed = rel_stratum[body_rel] + 1
                    else:
                        # Positive: body must be same or lower stratum
                        needed = rel_stratum[body_rel]
                    if needed > rel_stratum[head]:
                        rel_stratum[head] = needed
                        changed = True

        # Group rules by their head's stratum
        max_stratum = max(rel_stratum.values()) if rel_stratum else 0
        strata: list[list[Rule]] = [[] for _ in range(max_stratum + 1)]
        for rule in self.rules:
            s = rel_stratum[rule.head_relation]
            strata[s].append(rule)

        return [s for s in strata if s]

    def _evaluate_stratum(self, rules: list[Rule], verbose: bool):
        if not rules:
            return

        iteration = 0
        while True:
            iteration += 1
            new_facts = defaultdict(set)

            for rule in rules:
                derived = self._evaluate_rule(rule)
                for args in derived:
                    if args not in self.facts[rule.head_relation]:
                        new_facts[rule.head_relation].add(args)

            if not new_facts:
                break

            for rel, tuples in new_facts.items():
                self.facts[rel].update(tuples)
                self._invalidate_index(rel)

            if verbose:
                total = sum(len(t) for t in new_facts.values())
                print(f"  Iteration {iteration}: {total} new facts across {list(new_facts.keys())}")

    def _evaluate_rule(self, rule: Rule) -> set[tuple]:
        """Evaluate a single rule using index-nested-loop join."""
        pos_body = [(r, v) for r, v, neg in rule.body if not neg]
        neg_body = [(r, v) for r, v, neg in rule.body if neg]

        if not pos_body:
            return set()

        # Start with first positive atom
        rel0, vars0 = pos_body[0]
        bindings = []
        for fact_args in self.facts.get(rel0, set()):
            b = self._unify({}, vars0, fact_args)
            if b is not None:
                bindings.append(b)

        # Join with remaining positive atoms
        for rel, vars_ in pos_body[1:]:
            bindings = self._join_indexed(bindings, rel, vars_)
            if not bindings:
                return set()

        # Apply arithmetic constraints
        if rule.constraints:
            bindings = [b for b in bindings if self._check_constraints(b, rule.constraints)]

        # Filter by negated atoms
        for rel, vars_ in neg_body:
            bindings = self._filter_negated(bindings, rel, vars_)
            if not bindings:
                return set()

        # Project
        results = set()
        for b in bindings:
            args = tuple(
                b[v] if isinstance(v, str) and v[0].isupper() else v
                for v in rule.head_vars
            )
            results.add(args)
        return results

    def _unify(self, binding: dict, vars_: tuple, values: tuple) -> dict | None:
        if len(vars_) != len(values):
            return None
        b = binding.copy()
        for var, val in zip(vars_, values):
            if isinstance(var, str) and var[0].isupper():
                if var in b:
                    if b[var] != val:
                        return None
                else:
                    b[var] = val
            else:
                if var != val:
                    return None
        return b

    def _join_indexed(self, bindings: list[dict], relation: str, vars_: tuple) -> list[dict]:
        """Join using index on the first bound variable position."""
        if not bindings:
            return []

        # Find which positions in vars_ are already bound
        sample = bindings[0]
        bound_positions = []
        for i, v in enumerate(vars_):
            if isinstance(v, str) and v[0].isupper() and v in sample:
                bound_positions.append(i)
            elif not (isinstance(v, str) and v[0].isupper()):
                # constant — use for filtering
                bound_positions.append(i)

        if bound_positions:
            # Use index on first bound position for lookup
            idx_pos = bound_positions[0]
            idx = self._get_index(relation, idx_pos)

            result = []
            for binding in bindings:
                var = vars_[idx_pos]
                if isinstance(var, str) and var[0].isupper():
                    lookup_val = binding[var]
                else:
                    lookup_val = var

                for fact_args in idx.get(lookup_val, set()):
                    b = self._unify(binding, vars_, fact_args)
                    if b is not None:
                        result.append(b)
            return result
        else:
            # No bound vars — full scan
            result = []
            facts = self.facts.get(relation, set())
            for binding in bindings:
                for fact_args in facts:
                    b = self._unify(binding, vars_, fact_args)
                    if b is not None:
                        result.append(b)
            return result

    def _filter_negated(self, bindings: list[dict], relation: str, vars_: tuple) -> list[dict]:
        result = []
        facts = self.facts.get(relation, set())

        # Build a set of ground tuples for fast lookup when all vars are bound
        for binding in bindings:
            # Try to ground the pattern
            ground = []
            all_ground = True
            for v in vars_:
                if isinstance(v, str) and v[0].isupper():
                    if v in binding:
                        ground.append(binding[v])
                    else:
                        all_ground = False
                        break
                else:
                    ground.append(v)

            if all_ground:
                if tuple(ground) not in facts:
                    result.append(binding)
            else:
                # Partial ground — need to scan
                found = False
                for fact_args in facts:
                    if self._unify(binding, vars_, fact_args) is not None:
                        found = True
                        break
                if not found:
                    result.append(binding)

        return result

    def _check_constraints(self, binding: dict, constraints: list) -> bool:
        for op, left, right in constraints:
            lval = self._eval_expr(binding, left)
            rval = self._eval_expr(binding, right)
            if lval is None or rval is None:
                return False
            if op == "!=" and not (lval != rval):
                return False
            elif op == "<" and not (lval < rval):
                return False
            elif op == "<=" and not (lval <= rval):
                return False
            elif op == ">" and not (lval > rval):
                return False
            elif op == ">=" and not (lval >= rval):
                return False
            elif op == "==" and not (lval == rval):
                return False
        return True

    def _eval_expr(self, binding: dict, expr):
        if isinstance(expr, tuple):
            op, a, b = expr
            av = self._eval_expr(binding, a)
            bv = self._eval_expr(binding, b)
            if av is None or bv is None:
                return None
            if op == "*":
                return av * bv
            elif op == "+":
                return av + bv
            elif op == "-":
                return av - bv
        elif isinstance(expr, str) and expr[0].isupper():
            return binding.get(expr)
        return expr
