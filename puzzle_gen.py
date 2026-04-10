"""
Generate 2-person Knights and Knaves puzzles with brute-force solving.

Statements are claims in a small AST; speaker i utters statements[i].
Knight=True means the person's utterance is true; Knave=False means it is false.
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# --- Truth assignment ---------------------------------------------------------

Statement = tuple  # nested tuples, see eval_statement


def eval_statement(s: Statement, assignment: Tuple[bool, ...]) -> bool:
    tag = s[0]
    if tag == "knight":
        return assignment[s[1]]
    if tag == "not":
        return not eval_statement(s[1], assignment)
    if tag == "and":
        return eval_statement(s[1], assignment) and eval_statement(s[2], assignment)
    if tag == "or":
        return eval_statement(s[1], assignment) or eval_statement(s[2], assignment)
    if tag == "same":
        return assignment[s[1]] == assignment[s[2]]
    if tag == "diff":
        return assignment[s[1]] != assignment[s[2]]
    raise ValueError(f"unknown claim {s!r}")


def solve(statements: List[Statement]) -> List[bool]:
    """
    Find assignment (True=Knight) such that each speaker's claim matches their type.
    Same semantics as: all(eval_statement(s, a) == a[i] for i, s in enumerate(statements))
    """
    n = len(statements)
    for assignment in itertools.product([True, False], repeat=n):
        a = tuple(assignment)
        if all(eval_statement(s, a) == a[i] for i, s in enumerate(statements)):
            return list(a)
    return []


def all_solutions(statements: List[Statement]) -> List[Tuple[bool, ...]]:
    n = len(statements)
    out: List[Tuple[bool, ...]] = []
    for assignment in itertools.product([True, False], repeat=n):
        a = tuple(assignment)
        if all(eval_statement(s, a) == a[i] for i, s in enumerate(statements)):
            out.append(a)
    return out


def assignment_to_gold(a: Sequence[bool]) -> str:
    # Generalize to n people: A=Knight, B=Knave, C=Knight, ...
    return ", ".join(f"{chr(ord('A')+i)}={'Knight' if v else 'Knave'}" for i, v in enumerate(a))


def _name(idx: int) -> str:
    return chr(ord("A") + idx)


def _pronoun_subj(speaker: int, p: int) -> str:
    """Subject phrase for person p when embedded in speaker's quote."""
    if p == speaker:
        return "I"
    return _name(p)


def render_claim_clause(s: Statement, speaker: int, lex: int) -> str:
    """
    lex 0: knight / knave
    lex 1: truth-teller / liar (same truth conditions)
    """
    if lex == 0:
        wk, wv = "knight", "knave"
    else:
        wk, wv = "truth-teller", "liar"

    tag = s[0]
    if tag == "knight":
        p = s[1]
        subj = _pronoun_subj(speaker, p)
        if subj == "I":
            return f"I am a {wk}"
        return f"{subj} is a {wk}"
    if tag == "not":
        inner = s[1]
        if inner[0] == "knight":
            p = inner[1]
            subj = _pronoun_subj(speaker, p)
            if subj == "I":
                return f"I am a {wv}"
            return f"{subj} is a {wv}"
        inner_text = render_claim_clause(inner, speaker, lex)
        return f"it is not the case that {inner_text}"
    if tag == "and":
        return f"{render_claim_clause(s[1], speaker, lex)} and {render_claim_clause(s[2], speaker, lex)}"
    if tag == "or":
        return f"either {render_claim_clause(s[1], speaker, lex)} or {render_claim_clause(s[2], speaker, lex)}"
    if tag == "same":
        if speaker == s[1]:
            other = _name(s[2])
            return f"{other} and I are the same kind"
        if speaker == s[2]:
            other = _name(s[1])
            return f"{other} and I are the same kind"
        return f"{_name(s[1])} and {_name(s[2])} are the same kind"
    if tag == "diff":
        if speaker == s[1]:
            other = _name(s[2])
            return f"{other} and I are different kinds"
        if speaker == s[2]:
            other = _name(s[1])
            return f"{other} and I are different kinds"
        return f"{_name(s[1])} and {_name(s[2])} are different kinds"
    raise ValueError(s)


def render_puzzle(statements: List[Statement], lex: int) -> str:
    """n speakers: A, B, C, ...; A speaks first."""
    parts = []
    for i, st in enumerate(statements):
        who = _name(i)
        inner = render_claim_clause(st, i, lex)
        parts.append(f"{who} says: '{inner}.'")
    # Generalize question for n people
    n = len(statements)
    if n == 2:
        q = "What are A and B?"
    else:
        q = "What are " + ", ".join(_name(i) for i in range(n-1)) + f" and {_name(n-1)}?"
    return " ".join(parts) + f" {q}"


# --- Leaf perturbation (single substring replace, preserves semantics) ---------

_LEAF_SYNONYMS: Tuple[Tuple[str, str], ...] = (
    (" says: ", " asserts: "),
    (" says: ", " states: "),
    (" knave", " liar"),
    (" knave.", " liar."),
    (" knight", " truth-teller"),
    (" knight.", " truth-teller."),
    (" and ", " as well as "),
)


def apply_leaf_perturbation(text: str, rng: random.Random) -> str:
    """One minimal surface edit (not a full re-lettering of the puzzle)."""
    opts = [(o, n) for o, n in _LEAF_SYNONYMS if o in text]
    if not opts:
        return text
    old, new = rng.choice(opts)
    return text.replace(old, new, 1)


# --- Generator ---------------------------------------------------------------

def _claim_pool(n: int) -> List[Statement]:
    """Library of claims for n people (indices 0..n-1)."""
    pool = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pool.append(("knight", j))
            pool.append(("not", ("knight", j)))
            pool.append(("same", i, j))
            pool.append(("diff", i, j))
            pool.append(("and", ("not", ("knight", i)), ("not", ("knight", j))))
            pool.append(("or", ("knight", i), ("knight", j)))
            pool.append(("not", ("and", ("knight", i), ("knight", j))))
            pool.append(("and", ("knight", i), ("not", ("knight", j))))
    return pool


def try_generate_pair(rng: random.Random, n: int = 2, max_attempts: int = 80) -> Optional[Tuple[List[Statement], Tuple[bool, ...]]]:
    pool = _claim_pool(n)
    for _ in range(max_attempts):
        stmts = [rng.choice(pool) for _ in range(n)]
        sols = all_solutions(stmts)
        if len(sols) != 1:
            continue
        return stmts, sols[0]
    return None

def perturb_leaf(
    statements: List[Statement], 
    rng: random.Random, 
    max_attempts: int = 200
) -> Optional[Tuple[List[Statement], Tuple[bool, ...]]]:
    """
    Replace one leaf node in the AST to get a puzzle with a 
    DIFFERENT unique solution. Generalized for n people.
    """
    n = len(statements)
    leaf_pool = []
    for i in range(n):
        leaf_pool.append(("knight", i))
        leaf_pool.append(("not", ("knight", i)))
    for i in range(n):
        for j in range(n):
            if i != j:
                leaf_pool.append(("same", i, j))
                leaf_pool.append(("diff", i, j))
                leaf_pool.append(("and", ("not", ("knight", i)), ("not", ("knight", j))))
                leaf_pool.append(("or", ("knight", i), ("knight", j)))
                leaf_pool.append(("not", ("and", ("knight", i), ("knight", j))))
                leaf_pool.append(("and", ("knight", i), ("not", ("knight", j))))

    original_solutions = all_solutions(statements)
    if len(original_solutions) != 1:
        return None
    original_answer = original_solutions[0]

    for _ in range(max_attempts):
        # Pick which speaker's statement to modify
        speaker_idx = rng.randint(0, n - 1)
        new_leaf = rng.choice(leaf_pool)

        new_statements = list(statements)
        new_statements[speaker_idx] = new_leaf

        new_solutions = all_solutions(new_statements)

        # Must have unique solution AND different answer
        if len(new_solutions) == 1 and new_solutions[0] != original_answer:
            return new_statements, new_solutions[0]

    return None


@dataclass
class GeneratedPuzzle:
    statements: List[Statement]
    assignment: Tuple[bool, bool]
    perturbed_statements: List[Statement]
    perturbed_assignment: Tuple[bool, bool]
    original_text: str
    perturbed_text: str
    gold: str
    perturbed_gold: str

def generate_one(rng: random.Random, n: int = 2, max_outer: int = 50) -> Optional[GeneratedPuzzle]:
    for _ in range(max_outer):
        got = try_generate_pair(rng, n=n)
        if got is None:
            continue

        stmts, assign = got

        perturbed = perturb_leaf(stmts, rng)
        if perturbed is None:
            continue  # retry, don't fall back to hardcoded puzzle

        pert_stmts, pert_assign = perturbed

        orig_text = render_puzzle(stmts, lex=0)
        pert_text = render_puzzle(pert_stmts, lex=0)

        return GeneratedPuzzle(
            statements=stmts,
            assignment=assign,
            perturbed_statements=pert_stmts,
            perturbed_assignment=pert_assign,
            original_text=orig_text,
            perturbed_text=pert_text,
            gold=assignment_to_gold(assign),
            perturbed_gold=assignment_to_gold(pert_assign),
        )
    return None


def generate_n_pairs(num_pairs: int, rng: random.Random, max_people: int = 2) -> List[GeneratedPuzzle]:
    """Generate num_pairs puzzles, each with max_people participants."""
    out: List[GeneratedPuzzle] = []
    for _ in range(num_pairs):
        p = generate_one(rng, n=max_people)
        out.append(p)
    return out


if __name__ == "__main__":
    rng = random.Random(42)
    for i in range(5):
        p = generate_one(rng)
        print(p.original_text)
        print(p.perturbed_text)
        print(p.gold)
        print()
