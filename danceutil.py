#!/usr/bin/python
"""Dance utilities, e.g. generate move sequences, print move tables, etc."""

import ast
import argparse
import collections
import functools
import itertools
import logging
import math
import os
import random
import re
import sys
import time

from frozendict import frozendict
import tabulate
import yaml

logger = logging.getLogger(__name__)
loglevels = logging.getLevelNamesMapping()

## Data structures

def list_dedupe(l):
  return list(dict.fromkeys(l))

def groupby_unsorted(items, key=lambda x: [x]):
    groups = collections.defaultdict(list)
    for elem in items:
      for k in key(elem):
        groups[k].append(elem)
    return groups

def items_rec(d):
  for k, v in d.items():
    if hasattr(v, "items"):
      for k1, v1 in items_rec(v):
        yield (k,) + k1, v1
    else:
      yield (k,), v

# https://softwareengineering.stackexchange.com/a/344274
# the first example is buggy, but the second example works
def weighted_shuffle(items, weights):
  order = sorted(range(len(items)), key=lambda i: random.random() ** (1.0 / weights[i]), reverse=True)
  return [items[i] for i in order]

## Text parsing

def split(lst, f):
  if not lst: return []
  res = []
  cur = []
  cur_m = None
  for item in lst:
    m = f(item)
    if m is not None:
      if cur: res.append(cur)
      cur = []
      cur_m = m
    cur.append(cur_m(item) if cur_m else item)

  if cur: res.append(cur)
  return res

def is_new_block(line):
  m = re.search(r"^-\s+", line)
  if m:
    n = len(m.group(0))
    return lambda l: l if len(l) < n else l[n:]
  else:
    return None

def normalise_text(lines):
  return [re.match("[^#\n]*", l).group(0) for l in lines]

def normalise_block(block):
  while block and not block[-1]:
    block.pop()
  return block

def parse_blocks(lines):
  return [normalise_block(block)
    for block in split(normalise_text(lines), is_new_block)]

## Moves parsing

class Position(collections.namedtuple("Position", "stance connection")):
  @classmethod
  def parse(cls, pos):
    try:
      return cls(*pos.split("/", 1))
    except:
      raise ValueError("could not parse position: %s" % pos)

  def __str__(self):
    return "/".join(self)

  def sort_key(self, allpos):
    stances = allpos["stances"]
    return (list(stances.keys()).index(self.stance),
      stances[self.stance]["valid connections"].index(self.connection))

class Transition(collections.namedtuple("Transition", "fr to")):
  @classmethod
  def create(cls, p):
    return cls(*p)

  def __str__(self):
    return " - ".join([str(self.fr), str(self.to)])

class Move(collections.namedtuple("Move", "name count tags transitions parent")):
  def tag(self, tn):
    return self.tags.get(tn, None)

  def weight(self):
    return self.tags.get("weight", 1.0)

  def subweight(self):
    return self.tags.get("subweight", 1.0)

  def name_suf(self):
    if self.name.startswith(","):
      return self.name
    else:
      return (" " if self.parent else "") + self.name

  def __str__(self):
    return str(self.parent or "") + self.name_suf()

  def walk_ancestors(self):
    move = self
    while move:
      yield move
      move = move.parent

  def ancestors(self):
    return reversed(list(self.walk_ancestors()))

  def str_ditto(self, prev):
    paths = itertools.zip_longest(
      prev.ancestors() if prev else [], self.ancestors())
    parts = []
    for mprev, mcur in list(paths):
      if mcur is None:
        break
      suf = mcur.name_suf()
      parts.append(suf if mcur != mprev else "." * len(suf))
    return "".join(parts)

  def transitions_match(self, fr=None):
    return [tr for tr in self.transitions
      if fr is None or tr.fr == fr]

def parse_position_wild(allpos, pos):
  stance, cxn = pos.split("/", 1)
  stance = allpos["stances"].keys() if stance == "*" else [stance]
  return (Position(s, c) for s in stance
    for c in (allpos["stances"][s]["valid connections"] if cxn == "*" else [cxn]))

def parse_positions(allpos, pos):
  if not pos: return []
  pos = re.split(r"\s*,\s*", pos)
  return [p for pp in pos for p in parse_position_wild(allpos, pp)]

def parse_transition(allpos, trans, parent=None):
  if trans.strip() == "inherit":
    return parent.transitions

  fr, sep, to = re.split(r"\s+([-=])\s*", trans, maxsplit=1)
  fr = parse_positions(allpos, fr)
  to = parse_positions(allpos, to)
  if not to:
    to = fr[:]
  if sep == "-":
    return map(Transition.create, itertools.product(fr, to))
  elif sep == "=":
    return map(Transition.create, zip(fr, to))
  else:
    assert False

def parse_tag(word):
  res = re.split(r"\s*:\s*", word, maxsplit=1)
  if len(res) == 2:
    return res[0], ast.literal_eval(res[1])
  else:
    return res[0], True

def parse_move(allpos, block, parent=None):
  name, words = re.split(r"\s*:\s*", block[0], maxsplit=1)
  words = words.split()
  count = int(words.pop(0))
  tags = dict(parse_tag(w) for w in words)
  if "inherit" in tags:
    tags = dict(parent.tags | tags)
    del tags["inherit"]
  trans = [t
    for tt in block[1:] if tt
    for t in parse_transition(allpos, tt, parent)]
  trans = list_dedupe(trans)
  return Move(name, count, frozendict(tags), tuple(trans), parent)

def parse_moves_block(allpos, block, parent=None):
  if not block: return []
  variations = parse_blocks(block)
  move = parse_move(allpos, variations[0], parent)
  return [move] + [m
    for v in variations[1:]
    for m in parse_moves_block(allpos, v, move)]

def parse_moves(allpos, lines):
  return [m
          for b in parse_blocks(lines)
          for m in parse_moves_block(allpos, b)]

## Main program

class CountSpec(object):
  def is_last_move(self, move):
    return not self.use_count(move.count)

class ListCountSpec(CountSpec, collections.namedtuple("ListCountSpec", "counts")):
  @classmethod
  def parse(cls, ss):
    return cls([int(n) for n in re.split(r"\s*,\s*", ss)])

  def __str__(self):
    return ", ".join(str(c) for c in self.counts)

  def __bool__(self):
    return bool(self.counts)

  def use_count(self, count):
    if count == self.counts[0]:
      return self.__class__(self.counts[1:])
    raise ValueError("invalid count: %s" % count)

  def find_move(self, by_fr_count, fr):
    return by_fr_count.get((fr, self.counts[0]), [])

class SumCountSpec(CountSpec, collections.namedtuple("SumCountSpec", "sum")):
  @classmethod
  def parse(cls, ss):
    return cls(int(re.match(r"=(\d+)", ss).group(1)))

  def __str__(self):
    return "=%d" % self.sum

  def __bool__(self):
    return bool(self.sum)

  def use_count(self, count):
    if count <= self.sum:
      return self.__class__(self.sum - count)
    raise ValueError("invalid count: %s" % count)

  def find_move(self, by_fr_count, fr):
    return [m
      for (fr, c), moves in by_fr_count.items() if c and c <= self.sum
      for m in moves]

def parse_countspec(ss):
  if ss == "dance" or ss == "dance128":
    return [ListCountSpec.parse("6,6,6,6,8,6,6,6,6,8,8,8,8,8,6,6,6,6,8")]
  if ss == "phrases" or ss == "phrases32":
    return [ListCountSpec.parse("6,6,6,6,8"),
      ListCountSpec.parse("8,8,8,8"),
      SumCountSpec.parse("=32")]
  for cls in SumCountSpec, ListCountSpec:
    try:
      return [cls.parse(ss)]
    except:
      pass
  raise ValueError("could not parse countspec: %s" % ss)

class AppCtx(object):
  def __init__(self, positions, moves, log_level):
    self.moves = moves
    self.positions = positions
    self.log_level = log_level

    dummy_tr = Transition(None, None)
    self.by_fr_count = dict(groupby_unsorted(moves, lambda m: list_dedupe([
      (tr.fr, c)
      for tr in m.transitions + (dummy_tr,)
      for c in [m.count] + [None]
    ])))

    self.max_m = max(len(str(m)) for m in moves)
    self.max_t = max(max(len(str(tr.fr)), len(str(tr.to)))
      for m in moves
      for tr in m.transitions)

  def find_next_moves(
      self,
      fr,
      countspec,
      move_constraints=None
    ):
    res = ((move,
      move_constraints(countspec, move, move.transitions_match(fr=fr))
        if move_constraints else move.transitions_match(fr=fr))
      for move in list_dedupe(countspec.find_move(self.by_fr_count, fr)))
    return [(m, t) for (m, t) in res if t]

  def fmt_next_move(self, m, tto, pm):
    fmtstr = "%%-%ds -> %%s" % (self.max_m,)
    return fmtstr % (m.str_ditto(pm), ", ".join(str(t) for t in tto))

  def fmt_move_table(self, move_constraints):
    move_table = [
      (f, c, [(m, [t.to for t in trans])
        for m, trans in self.find_next_moves(f, ListCountSpec([c]), move_constraints)])
      for (f, c) in self.by_fr_count.keys() if f and c
    ]
    move_table.sort(key=lambda x: (x[0].sort_key(self.positions), x[1]))
    move_table = [
      [str(f), c, "\n".join(
        self.fmt_next_move(m, t, pm) for (m, t), (pm, _) in zip(l, [(None, [])] + l[:-1]))]
      for (f, c, l) in move_table if l
    ]
    return tabulate.tabulate(move_table,
      ["from", "#", "move"], tablefmt="simple_grid")

  def _weigh_moves(self, candidates):
    """Calculate weights for moves, taking into account the tree structure of parent relationships."""
    movetree = {}
    for m in candidates:
      curtree = movetree
      for p in m.ancestors():
        subtree = curtree.setdefault(p, {})
        if p == m:
          subtree[None] = True
        curtree = subtree
    assert sum(1 for _ in items_rec(movetree)) == len(candidates)

    weights = [0.0 for _ in candidates]
    def _distribute_weights(curtree, curmove, parent_contrib):
      moves = list(curtree.keys())
      totals = sum(m.weight() if m else 0 for m in moves)
      if None in moves:
        moves.remove(None)
        totals += curmove.subweight()
        curweight = curmove.subweight() * parent_contrib / totals
        weights[candidates.index(curmove)] = curweight
      for m in moves:
        contrib = m.weight() * parent_contrib / totals
        _distribute_weights(curtree[m], m, contrib)
    _distribute_weights(movetree, None, 1.0)

    assert 0.0 not in weights
    assert not weights or math.isclose(sum(weights), 1.0)
    return weights

  def _random_moveseq_rec(self, constraints, randmode, starttime, countspec, accum):
    """Recursive depth-first search through the graph of moves, taking into account user-supplied constraints on the moves we want to look at.
    """
    if not countspec:
      return accum

    elapsed = time.time() - starttime
    if sys.stdout.isatty():
      if elapsed > 4 and self.log_level <= logging.INFO:
        print("\033[Jsearching... %.3f" % elapsed,
          " ".join("%d/%d:%d/%d" % (i, ii, j, jj)
            for _, _, i, ii, j, jj in accum[1:]),
          end='\r')

    fr = accum[-1][1].to
    constraints = constraints or (lambda *args: True)
    move_constraints = functools.partial(constraints, accum)

    candidates = self.find_next_moves(fr, countspec, move_constraints)
    if randmode == 2:
      orig = candidates[:]
      eweights = self._weigh_moves([m for m, _ in candidates])
      candidates = weighted_shuffle(candidates, eweights)
      # guard against bugs in weighted_shuffle
      try:
        assert sorted(orig) == sorted(candidates)
      except TypeError:
        # ignore TypeError: '<' not supported between instances of 'frozendict' and 'frozendict'
        pass
    elif randmode == 1:
      random.shuffle(candidates)
    for i, (move, transitions) in enumerate(candidates):
      if randmode:
        # all transitions have equal weights
        random.shuffle(transitions)
      for j, trans in enumerate(transitions):
        try:
          return self._random_moveseq_rec(
            constraints, randmode, starttime,
            countspec.use_count(move.count),
            accum + [(move, trans, i+1, len(candidates), j+1, len(transitions))])
        except ValueError: # try next candidate
          pass
    logger.debug("dead end %s, backtracking...",
      [str(m[0]) for m in accum[1:]])
    raise ValueError("no moves matching the requested constraints")

  def random_moveseq(self, countspec, fr, constraints, randmode):
    moveseq = self._random_moveseq_rec(
      constraints, randmode, time.time(), countspec,
      [(None, Transition(None, fr), 0, 0, 0, 0)])[1:]
    if sys.stdout.isatty():
      print("\033[J", end="") # clear progress indicator
    return [(m, t) for m, t, _, _, _, _ in moveseq]

  def fmt_move_parts(self, m):
    fmtstr = "%%-%ds %%2d %%-%ds -> %%-%ds" % (
        self.max_t, self.max_m, self.max_t)
    return fmtstr % (m[1].fr, m[0].count, m[0], m[1].to)

  def fmt_moveseq(self, moves):
    return "\n".join(self.fmt_move_parts(m) for m in moves)

  def fmt_random_moveseqs(self, specs, num, fr, constraints, randmode):
    results = [
      [self.random_moveseq(cs, fr, constraints, randmode) for cs in specs]
      for _ in range(num)
    ]
    data = [
      [self.fmt_moveseq(m) for m in moveseq]
      for moveseq in results
    ]
    return tabulate.tabulate(data, specs, "simple_grid")

def main(*argv):
  parser = argparse.ArgumentParser(prog="DANCE",
    description="Dance moves helper")
  parser.add_argument("--log-level", "-l",
    help="Log level, one of %s. Default: INFO." % list(loglevels.keys()),
    type=loglevels.get, default=logging.INFO)
  parser.add_argument("--positions", "-p",
    help="Positions file, YAML", default="positions.yml")
  parser.add_argument("--moves", "-m",
    help="Moves file, custom format", default="swing-moves.txt")
  subparsers = parser.add_subparsers(help="Subcommand to execute")

  parser_mt = subparsers.add_parser("move-table", aliases=["mt", "tab"],
    help="Print table of moves, sorted by from-position and count")
  parser_mt.add_argument("--hide-basic",
    help="Hide basic moves, default: %(default)s.",
    default=True, action=argparse.BooleanOptionalAction)
  parser_mt.add_argument("--hide-filler",
    help="Hide filler moves, default: %(default)s.",
    default=True, action=argparse.BooleanOptionalAction)
  parser_mt.set_defaults(subcmd="move-table")

  parser_gen = subparsers.add_parser("gen-moveseq", aliases=["gen"],
    help="Generate random move sequences")
  parser_gen.set_defaults(subcmd="gen-moveseq")
  parser_gen.add_argument("--num", "-n",
    help="Number of move sequences to generate, default: %(default)s.",
    default=1, type=int, nargs="?")
  parser_gen.add_argument("--fr",
    help="Position to start the move sequence at. If omitted we use any position.", type=Position.parse)
  parser_gen.add_argument("--to",
    help="Position to end the move sequence at. If omitted we use any position.", type=Position.parse)
  parser_gen.add_argument("--randmode", "-r",
    help="Select the random mode for searching through the graph of moves. 0: not random, in the order listed in the moves file; 1: in a uniformly random order, 2: based on their family and weight. Default: %(default)s.",
    default=2, type=int, choices=[0, 1, 2])
  parser_gen.add_argument("--randseed",
    help="Random number generator seed, useful for testing.")
  parser_gen.add_argument("--deny-basic",
    help="Don't use basic moves, default: %(default)s.",
    default=True, action=argparse.BooleanOptionalAction)
  parser_gen.add_argument("--deny-initial-filler",
    help="Don't use filler moves in initial position, default: %(default)s.",
    default=True, action=argparse.BooleanOptionalAction)
  parser_gen.add_argument("--deny-adjacent-fillers",
    help="Don't use filler moves in adjacent positions, default: %(default)s.",
    default=True, action=argparse.BooleanOptionalAction)
  parser_gen.add_argument("--deny-endings-elsewhere",
    help="Don't use ending moves in non-ending position, default: %(default)s.",
    default=True, action=argparse.BooleanOptionalAction)
  parser_gen.add_argument("--deny-same-family",
    help="Don't use moves from the same family within this number of counts, default: %(default)s. Set to 0 to allow all.",
    default=88, type=int)
  default_counts = ["dance"]
  parser_gen.add_argument("countspec",
    help="What move counts to generate. Options: a comma-separated list of exact move counts, or =SUM for any number of moves with total count SUM, or the following special aliases: dance (6,6,6,6,8,6,6,6,6,8,8,8,8,8,6,6,6,6,8), phrases (6,6,6,6,8; 8,8,8,8; =32). Default: %s." % default_counts,
    nargs="*", type=parse_countspec,
    default=[parse_countspec(cs) for cs in default_counts])

  args = parser.parse_args(argv)
  logging.basicConfig(level=args.log_level)

  with open(args.positions) as fp:
    positions = yaml.full_load(fp)
  with open(args.moves) as fp:
    moves = parse_moves(positions, fp.readlines())

  app = AppCtx(positions, moves, args.log_level)

  if "subcmd" not in args:
    parser.print_help(sys.stderr)

  elif args.subcmd == "move-table":
    def move_constraints(_, move, transitions):
      if args.hide_filler:
        if move.tag("filler"):
          return []
      if args.hide_basic:
        if move.tag("basic"):
          return []
      return transitions
    print(app.fmt_move_table(move_constraints))

  elif args.subcmd == "gen-moveseq":
    if args.randseed:
      random.seed(args.randseed)

    def constraints(accum, countspec, move, transitions):
      if args.deny_same_family:
        spent = 0
        for m in reversed(accum):
          if spent > args.deny_same_family or not m[0]:
            break
          if next(m[0].ancestors()) == next(move.ancestors()):
            return []
          spent += m[0].count
      if args.deny_basic:
        if move.tag("basic"):
          return []
      if args.deny_initial_filler:
        if move.tag("filler") and accum[-1][0] is None:
          return []
      if args.deny_adjacent_fillers:
        if move.tag("filler") and accum[-1][0] and accum[-1][0].tag("filler"):
          return []
      if args.deny_endings_elsewhere:
        if move.tag("ending"):
          if not countspec.is_last_move(move):
            return []
      if args.to:
        if countspec.is_last_move(move):
          return [t for t in transitions if t.to == args.to]
      return transitions
    countspec = [cs for css in args.countspec for cs in css]
    print(app.fmt_random_moveseqs(
      countspec, args.num, args.fr, constraints, args.randmode))

  else:
    assert False

if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:]))
