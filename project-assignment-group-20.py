"""
Group 20 – MAS project
Voter Behaviour Change Based on the Information Source:
Human–Human versus Human–Computer, and the Role of Misinformation

Annotation convention:
    # [Deffuant et al., 2000]  -> rule taken from the original bounded-confidence model
    # [Mesa]                   -> standard Mesa 3.x usage pattern (Mesa documentation)
    # [NetworkX]               -> NetworkX implementation of the Watts–Strogatz (1998) small-world graph
    # [OWN]                    -> our own contribution / design choice
    # [Dietvorst et al., 2015] -> inspired the trust rule (the exact numbers are ours)
    # [Maineri et al., 2021]   -> motivated the two peak positions of the bimodal start
    Step 1–11                -> the order in which we built and tested the model

How to run:
    python project-assignment-group-20.py             full experiment (25+ minutes) -> results/
    python project-assignment-group-20.py quick       small fast check -> results/quick/
    python project-assignment-group-20.py bimodal     extension: bimodal start, subset -> results/bimodal/
    python project-assignment-group-20.py test        all verification tests

References:
    Deffuant, G., Neau, D., Amblard, F., & Weisbuch, G. (2000). Mixing beliefs among
        interacting agents. Advances in Complex Systems, 3, 87–98.
    Dietvorst, B. J., Simmons, J. P., & Massey, C. (2015). Algorithm aversion: People
        erroneously avoid algorithms after seeing them err. Journal of Experimental
        Psychology: General, 144(1), 114–126.
    Maineri, A., Aizpurua, E., Fitzgerald, R., Lomazzi, V., & Luijkx, R. (2021). In harmony:
        Exploring the feasibility of ex-post harmonisation of European Social Survey and
        European Values Study items. ESRA Conference.
    Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks.
        Nature, 393, 440–442.
    Mesa documentation: https://mesa.readthedocs.io
    NetworkX documentation: https://networkx.org
"""

import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from mesa import Agent, Model


# =============================================================================
# STEP 1 – ALL PARAMETERS IN ONE PLACE                                   [OWN]
# =============================================================================
# Every number the model uses lives here. Nothing is hard-coded further down.

@dataclass(frozen=True)
class Params:
    # --- general ---
    seed: int = 0              # random seed -> same seed gives exactly the same run
    N: int = 500               # number of voters
    T: int = 1000              # number of ticks (one tick = every voter acts once);
                               # 200 in the proposal, raised after the pilot (see report)

    # --- Deffuant core ---
    e: float = 0.2             # confidence threshold: only listen if |m - x_i| <= e
    mu: float = 0.5            # convergence parameter (plain Deffuant baseline only)

    # --- network (Step 3) ---
    network_type: str = "small_world"  # "small_world" (main) or "complete" (verification)
    mean_degree: int = 6       # Watts–Strogatz: each voter linked to k nearest
    rewire_prob: float = 0.1   # Watts–Strogatz rewiring probability

    # --- initial opinions (Step 11, extension) ---
    initial_dist: str = "uniform"      # "uniform" (main experiment) or "bimodal" (extension)
    # peak positions: side peaks at 3 and 7 (of 0-10) in Norway [Maineri et al., 2021];
    # two roughly equal groups without a centre peak, and SD 0.1, are our own choices [OWN]
    bimodal_peaks: tuple = (0.30, 0.70)  # centre-left and centre-right peak
    bimodal_sd: float = 0.1            # spread around each peak

    # --- voters (Step 3; computer trust from Step 5) ---
    s_min: float = 0.2         # susceptibility s_i ~ U(s_min, s_max)
    s_max: float = 1.0
    tau_h: float = 0.5         # fixed trust in humans

    # --- computer source (Steps 4–6) ---
    p: float = 0.0             # probability of consulting the computer
    k: int = 10                # computer averages k random voters
    tau_c0: float = 0.5        # initial computer trust
    trust_down: float = 0.05   # trust loss after rejecting computer message
    trust_up: float = 0.02     # trust gain after accepting computer message
    b: float = 0.0             # computer bias (misinformation)

    # --- noise (Step 7) ---
    sigma: float = 0.05        # SD of Gaussian noise on human messages

    # --- parties (Step 8) ---
    # party 0 = left, 1 = centre-left, 2 = centre-right, 3 = right
    party_positions: tuple = (0.125, 0.375, 0.625, 0.875)

    # --- measures ---
    cluster_gap: float = 0.05  # new cluster starts at a gap larger than this
    min_cluster_size: int = 25 # clusters smaller than this are ignored


PARAMS = Params()


@dataclass(frozen=True)
class Experiment:
    """Settings of the full experiment (Steps 9–10).                    [OWN]

    p = 0 means the computer is never used, so tau_c0 and b do not matter:
    then only e is varied -> 3 + 2 x 3 x 3 x 2 = 39 conditions.
    """
    p_values: tuple = (0.0, 0.5, 1.0)
    tau_c0_values: tuple = (0.25, 0.5, 0.75)
    e_values: tuple = (0.1, 0.2, 0.3)
    b_values: tuple = (0.0, 0.15)
    seeds: tuple = tuple(range(30))   # same seeds 0–29 in every condition
    record_every: int = 10            # per-tick summaries saved every 10 ticks (+ last tick)
    last_ticks: int = 10              # party shares are also averaged over the last 10 ticks
    # runs saved in full (every voter, every recorded tick), for plots:
    # (p, tau_c0, e, b), all with seed `example_seed`
    example_conditions: tuple = ((0.5, 0.5, 0.2, 0.0),
                                 (0.5, 0.5, 0.2, 0.15),
                                 (0.5, 0.5, 0.3, 0.15))
    example_seed: int = 0
    out_dir: str = "results"


EXPERIMENT = Experiment()


# =============================================================================
# MEASURES
# =============================================================================

def count_clusters(opinions, gap, min_size):
    """Number of opinion clusters.                                       [OWN]

    Sort opinions; a new cluster starts wherever two neighbouring opinions
    differ by more than `gap`. Clusters with fewer than `min_size` voters
    are ignored (they are stragglers, not real opinion groups).
    """
    x = np.sort(np.asarray(opinions))
    # positions where the gap between consecutive sorted opinions is too big
    breaks = np.where(np.diff(x) > gap)[0]
    # cluster sizes = lengths of the pieces between the breaks
    edges = np.concatenate(([0], breaks + 1, [len(x)]))
    sizes = np.diff(edges)
    return int(np.sum(sizes >= min_size))


# =============================================================================
# STEP 2 – PLAIN DEFFUANT BASELINE (random mixing, two-way update)
# =============================================================================

class DeffuantVoter(Agent):
    """A voter in the original Deffuant model: just an opinion in [0, 1]."""

    def __init__(self, model):
        super().__init__(model)                     # [Mesa] gives unique_id
        # initial opinion x_i ~ U(0, 1)              # [Deffuant et al., 2000]
        self.opinion = self.model.random.random()

    def step(self):
        """Pick a random other voter and maybe move towards each other."""
        p = self.model.params

        # random mixing: any other voter can be the partner  [Deffuant et al., 2000]
        partner = self.model.random.choice(self.model.voter_list)
        while partner is self:
            partner = self.model.random.choice(self.model.voter_list)

        xi, xj = self.opinion, partner.opinion
        # bounded confidence: only interact if opinions are close enough
        if abs(xi - xj) <= p.e:                     # [Deffuant et al., 2000]
            # two-way update, both use the OLD values  [Deffuant et al., 2000]
            self.opinion = xi + p.mu * (xj - xi)
            partner.opinion = xj + p.mu * (xi - xj)


class DeffuantModel(Model):
    """Original bounded-confidence model (Deffuant et al., 2000).

    Kept as a verification baseline: our extended model should reproduce
    its behaviour when the extensions are switched off.
    """

    def __init__(self, params: Params):
        super().__init__(rng=params.seed)           # [Mesa] seeds self.random and self.rng
        self.params = params
        for _ in range(params.N):
            DeffuantVoter(self)                      # [Mesa] auto-registers in self.agents
        self.voter_list = list(self.agents)          # fixed list for fast random choice

    def step(self):
        # every voter acts once, in a new random order each tick   [Mesa]
        self.agents.shuffle_do("step")

    def opinions(self):
        return np.array([a.opinion for a in self.voter_list])


# =============================================================================
# STEP 3 – SMALL-WORLD NETWORK + RECEIVER-ONLY UPDATE                    [OWN]
# =============================================================================
# Differences from the Deffuant baseline above:
#   (a) voters only talk to their network neighbours, not to anyone;
#   (b) only the RECEIVER changes opinion (the sender stays the same);
#   (c) the step size is s_i * tau_h instead of a fixed mu.
# The bounded-confidence rule |m - x_i| <= e is still Deffuant et al. (2000).

def build_network(params: Params):
    """Create the social network. Nodes are numbered 0 .. N-1."""
    if params.network_type == "small_world":
        # [NetworkX] Watts & Strogatz (1998) small-world graph, seeded
        return nx.watts_strogatz_graph(params.N, params.mean_degree,
                                       params.rewire_prob, seed=params.seed)
    if params.network_type == "complete":
        # everyone linked to everyone = random mixing (for verification)
        return nx.complete_graph(params.N)
    raise ValueError(f"unknown network_type: {params.network_type}")


def clip01(value):
    """Keep a number inside [0, 1]."""
    return min(max(value, 0.0), 1.0)


def party_of(opinion, positions):
    """Index of the party closest to this opinion.                 [OWN] Step 8

    E.g. with positions (0.125, 0.375, 0.625, 0.875): 0.30 -> party 1.
    On an exact tie (e.g. 0.25) the lower-numbered party wins.
    """
    distances = [abs(pos - opinion) for pos in positions]
    return distances.index(min(distances))


def initial_opinion(model):
    """Starting opinion of one voter.

    "uniform": x_i ~ U(0, 1)                             [Deffuant et al., 2000]
    "bimodal": about half the voters start near each peak      [OWN] Step 11 extension
               (centre-left / centre-right).
               Each voter picks a peak at random, then x_i ~ N(peak, bimodal_sd).
               Values outside [0, 1] are redrawn (not clipped), so no voters
               pile up exactly at 0 or 1, even if a wider SD is chosen.
    """
    params = model.params
    if params.initial_dist == "uniform":
        return model.random.random()
    if params.initial_dist == "bimodal":
        peak = model.random.choice(params.bimodal_peaks)
        while True:
            x = model.random.gauss(peak, params.bimodal_sd)
            if 0.0 <= x <= 1.0:
                return x
    raise ValueError(f"unknown initial_dist: {params.initial_dist}")


class Voter(Agent):
    """A voter in our extended model. [OWN, extends Deffuant et al., 2000]"""

    def __init__(self, model, node):
        super().__init__(model)                     # [Mesa]
        params = model.params
        self.node = node                            # position in the network
        self.opinion = initial_opinion(model)
        # fixed susceptibility s_i ~ U(s_min, s_max) # [OWN]
        self.susceptibility = model.random.uniform(params.s_min, params.s_max)
        # personal computer trust, starts at tau_c0  # [OWN] Step 5
        self.computer_trust = params.tau_c0
        # party: closest party; remember the start to measure switching  [OWN] Step 8
        self.initial_party = party_of(self.opinion, params.party_positions)
        self.party = self.initial_party
        self.neighbours = []                        # filled in by the model

    def step(self):
        """One action of this voter in one tick."""
        params = self.model.params

        # 1) choose the information source                             [OWN] Step 4
        use_computer = params.p > 0 and self.model.random.random() < params.p

        # 2) get the message and the trust in its source               [OWN]
        if use_computer:
            message = self.model.computer.consult()
            trust = self.computer_trust              # personal, changes over time
        else:
            partner = self.model.random.choice(self.neighbours)
            message = self.human_message(partner)    # opinion + noise (Step 7)
            trust = params.tau_h                     # fixed

        # 3) bounded confidence: only listen if close enough  [Deffuant et al., 2000]
        accepted = abs(message - self.opinion) <= params.e
        if accepted:
            # receiver-only update, weighted by susceptibility and trust  [OWN]
            self.opinion += self.susceptibility * trust * (message - self.opinion)

        # 4) after using the computer, update trust in it              [OWN] Step 5
        if use_computer:
            self.update_computer_trust(accepted)

        # 5) update party preference: closest party to the new opinion  [OWN] Step 8
        self.party = party_of(self.opinion, params.party_positions)

    def human_message(self, partner):
        """What a neighbour says: their opinion + Gaussian noise.   [OWN] Step 7

        The noise stands for imperfect communication (people do not state
        their opinion exactly). Clipped so the message stays in [0, 1].
        """
        sigma = self.model.params.sigma
        if sigma > 0:
            return clip01(partner.opinion + self.model.random.gauss(0, sigma))
        return partner.opinion

    def update_computer_trust(self, accepted):
        """Trust goes up a little after accepting the computer's message and
        down more after rejecting it: people lose trust in algorithms faster
        than they gain it.               [OWN, based on Dietvorst et al., 2015]
        """
        params = self.model.params
        change = params.trust_up if accepted else -params.trust_down
        self.computer_trust = clip01(self.computer_trust + change)


class Computer(Agent):
    """The computer source (e.g. a search engine or chatbot).       [OWN] Step 4

    It has no opinion of its own. When a voter consults it, it returns the mean
    opinion of k randomly chosen voters (from the whole population, not just
    the network), plus a bias b (misinformation, Step 6), kept inside [0, 1].
    """

    def consult(self):
        params = self.model.params
        sample = self.model.random.sample(self.model.voter_list, params.k)
        summary = np.mean([v.opinion for v in sample])
        return clip01(summary + params.b)            # bias b: Step 6


class VoterModel(Model):
    """Our extended model: network, computer source, trust, bias, noise, parties. [OWN]"""

    def __init__(self, params: Params):
        super().__init__(rng=params.seed)           # [Mesa]
        self.params = params
        self.network = build_network(params)

        # one voter per network node
        self.voter_list = [Voter(self, node) for node in self.network.nodes]
        # give every voter a list of its neighbouring Voter objects
        for voter in self.voter_list:
            voter.neighbours = [self.voter_list[n]
                                for n in self.network.neighbors(voter.node)]

        # one computer, shared by everyone                             [OWN]
        self.computer = Computer(self)

    def step(self):
        # only voters act; the computer just answers when consulted   [Mesa]
        self.agents_by_type[Voter].shuffle_do("step")

    def opinions(self):
        return np.array([a.opinion for a in self.voter_list])

    def computer_trusts(self):
        return np.array([a.computer_trust for a in self.voter_list])

    def parties(self):
        return np.array([a.party for a in self.voter_list])

    def initial_parties(self):
        return np.array([a.initial_party for a in self.voter_list])

    def party_shares(self):
        """Vote share of each party (fractions that add up to 1)."""
        counts = np.bincount(self.parties(), minlength=len(self.params.party_positions))
        return counts / len(self.voter_list)

    def share_switched(self):
        """Main outcome: proportion of voters whose party differs from the start."""
        return float(np.mean(self.parties() != self.initial_parties()))

    def measures(self):
        """Summary of the whole population right now (one row of ticks.csv). [OWN]"""
        x = self.opinions()
        row = dict(tick=self.steps,                 # [Mesa] counts calls to step()
                   mean_opinion=x.mean(),
                   median_opinion=float(np.median(x)),
                   var_opinion=x.var(),
                   clusters=count_clusters(x, self.params.cluster_gap,
                                           self.params.min_cluster_size),
                   share_switched=self.share_switched(),
                   mean_trust=self.computer_trusts().mean())
        for k, share in enumerate(self.party_shares()):
            row[f"share_party{k}"] = share
        return row


# =============================================================================
# STEP 9 – DATA OUTPUT                                                   [OWN]
# =============================================================================
# One run produces up to four tables (see Experiment for the settings):
#   runs     – 1 row per run: all parameters + all final measures (main analysis)
#   ticks    – 1 row per recorded tick per run: summary measures over time
#   voters   – 1 row per voter per run, at the end (e.g. trust vs opinion)
#   examples – 1 row per voter per recorded tick, only for a few example runs
# Every row carries condition_id and seed, so tables can be joined.

def code_version():
    """Git commit of the code that produced the data (for traceability)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                capture_output=True, text=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--",
                                  os.path.basename(__file__)], cwd=here,
                                 capture_output=True, text=True).stdout.strip()
        return (commit + "+uncommitted" if changed else commit) or "unknown"
    except OSError:
        return "unknown"


def voter_table(model):
    """Every voter's current state as a table."""
    return pd.DataFrame(dict(voter_id=[v.node for v in model.voter_list],
                             opinion=model.opinions(),
                             party=model.parties(),
                             trust=model.computer_trusts()))


def run_one(params, condition_id=0, exp=EXPERIMENT, keep_example=False):
    """Run one simulation and return its output tables.                [OWN]"""
    start_time = time.time()
    model = VoterModel(params)
    x_start = model.opinions().copy()
    shares_start = model.party_shares()

    ticks = [model.measures()]                       # tick 0 = start
    examples = [voter_table(model).assign(tick=0)] if keep_example else []
    last_shares = []
    for t in range(1, params.T + 1):
        model.step()
        if t % exp.record_every == 0 or t == params.T:
            ticks.append(model.measures())
            if keep_example:
                examples.append(voter_table(model).assign(tick=t))
        if t > params.T - exp.last_ticks:
            last_shares.append(model.party_shares())

    x_end = model.opinions()
    shares_end = model.party_shares()
    shares_last = np.mean(last_shares, axis=0)
    trust = model.computer_trusts()
    ids = dict(condition_id=condition_id, seed=params.seed)

    # --- runs: one row with all parameters and final measures ---
    run_row = dict(ids, **{k: (str(v) if isinstance(v, tuple) else v)
                           for k, v in asdict(params).items() if k != "seed"})
    run_row.update(
        share_switched=model.share_switched(),
        # how much the vote shares changed in total (0 = same, 1 = completely)
        net_vote_shift=0.5 * np.abs(shares_end - shares_start).sum(),
        mean_abs_opinion_change=np.abs(x_end - x_start).mean(),
        mean_opinion=x_end.mean(),
        median_opinion=float(np.median(x_end)),
        median_shift=float(np.median(x_end) - np.median(x_start)),
        var_opinion=x_end.var(),
        clusters=count_clusters(x_end, params.cluster_gap, params.min_cluster_size),
        mean_trust=trust.mean(),
        share_trust_0=np.mean(trust == 0.0),     # voters who fully distrust the computer
        share_trust_1=np.mean(trust == 1.0),     # voters who fully trust it
    )
    for k in range(len(params.party_positions)):
        run_row[f"share_party{k}_start"] = shares_start[k]
        run_row[f"share_party{k}_end"] = shares_end[k]
        run_row[f"share_party{k}_last{exp.last_ticks}"] = shares_last[k]
    run_row["runtime_s"] = round(time.time() - start_time, 2)

    # --- ticks: summary over time ---
    ticks_df = pd.DataFrame(ticks).assign(**ids)

    # --- voters: final state of every voter ---
    key = dict(ids, p=params.p, tau_c0=params.tau_c0, e=params.e, b=params.b)
    voters_df = pd.DataFrame(dict(
        voter_id=[v.node for v in model.voter_list],
        degree=[len(v.neighbours) for v in model.voter_list],
        susceptibility=[v.susceptibility for v in model.voter_list],
        opinion_start=x_start, opinion_end=x_end,
        party_start=model.initial_parties(), party_end=model.parties(),
        trust_end=trust)).assign(**key)

    examples_df = (pd.concat(examples).assign(**key) if keep_example else None)
    return dict(run=run_row, ticks=ticks_df, voters=voters_df, examples=examples_df)


# =============================================================================
# STEP 10 – EXPERIMENT RUNNER                                            [OWN]
# =============================================================================

def make_conditions(base=PARAMS, exp=EXPERIMENT):
    """All conditions as a list of Params (seed still to be filled in).

    With p = 0 the computer is never used, so only e varies there.
    Default: 3 + 2 x 3 x 3 x 2 = 39 conditions.
    """
    conditions = []
    for p in exp.p_values:
        if p == 0:
            for e in exp.e_values:
                conditions.append(replace(base, p=0.0, e=e))
        else:
            for tau_c0 in exp.tau_c0_values:
                for e in exp.e_values:
                    for b in exp.b_values:
                        conditions.append(replace(base, p=p, tau_c0=tau_c0, e=e, b=b))
    return conditions


def _job(args):
    """One (condition, seed) job; must be a top-level function for Pool."""
    params, condition_id, exp = args
    is_example = (params.seed == exp.example_seed and
                  (params.p, params.tau_c0, params.e, params.b) in exp.example_conditions)
    return run_one(params, condition_id, exp, keep_example=is_example)


def run_experiment(base=PARAMS, exp=EXPERIMENT, processes=None):
    """Run every condition with every seed, in parallel, and save 4 files."""
    conditions = make_conditions(base, exp)
    jobs = [(replace(cond, seed=seed), cid, exp)
            for cid, cond in enumerate(conditions) for seed in exp.seeds]
    version = code_version()
    print(f"{len(conditions)} conditions x {len(exp.seeds)} seeds = {len(jobs)} runs, "
          f"T = {base.T}, code version {version}")

    start = time.time()
    results = []
    with Pool(processes or os.cpu_count()) as pool:     # one run per CPU core at a time
        for i, res in enumerate(pool.imap_unordered(_job, jobs), start=1):
            results.append(res)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs done ({time.time() - start:.0f} s)",
                      flush=True)                # show progress immediately

    order = ["condition_id", "seed"]
    runs = pd.DataFrame([r["run"] for r in results]).assign(code_version=version)
    runs = runs.sort_values(order).reset_index(drop=True)
    ticks = pd.concat([r["ticks"] for r in results]).sort_values(order + ["tick"])
    voters = pd.concat([r["voters"] for r in results]).sort_values(order + ["voter_id"])
    examples = [r["examples"] for r in results if r["examples"] is not None]

    os.makedirs(exp.out_dir, exist_ok=True)
    runs.to_csv(os.path.join(exp.out_dir, "runs.csv"), index=False)
    ticks.to_csv(os.path.join(exp.out_dir, "ticks.csv"), index=False)
    voters.to_csv(os.path.join(exp.out_dir, "voters.csv.gz"), index=False)
    if examples:
        pd.concat(examples).sort_values(order + ["tick", "voter_id"]).to_csv(
            os.path.join(exp.out_dir, "examples.csv"), index=False)
    print(f"Saved runs.csv, ticks.csv, voters.csv.gz, examples.csv to {exp.out_dir}/ "
          f"in {time.time() - start:.0f} s")
    return runs


# =============================================================================
# TESTS / QUICK CHECKS
# =============================================================================

# Tests use T = 200: fast, and the Step 3 reference values were made with T = 200.
TEST_PARAMS = replace(PARAMS, T=200)

def run(model, T):
    """Run T ticks; check after every tick that opinions stay in [0, 1]."""
    for _ in range(T):
        model.step()
        x = model.opinions()
        assert x.min() >= 0 and x.max() <= 1, "opinion left [0, 1]!"
    return model

def test_deffuant_clusters():
    """Step 2 test: larger e -> fewer clusters.                          [OWN]

    Expected (Deffuant et al., 2000): roughly 1 / (2e) clusters,
    so e = 0.1 -> several clusters, e = 0.3 -> about one cluster.
    """
    print("Step 2 test – plain Deffuant, number of clusters after T ticks")
    print(f"{'e':>5} | clusters per seed (0-4)")
    for e in (0.1, 0.2, 0.3):
        counts = []
        for seed in range(5):
            params = replace(TEST_PARAMS, e=e, seed=seed)
            model = DeffuantModel(params)
            for _ in range(params.T):
                model.step()
            x = model.opinions()
            assert x.min() >= 0 and x.max() <= 1
            counts.append(count_clusters(x, params.cluster_gap,
                                         params.min_cluster_size))
        print(f"{e:>5} | {counts}")

    # reproducibility check: same seed -> identical result
    a = DeffuantModel(TEST_PARAMS); b = DeffuantModel(TEST_PARAMS)
    for _ in range(10):
        a.step(); b.step()
    assert np.array_equal(a.opinions(), b.opinions()), "seed not reproducible!"
    print("Reproducibility check passed (same seed -> same opinions).")


def test_step3():
    """Step 3 tests: network, one interaction by hand, bounds, clusters. [OWN]"""
    print("\nStep 3 test – small-world network, receiver-only update (sigma = 0)")
    base = replace(TEST_PARAMS, sigma=0.0)        # no noise yet: that is Step 7

    # 1) network has the right shape
    model = VoterModel(base)
    degrees = [d for _, d in model.network.degree()]
    print(f"Network: {model.network.number_of_nodes()} voters, "
          f"mean degree {np.mean(degrees):.1f}, min degree {min(degrees)}, "
          f"connected: {nx.is_connected(model.network)}")
    assert np.mean(degrees) == base.mean_degree

    # 2) one interaction checked by hand
    #    x_i = 0.40, x_j = 0.45, s_i = 0.8, tau_h = 0.5, e = 0.2
    #    -> x_i_new = 0.40 + 0.8 * 0.5 * (0.45 - 0.40) = 0.42, x_j unchanged
    i = model.voter_list[0]
    j = i.neighbours[0]
    i.opinion, j.opinion, i.susceptibility = 0.40, 0.45, 0.8
    i.neighbours = [j]                       # force i to talk to j
    i.step()
    assert np.isclose(i.opinion, 0.42) and j.opinion == 0.45
    #    too far apart (0.40 vs 0.90 > e) -> nothing happens
    i.opinion, j.opinion = 0.40, 0.90
    i.step()
    assert i.opinion == 0.40
    print("Hand-checked interaction passed (0.40 -> 0.42, sender unchanged; "
          "no update when too far apart).")

    # 3) opinions stay in [0, 1] (checked every tick inside run())
    #    + clusters on small-world vs complete network.
    #    Deffuant verification: p = 0, sigma = 0, complete network,
    #    and s = 1 for everyone, so that s * tau_h = 0.5 = Deffuant's mu.
    print(f"{'e':>5} | {'small-world clusters':<22} | complete network, s=1 (= Deffuant)")
    for e in (0.1, 0.2, 0.3):
        sw, cp = [], []
        for seed in range(5):
            for net, out in (("small_world", sw), ("complete", cp)):
                params = replace(base, e=e, seed=seed, network_type=net)
                if net == "complete":
                    params = replace(params, s_min=1.0, s_max=1.0)
                x = run(VoterModel(params), params.T).opinions()
                out.append(count_clusters(x, params.cluster_gap,
                                          params.min_cluster_size))
        print(f"{e:>5} | {str(sw):<22} | {cp}")
    print("Opinions stayed in [0, 1] in every tick of every run.")

    # 4) is T = 200 long enough? how much do opinions still move at the end?
    params = replace(base, e=0.2)
    model = VoterModel(params)
    run(model, params.T - 10)
    before = model.opinions()
    run(model, 10)
    change = np.mean(np.abs(model.opinions() - before))
    print(f"Mean opinion change in the last 10 ticks (e=0.2): {change:.6f}")


def test_step4():
    """Step 4 tests: p = 0 identical to Step 3; computer summary correct. [OWN]"""
    print("\nStep 4 test – computer source")

    # 1) with p = 0 the result must be EXACTLY the same as in Step 3.
    #    Reference values: sum of all final opinions, produced by the Step 3
    #    code (commit c260756) with default parameters and seeds 0, 1, 2.
    #    (sigma = 0 because Step 3 had no noise yet.)
    step3_reference = {0: 249.6097507906078, 1: 258.30506738389084,
                       2: 254.33135948572186}
    for seed, expected in step3_reference.items():
        params = replace(TEST_PARAMS, seed=seed, p=0.0, sigma=0.0)
        got = float(run(VoterModel(params), params.T).opinions().sum())
        assert got == expected, f"seed {seed}: p=0 differs from Step 3!"
    print("p = 0 gives exactly the Step 3 results (seeds 0, 1, 2).")

    # 2) computer summary checked by hand
    model = VoterModel(replace(TEST_PARAMS, b=0.0))
    for v in model.voter_list:
        v.opinion = 0.3
    assert np.isclose(model.computer.consult(), 0.3)  # mean of 0.3s = 0.3
    model = VoterModel(replace(TEST_PARAMS, b=0.15))
    for v in model.voter_list:
        v.opinion = 0.3
    assert np.isclose(model.computer.consult(), 0.45)  # 0.3 + bias 0.15
    for v in model.voter_list:
        v.opinion = 0.95
    assert model.computer.consult() == 1.0             # 1.10 clipped to 1
    print("Computer summary passed (0.3 -> 0.3; +bias -> 0.45; 1.10 clipped to 1.0).")

    # 3) opinions stay in [0, 1] when the computer is used
    for p in (0.5, 1.0):
        run(VoterModel(replace(TEST_PARAMS, p=p)), TEST_PARAMS.T)
    print("Opinions stayed in [0, 1] with p = 0.5 and p = 1.")


def test_step5():
    """Step 5 tests: trust update by hand; plot one voter's trust.    [OWN]"""
    print("\nStep 5 test – dynamic computer trust")

    # 1) hand check. Voter: x = 0.40, s = 0.8, computer trust = 0.5, e = 0.2.
    model = VoterModel(replace(TEST_PARAMS, p=1.0))       # p = 1: always the computer
    i = model.voter_list[0]
    i.opinion, i.susceptibility, i.computer_trust = 0.40, 0.8, 0.5
    #    computer says 0.45 -> accepted:
    #    x = 0.40 + 0.8 * 0.5 * 0.05 = 0.42, trust = 0.5 + 0.02 = 0.52
    model.computer.consult = lambda: 0.45
    i.step()
    assert np.isclose(i.opinion, 0.42) and np.isclose(i.computer_trust, 0.52)
    #    computer says 0.90 -> rejected: x stays 0.42, trust = 0.52 - 0.05 = 0.47
    model.computer.consult = lambda: 0.90
    i.step()
    assert np.isclose(i.opinion, 0.42) and np.isclose(i.computer_trust, 0.47)
    #    trust never goes below 0
    i.computer_trust = 0.03
    i.step()
    assert i.computer_trust == 0.0
    print("Hand-checked trust update passed (accept: 0.50 -> 0.52, "
          "reject: 0.52 -> 0.47, floor at 0).")

    # 2) follow trust over time and save a plot
    import matplotlib
    matplotlib.use("Agg")                  # save to file, no window
    import matplotlib.pyplot as plt

    params = replace(TEST_PARAMS, p=0.5, e=0.2)
    model = VoterModel(params)
    follow = model.voter_list[:3]          # follow three voters
    history = {v.unique_id: [v.computer_trust] for v in follow}
    mean_trust = [model.computer_trusts().mean()]
    for _ in range(params.T):
        model.step()
        for v in follow:
            history[v.unique_id].append(v.computer_trust)
        mean_trust.append(model.computer_trusts().mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    for v in follow:
        ax.plot(history[v.unique_id],
                label=f"voter {v.unique_id} (final opinion {v.opinion:.2f})")
    ax.plot(mean_trust, "k--", label="mean of all voters")
    ax.set(xlabel="tick", ylabel="computer trust", ylim=(-0.02, 1.02),
           title=f"Computer trust over time (p={params.p}, e={params.e}, b={params.b})")
    ax.legend()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/test_step5_trust.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Mean computer trust: start {mean_trust[0]:.2f}, "
          f"end {mean_trust[-1]:.2f}. Plot saved to results/test_step5_trust.png")


def test_step6():
    """Step 6 test: b = 0.15, p = 1 -> the bulk of opinions moves up. [OWN]

    Note: we test the median because the mean hardly moves.
    Voters on both sides of the biased message are pulled towards it, while
    voters far below it reject it and stay behind. So the majority moves up
    and a minority is left behind, and the average stays about the same.
    The median (the middle voter) does show the shift.
    """
    print("\nStep 6 test – computer bias (p = 1, e = 0.2, seeds 0-4)")
    print(f"{'b':>5} | {'start median':>12} | {'end median':>10} | "
          f"{'end mean':>8} | {'end mean trust':>14}")
    end_medians = {}
    for b in (0.0, 0.15):
        starts, medians, means, trusts = [], [], [], []
        for seed in range(5):
            params = replace(TEST_PARAMS, p=1.0, e=0.2, b=b, seed=seed)
            model = VoterModel(params)
            starts.append(np.median(model.opinions()))
            x = run(model, params.T).opinions()
            medians.append(np.median(x))
            means.append(x.mean())
            trusts.append(model.computer_trusts().mean())
        end_medians[b] = np.mean(medians)
        print(f"{b:>5} | {np.mean(starts):>12.3f} | {np.mean(medians):>10.3f} | "
              f"{np.mean(means):>8.3f} | {np.mean(trusts):>14.3f}")
    assert end_medians[0.15] > end_medians[0.0] + 0.05, "bias did not move opinions up!"
    print("With bias the median voter ends clearly higher (the mean barely moves).")


def test_step7():
    """Step 7 tests: noisy human messages.                           [OWN]"""
    print("\nStep 7 test – noise on human messages")
    model = VoterModel(TEST_PARAMS)                       # default sigma = 0.05
    i, j = model.voter_list[0], model.voter_list[1]

    # 1) a neighbour at 0.5: messages should average 0.5 with SD ~ sigma
    j.opinion = 0.5
    msgs = np.array([i.human_message(j) for _ in range(20000)])
    print(f"Partner at 0.50: message mean {msgs.mean():.3f}, SD {msgs.std():.3f} "
          f"(expected 0.500, {TEST_PARAMS.sigma})")
    assert abs(msgs.mean() - 0.5) < 0.005 and abs(msgs.std() - TEST_PARAMS.sigma) < 0.005

    # 2) a neighbour at 0.99: noise may not push the message above 1
    j.opinion = 0.99
    msgs = np.array([i.human_message(j) for _ in range(20000)])
    assert msgs.max() <= 1.0 and msgs.min() >= 0.0
    print(f"Partner at 0.99: highest message {msgs.max():.3f} (clipped at 1).")

    # 3) sigma = 0: message is exactly the opinion
    model0 = VoterModel(replace(TEST_PARAMS, sigma=0.0))
    model0.voter_list[1].opinion = 0.42
    assert model0.voter_list[0].human_message(model0.voter_list[1]) == 0.42

    # 4) effect on whole runs (p = 0, small-world, seeds 0-4)
    print(f"{'e':>5} | {'sigma':>5} | {'opinion SD':>10} | {'change last 10 ticks':>20}")
    for e in (0.1, 0.2, 0.3):
        for sigma in (0.0, 0.05):
            sds, changes = [], []
            for seed in range(5):
                params = replace(TEST_PARAMS, e=e, sigma=sigma, seed=seed)
                model = VoterModel(params)
                run(model, params.T - 10)
                before = model.opinions()
                x = run(model, 10).opinions()
                sds.append(x.std())
                changes.append(np.mean(np.abs(x - before)))
            print(f"{e:>5} | {sigma:>5} | {np.mean(sds):>10.3f} | {np.mean(changes):>20.5f}")
    print("Opinions stayed in [0, 1] with noise.")


def test_step8():
    """Step 8 tests: parties and party switching.                    [OWN]"""
    print("\nStep 8 test – four parties")
    pos = TEST_PARAMS.party_positions

    # 1) hand checks of the closest-party rule
    assert party_of(0.10, pos) == 0
    assert party_of(0.30, pos) == 1
    assert party_of(0.60, pos) == 2
    assert party_of(0.90, pos) == 3
    assert party_of(0.25, pos) == 0     # exact tie -> lower party
    assert party_of(0.0, pos) == 0 and party_of(1.0, pos) == 3
    print("Closest-party rule passed (0.10->0, 0.30->1, 0.60->2, 0.90->3, tie 0.25->0).")

    # 2) at the start ~25% per party (uniform opinions), nobody has switched
    model = VoterModel(TEST_PARAMS)
    print(f"Start shares: {np.round(model.party_shares(), 3)}  "
          f"switched: {model.share_switched():.3f}")
    assert model.share_switched() == 0.0

    # 3) party always matches the current opinion during a run
    for _ in range(20):
        model.step()
        expected = [party_of(x, pos) for x in model.opinions()]
        assert list(model.parties()) == expected

    # 4) switching in a few conditions (e = 0.2, seeds 0-4)
    print(f"{'p':>4} | {'b':>5} | {'switched':>8} | final shares (left .. right)")
    for p, b in ((0.0, 0.0), (1.0, 0.0), (1.0, 0.15)):
        switched, shares = [], []
        for seed in range(5):
            params = replace(TEST_PARAMS, p=p, b=b, e=0.2, seed=seed)
            model = run(VoterModel(params), params.T)
            switched.append(model.share_switched())
            shares.append(model.party_shares())
        print(f"{p:>4} | {b:>5} | {np.mean(switched):>8.3f} | "
              f"{np.round(np.mean(shares, axis=0), 3)}")


def test_step9_10():
    """Steps 9–10 tests: output tables, conditions, paired seeds.    [OWN]"""
    print("\nStep 9–10 test – data output and experiment runner")

    # 1) 39 conditions, as in the proposal
    conditions = make_conditions()
    assert len(conditions) == 39
    assert sum(c.p == 0 for c in conditions) == 3
    print(f"make_conditions: {len(conditions)} conditions (3 with p = 0).")

    # 2) one short run: tables have the right size and sensible values
    params = replace(TEST_PARAMS, p=0.5, b=0.15, T=30, seed=3)
    out = run_one(params, condition_id=7, keep_example=True)
    r, ticks, voters, ex = out["run"], out["ticks"], out["voters"], out["examples"]
    shares = [r[f"share_party{k}_end"] for k in range(4)]
    assert np.isclose(sum(shares), 1.0)
    assert list(ticks["tick"]) == [0, 10, 20, 30]           # record_every = 10
    assert len(voters) == params.N and len(ex) == params.N * 4
    assert 0 <= r["share_switched"] <= 1 and 0 <= r["net_vote_shift"] <= 1
    assert r["condition_id"] == 7 and r["seed"] == 3 and r["b"] == 0.15
    # numbers in the tables agree with each other
    assert np.isclose(ticks["share_switched"].iloc[-1], r["share_switched"])
    assert np.isclose(np.mean(voters["party_end"] != voters["party_start"]),
                      r["share_switched"])
    print("run_one: table sizes and values consistent (shares sum to 1, "
          "ticks/voters/runs agree).")

    # 3) paired design: same seed and p -> identical random draws for any b
    #    (so 'b = 0.15 minus b = 0' compares like with like)
    states = []
    for b in (0.0, 0.15):
        model = VoterModel(replace(TEST_PARAMS, p=0.5, b=b, T=30, seed=3))
        run(model, 30)
        states.append(model.random.getstate())
    assert states[0] == states[1]
    print("Paired seeds: b = 0 and b = 0.15 use exactly the same random draws.")

    # 4) running in parallel gives exactly the same result as running directly
    exp = replace(EXPERIMENT, example_conditions=())
    with Pool(2) as pool:
        par = pool.map(_job, [(params, 7, exp)])[0]["run"]
    direct = run_one(params, 7, exp)["run"]
    assert all(par[k] == direct[k] for k in par if k != "runtime_s")
    print("Parallel run identical to direct run.")


def test_step11():
    """Step 11 tests: bimodal start switch.                          [OWN]"""
    print("\nStep 11 test – bimodal start")

    # 1) default is still uniform (also guaranteed by the Step 4 test, which
    #    reproduces the Step 3 results exactly)
    assert TEST_PARAMS.initial_dist == "uniform"

    # 2) bimodal: inside [0, 1], two groups around the peaks, few voters at 0.5
    lo, hi = TEST_PARAMS.bimodal_peaks
    xs = np.concatenate([
        VoterModel(replace(TEST_PARAMS, initial_dist="bimodal", seed=s)).opinions()
        for s in range(10)])                                   # 5,000 voters
    left, right = xs[xs < 0.5], xs[xs >= 0.5]
    near_middle = np.mean(np.abs(xs - 0.5) < 0.05)
    near_peak = np.mean(np.abs(xs - lo) < 0.05)
    print(f"Bimodal start (10 seeds): left group mean {left.mean():.3f} "
          f"(peak {lo}), right group mean {right.mean():.3f} (peak {hi}), "
          f"share left {len(left) / len(xs):.2f}")
    print(f"Share of voters within 0.05 of: 0.5 -> {near_middle:.3f}, "
          f"peak {lo} -> {near_peak:.3f}  (uniform start: 0.100 everywhere)")
    assert xs.min() >= 0 and xs.max() <= 1
    assert abs(len(left) / len(xs) - 0.5) < 0.03
    assert abs(left.mean() - lo) < 0.02 and abs(right.mean() - hi) < 0.02
    assert near_middle < near_peak          # a dip between the two peaks

    # 3) unknown setting gives a clear error
    try:
        VoterModel(replace(TEST_PARAMS, initial_dist="triangle"))
        raise AssertionError("no error for unknown initial_dist")
    except ValueError:
        pass
    print("Bimodal switch passed (bounds, two peaks, dip at 0.5, clear error).")


def run_all_tests():
    test_deffuant_clusters()
    test_step3()
    test_step4()
    test_step5()
    test_step6()
    test_step7()
    test_step8()
    test_step9_10()
    test_step11()
    print("\nAll tests passed.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "experiment"
    if mode == "test":
        run_all_tests()
    elif mode == "quick":
        # fast check of the whole pipeline: 2 seeds, 50 ticks, separate folder
        run_experiment(replace(PARAMS, T=50),
                       replace(EXPERIMENT, seeds=(0, 1),
                               out_dir=os.path.join("results", "quick")))
    elif mode == "pilot":
        # pilot used to choose the run length T: full grid,
        # T = 1000, separate folder. Because seeds are identical, tick t of this
        # pilot equals tick t of a shorter run, so any T <= 1000 can be judged.
        run_experiment(replace(PARAMS, T=1000),
                       replace(EXPERIMENT, out_dir=os.path.join("results", "pilot_T1000")))
    elif mode == "bimodal":
        # extension (Step 11): bimodal start on a SUBSET of the grid:
        # p = 0 (baseline) and p = 0.5; tau_c0 = 0.5; e = 0.2, 0.3; b = 0, 0.15
        # -> 2 + 1 x 1 x 2 x 2 = 6 conditions x 30 seeds = 180 runs
        run_experiment(replace(PARAMS, initial_dist="bimodal"),
                       replace(EXPERIMENT, p_values=(0.0, 0.5), tau_c0_values=(0.5,),
                               e_values=(0.2, 0.3), b_values=(0.0, 0.15),
                               out_dir=os.path.join("results", "bimodal")))
    elif mode == "experiment":
        run_experiment()
    else:
        print("Use: python project-assignment-group-20.py "
              "[experiment | quick | pilot | bimodal | test]")
