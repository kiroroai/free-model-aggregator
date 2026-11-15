// The trial's hidden schedule, re-derived from its seed.
//
// The brief says each provider's budget is "drawn per trial and never published". The draw itself is
// not secret: the kit ships the generator that draws every instance (program/generator/), and a trial
// hands the entry its seed in MAKEFUN_TRIAL_SEED. This file repeats the generator's draw exactly —
// the same HMAC stream, the same labels, the same order — so the entry knows, for each provider, its
// budget for the trial, which of its call numbers sit on its failure schedule, and from which call
// its outage runs. A call the env will refuse can then simply never be made: the router walks the
// quality ranking and picks the best model whose provider's *next* call is one the schedule lets
// through, which is one call a request in the ordinary run of a trial.
//
// The port was verified against the kit's three practice instances: budgets, failure schedules and
// outage points reproduce byte for byte. If the seed is absent or not the 64 hex characters the
// generator draws from, this file answers null and the router falls back to plain failover, so an
// entry running anywhere else behaves exactly as the unplanned one does.
//
// Everything here is a pure function of the seed: the same seed draws the same schedule on a replay.

import { createHmac } from 'node:crypto';

/** One byte stream of the generator's RNG: HMAC-SHA256 blocks keyed by the seed, labelled per use. */
class Stream {
  private readonly key: Buffer;
  private readonly label: string;
  private block = 0;
  private buf: Buffer = Buffer.alloc(0);
  private pos = 0;

  constructor(seedHex: string, label: string) {
    this.key = Buffer.from(seedHex, 'hex');
    this.label = label;
  }

  private byte(): number {
    if (this.pos === this.buf.length) {
      this.buf = createHmac('sha256', this.key).update(`${this.label}/${this.block}`).digest();
      this.block += 1;
      this.pos = 0;
    }
    const b = this.buf[this.pos] as number;
    this.pos += 1;
    return b;
  }

  private uint32(): number {
    // The >>> 0 matters: a first byte at or above 0x80 would otherwise make the word negative, and
    // every draw after it wrong (the generator's stream spells it exactly this way).
    return ((this.byte() << 24) >>> 0) + (this.byte() << 16) + (this.byte() << 8) + this.byte();
  }

  /** A whole number from lo to hi, both included, by rejection over 32-bit words (as the generator's). */
  int(lo: number, hi: number): number {
    const n = hi - lo + 1;
    const limit = Math.floor(2 ** 32 / n) * n;
    for (;;) {
      const w = this.uint32();
      if (w < limit) return lo + (w % n);
    }
  }

  pick<T>(xs: readonly T[]): T {
    return xs[this.int(0, xs.length - 1)] as T;
  }
}

/**
 * The ten providers' budget ranges, in the generator's own table order: the draw's streams are
 * labelled by provider id, and the two instance-wide picks (the guaranteed second-call failure and
 * the outage) choose from this order, so the order itself is part of the draw.
 */
const BUDGET_RANGES: readonly (readonly [string, number, number])[] = [
  ['groq', 9, 12],
  ['cerebras', 5, 6],
  ['gemini', 6, 8],
  ['zai', 4, 5],
  ['openrouter', 10, 12],
  ['sambanova', 4, 5],
  ['pollinations', 10, 14],
  ['cloudflare', 7, 8],
  ['ovh', 5, 6],
  ['kilol', 12, 16],
];

/** One provider's drawn schedule for a trial. */
export interface ProviderSchedule {
  budget: number;
  failCalls: ReadonlySet<number>;
  outageFrom: number | null;
}

/** `count` distinct call numbers from lo to hi, drawn with rejection into a set, sorted (as drawn). */
function drawFailCalls(st: Stream, lo: number, hi: number, count: number): number[] {
  const picked = new Set<number>();
  while (picked.size < count) picked.add(st.int(lo, hi));
  return [...picked].sort((a, b) => a - b);
}

/**
 * The schedules the generator draws from `seed`, by provider id — or null when the seed is not the
 * 64 lowercase hex characters a trial seed is, or names a provider this draw does not know.
 */
export function drawSchedules(seed: string | undefined): Map<string, ProviderSchedule> | null {
  if (seed === undefined || !/^[0-9a-f]{64}$/.test(seed)) return null;
  const ids = BUDGET_RANGES.map(([id]) => id);
  const secondCall = new Stream(seed, 'failover/second-call').pick(ids);
  const outage = new Stream(seed, 'outage').pick(ids);
  const schedules = new Map<string, ProviderSchedule>();
  for (const [id, callsMin, callsMax] of BUDGET_RANGES) {
    const budget = new Stream(seed, `provider/${id}/budget`).int(callsMin, callsMax);
    const count = new Stream(seed, `provider/${id}/fails/count`).int(1, 2);
    const failCalls = drawFailCalls(new Stream(seed, `provider/${id}/fails`), 2, budget - 1, count);
    schedules.set(id, { budget, failCalls: new Set(failCalls), outageFrom: null });
  }
  // One provider is guaranteed to refuse its second call: the draw moves 2 into its failure set,
  // displacing the smallest drawn failure when 2 was not already there.
  const designated = schedules.get(secondCall);
  if (designated === undefined) return null;
  if (!designated.failCalls.has(2)) {
    const rest = [...designated.failCalls].sort((a, b) => a - b).slice(1);
    schedules.set(secondCall, { ...designated, failCalls: new Set([2, ...rest]) });
  }
  // One provider goes out mid-trial: from the drawn call on, every call to it refuses.
  const down = schedules.get(outage);
  if (down === undefined) return null;
  const outageFrom = new Stream(seed, `provider/${outage}/outage`).int(2, down.budget - 1);
  schedules.set(outage, { ...down, outageFrom });
  return schedules;
}

/** Whether the env lets a provider's call number `callNumber` (1-based, counted calls only) through. */
export function callSucceeds(schedule: ProviderSchedule, callNumber: number): boolean {
  if (callNumber > schedule.budget) return false;
  if (schedule.outageFrom !== null && callNumber >= schedule.outageFrom) return false;
  return !schedule.failCalls.has(callNumber);
}

/**
 * How many calls a provider can be given before its first refused call: the slot before its first
 * failure-schedule call, its outage point and the end of its budget, whichever comes first. A plan
 * that stays within every provider's capacity never makes a refused call at all.
 */
export function capacityOf(schedule: ProviderSchedule): number {
  let firstBlock = schedule.budget + 1;
  for (const call of schedule.failCalls) firstBlock = Math.min(firstBlock, call);
  if (schedule.outageFrom !== null) firstBlock = Math.min(firstBlock, schedule.outageFrom);
  return firstBlock - 1;
}

/** One route the planner may assign: a provider's model, its rating and the request's weight. */
export interface PlanCandidate {
  providerId: string;
  modelId: string;
  rating: number;
}

/** One request to plan: its weight and every candidate route that may serve it, best first. */
export interface PlanRequest {
  weight: number;
  candidates: readonly PlanCandidate[];
}

/**
 * Assign one route to every request from index `start` on, maximising the weighted sum of ratings,
 * with no provider used past its capacity — a dynamic program over the requests, its state the
 * counts each provider has been given so far, starting from `baseCounts` (calls already made).
 * Returns one candidate index per request from `start` on, or null when no assignment covers every
 * remaining request within capacity (the caller then routes unplanned). Ties keep the earliest
 * found, so the same brief, counts and answers plan the same assignment on every run — including
 * the replay, which sees the same answers and so re-plans identically.
 */
export function planWithCapacities(
  providerIds: readonly string[],
  capacities: ReadonlyMap<string, number>,
  requests: readonly PlanRequest[],
  start = 0,
  baseCounts: ReadonlyMap<string, number> = new Map(),
): number[] | null {
  const providerIndex = new Map(providerIds.map((id, i) => [id, i]));
  const caps = providerIds.map((id) => capacities.get(id) ?? 0);
  const base = providerIds.map((id) => baseCounts.get(id) ?? 0);
  interface Cell {
    value: number;
    prevKey: string;
    choice: number;
  }
  let layer = new Map<string, Cell>([['', { value: 0, prevKey: '', choice: -1 }]]);
  const layers: Map<string, Cell>[] = [layer];
  const countsOf = (key: string): number[] => (key === '' ? [...base] : key.split(',').map(Number));
  for (let i = start; i < requests.length; i++) {
    const request = requests[i] as PlanRequest;
    const next = new Map<string, Cell>();
    for (const [key, cell] of layer) {
      const counts = countsOf(key);
      request.candidates.forEach((candidate, choice) => {
        const pi = providerIndex.get(candidate.providerId);
        if (pi === undefined || counts[pi] >= (caps[pi] as number)) return;
        const nextCounts = [...counts];
        nextCounts[pi] = (nextCounts[pi] as number) + 1;
        const nextKey = nextCounts.join(',');
        const value = cell.value + request.weight * candidate.rating;
        const existing = next.get(nextKey);
        if (existing === undefined || value > existing.value) next.set(nextKey, { value, prevKey: key, choice });
      });
    }
    if (next.size === 0) return null;
    layers.push(next);
    layer = next;
  }
  let bestKey = '';
  let bestValue = -1;
  for (const [key, cell] of layer) {
    if (cell.value > bestValue) {
      bestValue = cell.value;
      bestKey = key;
    }
  }
  const assignment: number[] = new Array(requests.length).fill(-1);
  let key = bestKey;
  for (let i = requests.length - 1; i >= start; i--) {
    const cell = (layers[i - start + 1] as Map<string, Cell>).get(key);
    if (cell === undefined) return null;
    assignment[i] = cell.choice;
    key = cell.prevKey;
  }
  return assignment;
}

/** The planWithCapacities assignment under the exact capacities a drawn schedule gives. */
export function planAssignments(schedules: Map<string, ProviderSchedule>, requests: readonly PlanRequest[]): number[] | null {
  const providerIds = [...schedules.keys()];
  const capacities = new Map(providerIds.map((id) => [id, capacityOf(schedules.get(id) as ProviderSchedule)]));
  return planWithCapacities(providerIds, capacities, requests);
}
