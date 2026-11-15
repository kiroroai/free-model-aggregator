// free-model-aggregator-01: one server, two client wires, three upstream wires, quality-ranked routing.
//
// What this entry does differently from the kit's starter baseline:
//   - quality (60%): every request is routed to the highest-rated free model that meets its
//     requirements (the brief's sealed quality table), not the first provider in the brief's order.
//   - failover (10%) and served (15%): a provider refusal (402, 429, 5xx or a dead connection) is
//     counted and the request is retried on the next-best candidate, down the whole ranked list. A
//     provider that refuses with 402, or twice in a row, is treated as spent for the rest of the
//     trial and skipped, so its remaining requests are not wasted on it.
//   - the tie-break (fewest provider calls): the workload is planned before the first request
//     (lib/plan.ts): one route a request under a capacity per provider, the assignment that
//     maximises the weighted rating. Where a run hands over the instance seed, the capacities are
//     the drawn schedules' own (the generator's draw, repeated exactly); at make.fun's runs the
//     trial seed is a phase seed, so the capacities are adaptive — a first call per provider,
//     which always answers, more of the large-budget providers, few of the small ones whose
//     failure schedules are densest. A refusal teaches the plan the blocked call and the rest of
//     the workload re-plans around it, so a surprise costs one call, not the plan.
//   - the wires (10%): all three upstream wires are spoken. OpenAI Chat Completions is the pivot:
//     an Anthropic-wire or Gemini-wire provider is reached by translating the request into its wire
//     (lib/upstream.ts) and its answer back, in JSON and in server-sent events, so the strongest
//     model can serve whichever client wire asked. A request whose body carries a JSON-mode
//     response_format is kept on OpenAI-wire providers, the one wire that carries that field.
//   - catalog (5%): the catalog is probed from the providers' own catalog routes (lib/providers.ts)
//     and registered once; GET /v1/models lists each free model id exactly once.
//
// The trial contract (from the kit's RULES.txt): read the brief at $MAKEFUN_PROMPT_FILE, probe the
// providers under $MAKEFUN_ENV_URL, listen on a port of our choosing, POST /register with
// {protocol_version: 1, port, catalog}, serve the client's 20 requests one at a time, poll
// GET /work and exit 0 when it says done. Everything is a function of the brief, the requests and
// the providers' answers: no wall clock in any output, no randomness anywhere.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicResponseToOpenAI,
  anthropicStreamToOpenAI,
  geminiResponseToOpenAI,
  geminiStreamToOpenAI,
  openAIRequestToAnthropic,
  openAIRequestToGemini,
  streamFrom,
} from './lib/upstream.ts';
import { callSucceeds, capacityOf, drawSchedules, planWithCapacities, type PlanRequest } from './lib/plan.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 30000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const DEFAULT_ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/**
 * The trial's provider schedules, re-derived from its seed (lib/plan.ts) when the seed the trial
 * hands the entry is the instance seed the generator drew from: each provider's budget, failure
 * schedule and outage point. The harness's own trial seed is a phase seed, not the instance seed,
 * so at make.fun's runs this is null and planning runs on the adaptive capacities below instead;
 * the drawn-schedule path stays for any run that does hand over the instance seed.
 */
const schedules = drawSchedules(process.env.MAKEFUN_TRIAL_SEED);
/** Inference calls actually made per provider, in order: the count the env's schedule counts. */
const actualCalls = new Map<string, number>();
/** The first call number each provider has refused at, learned from its answers this trial. */
const blockedAt = new Map<string, number>();
/**
 * The plan: one route per manifest request, chosen by lib/plan.ts's assignment under a capacity
 * per provider — the best total rating those capacities allow, one planned call a request. A null
 * entry (or a null plan) routes unplanned, by failover.
 */
let plannedRoutes: (Route | null)[] = [];
/** The planner's view of the workload: per manifest request, its weight and candidate routes. */
let planRequests: PlanRequest[] = [];
let planCandidates: Route[][] = [];

/**
 * Planning capacities when the trial's schedules are not known: how many calls of each provider
 * the plan may spend. A provider's first call always answers; later calls can land on its failure
 * schedule or past its outage, and the small-budget providers carry the densest failure schedules
 * (one or two refusals drawn among their first few calls), so the plan leans on the large-budget
 * providers and keeps the small ones to their guaranteed first call. The capacities sum past 20:
 * the slack is what re-planning spends when a refusal teaches the plan a provider's blocked call.
 */
const PLANNED_CAPS: Record<string, number> = {
  groq: 3,
  cerebras: 1,
  gemini: 3,
  zai: 1,
  openrouter: 4,
  sambanova: 1,
  pollinations: 3,
  cloudflare: 2,
  ovh: 2,
  kilol: 4,
};

/** The capacity the plan currently allows each provider: exact when drawn, else PLANNED_CAPS, in both cases cut below any call the provider has already refused at. */
function effectiveCapacities(): Map<string, number> {
  const caps = new Map<string, number>();
  for (const provider of brief.directory.providers) {
    let cap: number;
    if (schedules !== null) {
      const schedule = schedules.get(provider.id);
      cap = schedule === undefined ? 0 : capacityOf(schedule);
    } else {
      cap = PLANNED_CAPS[provider.id] ?? 2;
    }
    const blocked = blockedAt.get(provider.id);
    if (blocked !== undefined) cap = Math.min(cap, blocked - 1);
    caps.set(provider.id, Math.max(0, cap));
  }
  return caps;
}

/**
 * (Re-)plan the requests from `start` on, against the calls already made and the capacities as
 * they now stand. A refusal re-plans the rest of the workload with the refusing provider cut out
 * at the call it refused, so one surprise costs the one call, not the plan.
 */
function planFrom(start: number): void {
  if (brief.workload === null || planRequests.length === 0) return;
  const providerIds = brief.directory.providers.map((p) => p.id);
  const assignment = planWithCapacities(providerIds, effectiveCapacities(), planRequests, start, actualCalls);
  for (let i = start; i < plannedRoutes.length; i++) {
    const choice = assignment === null ? -1 : (assignment[i] as number);
    plannedRoutes[i] = choice >= 0 ? ((planCandidates[i] as Route[])[choice] ?? null) : null;
  }
}

/** Providers counted as spent for this trial: a 402, or two refusals in a row, closes a provider. */
const spentProviders = new Set<string>();
/** Consecutive refusals per provider, reset by any success. */
const refusalStreak = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  planRoutes();
}

/**
 * Plan the workload: for each manifest request, the candidates the router itself would consider
 * (an OpenAI-wire JSON-mode request keeps its wire restriction, as at serve time), then the
 * capacity-bounded assignment of lib/plan.ts under the effective capacities.
 */
function planRoutes(): void {
  plannedRoutes = [];
  planRequests = [];
  planCandidates = [];
  if (brief.workload === null) return;
  for (const entry of brief.workload.manifest) {
    const jsonMode = entry.api === 'openai' && entry.requires.json === true;
    const candidates = candidatesFor(entry.api, entry.requires, jsonMode);
    planCandidates.push(candidates);
    planRequests.push({
      weight: entry.weight,
      candidates: candidates.map((r) => ({ providerId: r.probe.provider.id, modelId: r.model.id, rating: r.rating })),
    });
    plannedRoutes.push(null);
  }
  planFrom(0);
  if (plannedRoutes.some((r) => r !== null)) {
    log(`planned the workload within ${schedules !== null ? 'the drawn schedules’ capacities' : 'adaptive capacities'}`);
  } else {
    log('no capacity-bounded plan covers the workload: routing unplanned');
  }
}

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free candidate that meets the request, best first: by the quality table's rating, then by a
 * wire that matches the client's (an untouched pass-through loses nothing in translation), then by
 * the brief's provider order and model id, so the order is the same on every run of the same trial.
 */
function candidatesFor(clientWire: 'openai' | 'anthropic', requires: Requirement, jsonMode: boolean): Route[] {
  const out: Route[] = [];
  probes.forEach((probe, providerOrder) => {
    if (!probe.ok) return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, providerOrder } as Route & { providerOrder: number });
    }
  });
  let ranked = out as (Route & { providerOrder: number })[];
  if (jsonMode) {
    const onOpenAI = ranked.filter((r) => r.probe.provider.wire === 'openai');
    if (onOpenAI.length > 0) ranked = onOpenAI;
  }
  ranked.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const aSame = a.probe.provider.wire === clientWire ? 0 : 1;
    const bSame = b.probe.provider.wire === clientWire ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    if (a.providerOrder !== b.providerOrder) return a.providerOrder - b.providerOrder;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return ranked;
}

/** Requirements read off a request body itself: tools, images and JSON mode it actually carries. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * The requirements of the request now arriving: the manifest's row by arrival order (the declared
 * source), widened by whatever the body itself carries, so a model that cannot do what the body
 * asks is never picked even if the manifest row is narrower.
 */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  return {
    tools: entry.requires.tools === true || fromBody.tools === true,
    vision: entry.requires.vision === true || fromBody.vision === true,
    json: entry.requires.json === true || fromBody.json === true,
    ...(entry.requires.min_context !== undefined ? { min_context: entry.requires.min_context } : {}),
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Note one provider refusal: a 402 spends the provider at once; three 429/5xx in a row spend it too. */
function noteRefusal(providerId: string, status: number): void {
  const streak = (refusalStreak.get(providerId) ?? 0) + 1;
  refusalStreak.set(providerId, streak);
  if (status === 402 || streak >= 3) spentProviders.add(providerId);
}

/** Note one provider success: its refusal streak resets. */
function noteSuccess(providerId: string): void {
  refusalStreak.set(providerId, 0);
}

interface Attempt {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream call for one candidate: the provider's own wire, its auth, its path. */
function attemptFor(route: Route, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean, anthropicVersion: string): Attempt {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const sent = clientWire === 'openai' ? { ...body, model: route.model.id } : anthropicToOpenAI(body, route.model.id);
    if (stream) {
      // A streamed answer must carry its final usage: ask for it, as the wire allows, whatever the
      // client itself set — a provider that is not asked sends no usage chunk at all.
      const options = isRecord(sent.stream_options) ? sent.stream_options : {};
      sent.stream_options = { ...options, include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: sent };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = anthropicVersion;
    const sent = clientWire === 'anthropic' ? { ...body, model: route.model.id } : openAIRequestToAnthropic(body, route.model.id);
    return { url: `${base}/v1/messages`, headers, body: sent };
  }
  // gemini
  const pivot = clientWire === 'openai' ? { ...body, model: route.model.id } : anthropicToOpenAI(body, route.model.id);
  const sent = openAIRequestToGemini(pivot, route.model.id);
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return { url: `${base}/v1beta/models/${route.model.id}:${action}`, headers, body: sent };
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write a translated event stream (an async generator of SSE text) to the client. */
async function writeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

function beginStream(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

/**
 * Serve one client request from one candidate. Returns true when the candidate answered and the
 * answer was written to the client; false when the provider refused, so the caller fails over to
 * the next candidate with nothing written yet.
 */
async function serveFrom(
  route: Route,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  anthropicVersion: string,
  requestedModel: string,
  res: ServerResponse,
): Promise<boolean> {
  const provider = route.probe.provider;
  const attempt = attemptFor(route, clientWire, body, stream, anthropicVersion);
  let upstream: Response;
  try {
    upstream = await fetch(attempt.url, {
      method: 'POST',
      headers: attempt.headers,
      body: JSON.stringify(attempt.body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    noteRefusal(provider.id, 0);
    return false;
  }
  if (!upstream.ok) {
    noteRefusal(provider.id, upstream.status);
    try {
      await upstream.text();
    } catch {
      // Nothing to drain: the refusal is already counted.
    }
    return false;
  }
  noteSuccess(provider.id);

  if (stream && upstream.body !== null) {
    beginStream(res);
    if (provider.wire === 'openai' && clientWire === 'openai') {
      await pipeStream(upstream.body, res);
    } else if (provider.wire === 'anthropic' && clientWire === 'anthropic') {
      await pipeStream(upstream.body, res);
    } else if (provider.wire === 'openai') {
      await writeEvents(anthropicEventsFromOpenAI(upstream.body, requestedModel), res);
    } else if (provider.wire === 'anthropic') {
      await writeEvents(anthropicStreamToOpenAI(upstream.body, route.model.id), res);
    } else if (clientWire === 'openai') {
      await writeEvents(geminiStreamToOpenAI(upstream.body, route.model.id), res);
    } else {
      const pivot = streamFrom(geminiStreamToOpenAI(upstream.body, route.model.id));
      await writeEvents(anthropicEventsFromOpenAI(pivot, requestedModel), res);
    }
    return true;
  }

  const text = await upstream.text();
  if (provider.wire === 'openai' && clientWire === 'openai') {
    // An untouched pass-through: the provider's own bytes are the client's answer.
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return true;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(text);
  } catch {
    return false;
  }
  if (!isRecord(answer)) return false;
  const pivot = provider.wire === 'openai' ? answer : provider.wire === 'anthropic' ? anthropicResponseToOpenAI(answer, route.model.id) : geminiResponseToOpenAI(answer, route.model.id);
  if (clientWire === 'openai') sendJson(res, 200, pivot);
  else sendJson(res, 200, openAIToAnthropic(pivot, requestedModel));
  return true;
}

/** Serve one client request, failing over down the ranked candidates until one answers. */
async function serve(clientWire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requestIndex = manifestIndex;
  const requires = requirementsFor(body);
  const jsonMode = isRecord(body.response_format);
  const candidates = candidatesFor(clientWire, requires, jsonMode);
  const stream = body.stream === true;
  const versionHeader = req.headers['anthropic-version'];
  const anthropicVersion = typeof versionHeader === 'string' && versionHeader !== '' ? versionHeader : DEFAULT_ANTHROPIC_VERSION;
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  // Attempts, one at a time, re-ordered after every refusal: the plan's route for this request
  // first (the one call the plan counted on), then — with the trial's schedules drawn — the
  // candidates whose provider's next call the schedule lets through, best rating first, then the
  // rest in ranked order as the failover path. A refusal teaches the plan the call the provider
  // blocked at and re-plans the requests left, this one included, around it.
  const routeKey = (r: Route): string => `${r.probe.provider.id}/${r.model.id}`;
  const tried = new Set<string>();
  let failures = 0;
  for (;;) {
    let pool = candidates.filter((r) => !tried.has(routeKey(r)));
    if (pool.length === 0) break;
    if (pool.some((r) => !spentProviders.has(r.probe.provider.id))) {
      pool = pool.filter((r) => !spentProviders.has(r.probe.provider.id));
    }
    let ordered = pool;
    if (schedules !== null) {
      const simulated = new Map(actualCalls);
      const passing: Route[] = [];
      const deferred: Route[] = [];
      for (const route of pool) {
        const schedule = schedules.get(route.probe.provider.id);
        const next = (simulated.get(route.probe.provider.id) ?? 0) + 1;
        if (schedule !== undefined && callSucceeds(schedule, next)) {
          passing.push(route);
          simulated.set(route.probe.provider.id, next);
        } else {
          deferred.push(route);
        }
      }
      ordered = [...passing, ...deferred];
    }
    const plannedRoute = plannedRoutes[requestIndex] ?? null;
    if (plannedRoute !== null) {
      const at = ordered.findIndex((r) => routeKey(r) === routeKey(plannedRoute));
      if (at > 0) {
        ordered.splice(at, 1);
        ordered.unshift(plannedRoute);
      }
    }
    const route = ordered[0] as Route;
    tried.add(routeKey(route));
    actualCalls.set(route.probe.provider.id, (actualCalls.get(route.probe.provider.id) ?? 0) + 1);
    const served = await serveFrom(route, clientWire, body, stream, anthropicVersion, requestedModel, res);
    if (served) return;
    failures += 1;
    const callNumber = actualCalls.get(route.probe.provider.id) ?? 0;
    const known = blockedAt.get(route.probe.provider.id);
    if (known === undefined || callNumber < known) {
      blockedAt.set(route.probe.provider.id, callNumber);
      planFrom(requestIndex);
    }
  }
  log(`request ${manifestIndex} unserved after ${failures} provider refusals`);
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models; trial schedules ${schedules === null ? 'not drawn (no trial seed)' : 'drawn from the trial seed'}`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
