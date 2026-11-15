# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/kiroroai/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/kiroroai/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One zero-config server that serves the OpenAI Chat Completions API and the Anthropic Messages API — in JSON and in server-sent events — over the free tiers of ten model providers. It reads the trial's brief, probes every provider's own catalog route, registers the catalog it found, and then answers each request with the strongest free model that meets the request's requirements (tools, vision, JSON mode, context length), ranked by the brief's quality table. When a provider refuses — budget spent, failure schedule, outage — the request fails over to the next-best model instead of failing with it. All three provider wires are spoken (OpenAI, Anthropic, Gemini), with OpenAI Chat Completions as the translation pivot, so the best model can serve whichever client wire asked. `node main.ts --list` prints the catalog it registers.

## Run it

```sh
node main.ts --list   # print the catalog document and exit
node main.ts          # serve: needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE (a trial brief)
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=wfy52w8q&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
