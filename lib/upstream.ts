// The two upstream wires the starter baseline does not speak: Anthropic and Gemini.
//
// Architecture: OpenAI Chat Completions is the pivot. A client request in either client wire is first
// expressed as a Chat Completions body (wire.ts already turns an Anthropic Messages body into one),
// then this file turns that pivot body into the upstream provider's own wire:
//
//   pivot  -> Anthropic Messages   (openAIRequestToAnthropic)
//   pivot  -> Gemini generateContent (openAIRequestToGemini)
//
// and a provider's answer comes back through the pivot:
//
//   Anthropic Messages answer  -> Chat Completions answer  (anthropicResponseToOpenAI)
//   Gemini generateContent answer -> Chat Completions answer (geminiResponseToOpenAI)
//
// Streaming answers are converted event by event into Chat Completions SSE chunks, so the one
// Chat-Completions-to-Anthropic stream translator in wire.ts can serve an Anthropic client from any
// upstream wire. Everything is a pure function of its arguments: no clock, no randomness, so a replay
// of the same trial produces the same bytes.

import { isRecord, type Json } from './providers.ts';

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

// ---------------------------------------------------------------------------
// pivot -> Anthropic (the request an anthropic-wire provider receives)
// ---------------------------------------------------------------------------

/** One OpenAI content part as an Anthropic content block: text or an inline/url image; null otherwise. */
function anthropicBlockOf(part: unknown): Json | null {
  if (typeof part === 'string') return { type: 'text', text: part };
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+);base64,(.*)$/.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * A Chat Completions request as an Anthropic Messages request naming `model`: the system messages join
 * into the system prompt, tool_calls become tool_use blocks, tool results become tool_result blocks in a
 * user turn, and tools, tool choice, the output cap and stop sequences are rewritten in Anthropic's
 * spellings. Anthropic requires a max_tokens, so a request without one is given a plain default.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: textOf(raw.content) !== '' ? textOf(raw.content) : typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? ''),
          },
        ],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: unknown = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            input = JSON.parse(call.function.arguments);
          } catch {
            input = {};
          }
        }
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: isRecord(input) ? input : {},
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user turn: a string, or an array of parts.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks = raw.content.flatMap((part) => {
        const block = anthropicBlockOf(part);
        return block === null ? [] : [block];
      });
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages, max_tokens: 1024 };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (cap !== null) out.max_tokens = cap;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Anthropic -> pivot (the answer an anthropic-wire provider gives)
// ---------------------------------------------------------------------------

/** An Anthropic Messages answer as a Chat Completions answer. */
export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const textParts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: typeof body.id === 'string' ? body.id : 'aggregate',
    object: 'chat.completion',
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: textParts.length > 0 ? textParts.join('') : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: finishReasonOf(body.stop_reason),
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** One server-sent event frame: `event: <name>` when named, the data line, the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split an SSE byte stream into its frames (the text between blank lines), as they complete. */
export async function* sseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/** The JSON payload of one SSE frame's data lines, or null when it carries none or is [DONE]. */
export function sseData(frame: string): Json | null {
  const lines = frame.split('\n').flatMap((line) => {
    const trimmed = line.trim();
    return trimmed.startsWith('data:') ? [trimmed.slice(5).trim()] : [];
  });
  if (lines.length === 0 || lines[0] === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(lines.join(''));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A Chat Completions chunk as one SSE frame. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage?: Json): string {
  const chunk: Json = {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    model,
    choices: finish === null && usage !== undefined ? [] : [{ index: 0, delta, finish_reason: finish }],
  };
  if (usage !== undefined) chunk.usage = usage;
  return sseEvent(null, chunk);
}

/**
 * An Anthropic Messages SSE stream as Chat Completions SSE chunks: text deltas become content deltas,
 * tool_use blocks become tool_calls deltas with input_json_delta as their arguments, and message_delta's
 * stop reason and usage close the stream with a finish chunk, a usage chunk and [DONE].
 */
export async function* anthropicStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  let toolIndex = -1;
  let roleSent = false;
  for await (const frame of sseFrames(upstream)) {
    const data = sseData(frame);
    if (data === null) continue;
    if (data.type === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
      if (!roleSent) {
        roleSent = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      continue;
    }
    if (data.type === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      toolIndex += 1;
      const block = data.content_block;
      yield openAIChunk(
        model,
        {
          tool_calls: [
            {
              index: toolIndex,
              id: typeof block.id === 'string' ? block.id : `toolu_${toolIndex}`,
              type: 'function',
              function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' },
            },
          ],
        },
        null,
      );
      continue;
    }
    if (data.type === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        yield openAIChunk(model, { content: data.delta.text }, null);
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        yield openAIChunk(model, { tool_calls: [{ index: Math.max(toolIndex, 0), function: { arguments: data.delta.partial_json } }] }, null);
      }
      continue;
    }
    if (data.type === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      continue;
    }
    if (data.type === 'message_stop') break;
  }
  yield openAIChunk(model, {}, finishReasonOf(stopReason));
  yield openAIChunk(model, {}, null, {
    prompt_tokens: inputTokens,
    completion_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
  });
  yield 'data: [DONE]\n\n';
}

// ---------------------------------------------------------------------------
// pivot -> Gemini (the request a gemini-wire provider receives)
// ---------------------------------------------------------------------------

/** One OpenAI content part as a Gemini part: text, inline image data, or a function response. */
function geminiPartOf(part: unknown, toolNames: Map<string, string>): Json | null {
  if (typeof part === 'string') return { text: part };
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const data = /^data:([^;,]+);base64,(.*)$/.exec(part.image_url.url);
    if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
    return { fileData: { mimeType: 'image/png', fileUri: part.image_url.url } };
  }
  void toolNames;
  return null;
}

/**
 * A Chat Completions request as a Gemini generateContent request: system messages become the system
 * instruction, assistant turns become model turns (tool calls become functionCall parts), tool results
 * become functionResponse parts in a user turn, and tools, the output cap and stop sequences are
 * rewritten in Gemini's spellings.
 */
export function openAIRequestToGemini(body: Json, model: string): Json {
  void model;
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      const name = (typeof raw.tool_call_id === 'string' && toolNames.get(raw.tool_call_id)) || 'tool';
      let response: unknown = raw.content;
      if (typeof response === 'string') {
        try {
          response = JSON.parse(response);
        } catch {
          response = { result: response };
        }
      }
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: isRecord(response) ? response : { result: response } } }] });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: unknown = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            args = JSON.parse(call.function.arguments);
          } catch {
            args = {};
          }
        }
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (typeof call.id === 'string') toolNames.set(call.id, name);
        parts.push({ functionCall: { name, args: isRecord(args) ? args : {} } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (typeof raw.content === 'string') {
      contents.push({ role: 'user', parts: [{ text: raw.content }] });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const parts = raw.content.flatMap((part) => {
        const converted = geminiPartOf(part, toolNames);
        return converted === null ? [] : [converted];
      });
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const generation: Json = {};
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (cap !== null) generation.maxOutputTokens = cap;
  if (typeof body.temperature === 'number') generation.temperature = body.temperature;
  if (typeof body.top_p === 'number') generation.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) generation.stopSequences = body.stop;
  if (typeof body.stop === 'string') generation.stopSequences = [body.stop];
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    generation.responseMimeType = 'application/json';
  }
  if (Object.keys(generation).length > 0) out.generationConfig = generation;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const choice = body.tool_choice;
  if (choice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Gemini -> pivot (the answer a gemini-wire provider gives)
// ---------------------------------------------------------------------------

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiFinishOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'PROHIBITED_CONTENT' || reason === 'BLOCKLIST') return 'content_filter';
  return 'stop';
}

/** The text and tool calls of one Gemini candidate's parts, in order. */
function geminiParts(candidate: Json): { text: string; toolCalls: Json[] } {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  return { text: texts.join(''), toolCalls };
}

function firstCandidate(body: Json): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  return isRecord(candidates[0]) ? candidates[0] : {};
}

/** A Gemini generateContent answer as a Chat Completions answer. */
export function geminiResponseToOpenAI(body: Json, model: string): Json {
  const candidate = firstCandidate(body);
  const { text, toolCalls } = geminiParts(candidate);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const finish = toolCalls.length > 0 && geminiFinishOf(candidate.finishReason) === 'stop' ? 'tool_calls' : geminiFinishOf(candidate.finishReason);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text !== '' ? text : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: finish,
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/**
 * A Gemini streamed answer (SSE, one generateContent JSON a frame) as Chat Completions SSE chunks:
 * each frame's text becomes a content delta and its function calls become tool_calls deltas; the frame
 * carrying a finishReason closes the stream, followed by the usage chunk and [DONE].
 */
export async function* geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let promptTokens = 0;
  let completionTokens = 0;
  let totalTokens = 0;
  let finish: string | null = null;
  let toolIndex = -1;
  yield openAIChunk(model, { role: 'assistant', content: '' }, null);
  for await (const frame of sseFrames(upstream)) {
    const data = sseData(frame);
    if (data === null) continue;
    const candidate = firstCandidate(data);
    const { text, toolCalls } = geminiParts(candidate);
    if (text !== '') yield openAIChunk(model, { content: text }, null);
    for (const call of toolCalls) {
      toolIndex += 1;
      const fn = isRecord(call.function) ? call.function : {};
      yield openAIChunk(
        model,
        { tool_calls: [{ index: toolIndex, id: typeof call.id === 'string' ? call.id : `call_${toolIndex}`, type: 'function', function: { name: fn.name ?? '', arguments: fn.arguments ?? '' } }] },
        null,
      );
    }
    if (typeof candidate.finishReason === 'string') {
      finish = toolCalls.length > 0 && geminiFinishOf(candidate.finishReason) === 'stop' ? 'tool_calls' : geminiFinishOf(candidate.finishReason);
    }
    if (isRecord(data.usageMetadata)) {
      promptTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      completionTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
      totalTokens = numberOrZero(data.usageMetadata.totalTokenCount);
    }
  }
  yield openAIChunk(model, {}, finish ?? 'stop');
  yield openAIChunk(model, {}, null, { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: totalTokens });
  yield 'data: [DONE]\n\n';
}

/** Wrap an async generator of strings as a ReadableStream of bytes, for chaining translators. */
export function streamFrom(source: AsyncGenerator<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const next = await source.next();
      if (next.done === true) controller.close();
      else controller.enqueue(encoder.encode(next.value));
    },
    async cancel() {
      await source.return(undefined);
    },
  });
}
