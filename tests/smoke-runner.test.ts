import { afterEach, describe, expect, it, vi } from "vitest";
import { parseSmokeModels, runSmoke, runSmokeFromEnv } from "../scripts/smoke-runner.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("parseSmokeModels", () => {
  it("parses comma and whitespace separated model ids", () => {
    expect(parseSmokeModels(" github-gpt-4.1-mini,openai-gpt-5.4-nano\nanthropic-claude-haiku ")).toEqual([
      "github-gpt-4.1-mini",
      "openai-gpt-5.4-nano",
      "anthropic-claude-haiku",
    ]);
  });
});

describe("runSmoke", () => {
  it("discovers models and sends a chat completion request to each requested model", async () => {
    const requests: Array<{ url: string; body?: unknown }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requests.push({
        url,
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
      });
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "github-gpt-4.1-mini", model_info: { mode: "chat" } },
            { model_name: "gemini-flash", model_info: { mode: "chat" } },
          ],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, {
          choices: [{ message: { content: "pong" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runSmoke({
      baseUrl: "http://127.0.0.1:4000/v1",
      apiKey: "sk-smoke",
      modelIds: ["github-gpt-4.1-mini", "gemini-flash"],
      timeoutMs: 1000,
    });

    expect(result).toEqual({
      source: "model_info",
      discoveredCount: 2,
      completions: [
        { modelId: "github-gpt-4.1-mini", content: "pong" },
        { modelId: "gemini-flash", content: "pong" },
      ],
    });
    expect(requests.filter((request) => request.url.endsWith("/v1/chat/completions"))).toMatchObject([
      {
        url: "http://127.0.0.1:4000/v1/chat/completions",
        body: {
          model: "github-gpt-4.1-mini",
          messages: [{ role: "user", content: "Reply with one short word." }],
          max_tokens: 16,
          temperature: 0,
        },
      },
      {
        url: "http://127.0.0.1:4000/v1/chat/completions",
        body: {
          model: "gemini-flash",
          messages: [{ role: "user", content: "Reply with one short word." }],
          max_tokens: 16,
          temperature: 0,
        },
      },
    ]);
  });

  it("fails before completion calls when a requested model is not discovered", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-gpt-4.1-mini", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["anthropic-claude-haiku"],
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/Requested smoke models were not discovered: anthropic-claude-haiku/);
  });

  it("includes provider response bodies in chat completion failures", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(429, { error: "rate limited" });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["github-models-openai"],
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/\/v1\/chat\/completions for github-models-openai returned 429.*rate limited/);
  });
});

describe("runSmokeFromEnv", () => {
  it("loads LiteLLM smoke settings from the environment", async () => {
    const requests: Array<{ url: string; headers?: Record<string, string> }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requests.push({
        url,
        headers: init?.headers as Record<string, string>,
      });
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, {
          choices: [{ message: { content: "pong" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runSmokeFromEnv({
      LITELLM_BASE_URL: " http://127.0.0.1:4000/v1 ",
      LITELLM_API_KEY: " sk-env ",
      LITELLM_SMOKE_MODELS: "github-models-openai",
      LITELLM_SMOKE_TIMEOUT_MS: "1000",
    });

    expect(result.completions).toEqual([{ modelId: "github-models-openai", content: "pong" }]);
    expect(requests[0]).toMatchObject({
      url: "http://127.0.0.1:4000/model/info",
      headers: { Authorization: "Bearer sk-env" },
    });
  });

  it("requires LiteLLM base URL and API key settings", async () => {
    await expect(runSmokeFromEnv({ LITELLM_BASE_URL: "http://127.0.0.1:4000" })).rejects.toThrow(
      /LITELLM_BASE_URL and LITELLM_API_KEY must be set/,
    );
  });
});
