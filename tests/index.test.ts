import { readFileSync, writeFileSync } from "node:fs";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fingerprint } from "../src/cache.js";

const ENV_KEYS = [
  "LITELLM_BASE_URL",
  "LITELLM_API_KEY",
  "LITELLM_API_KEY_HELPER",
  "LITELLM_DISCOVERY_TIMEOUT_MS",
  "STORED_LITELLM_KEY",
];
const ORIGINAL_ENV = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));

type TestProviderConfig = {
  baseUrl?: string;
  models?: unknown[];
  oauth?: {
    login: (callbacks: {
      onPrompt: (options: { message: string; placeholder?: string }) => Promise<string>;
      onProgress?: (message: string) => void;
      signal?: AbortSignal;
    }) => Promise<{ access: string; refresh: string; expires: number; baseUrl?: string }>;
    refreshToken: (credential: { access: string; refresh: string; expires: number; baseUrl?: string }) => Promise<{
      access: string;
      refresh: string;
      expires: number;
      baseUrl?: string;
    }>;
    getApiKey: (credential: { access: string; refresh: string; expires: number; baseUrl?: string }) => string;
  };
};

type TestCommand = {
  description: string;
  handler: (args: string[], ctx: { ui: { notify: (message: string, type: string) => void } }) => Promise<void> | void;
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function makeAgentDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "pi-litellm-index-"));
}

function makeJwt(expSeconds: number): string {
  const encode = (value: unknown): string => Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode({ exp: expSeconds })}.sig`;
}

async function writeHelper(
  agentDir: string,
  tokens: string[],
  helperPath = join(agentDir, "litellm-token-helper.sh"),
): Promise<string> {
  await writeFile(
    helperPath,
    `#!/usr/bin/env bash\ncount_file="${join(agentDir, "helper-count")}"\ncount=0\n[ -f "$count_file" ] && count=$(cat "$count_file")\ncase "$count" in\n${tokens.map((token, index) => `  ${index}) printf %s '${token}' ;;`).join("\n")}\n  *) printf %s '${tokens.at(-1)}' ;;\nesac\necho $((count + 1)) > "$count_file"\n`,
    { mode: 0o700 },
  );
  return helperPath;
}

async function readHelperCount(agentDir: string): Promise<number> {
  try {
    return Number(await readFile(join(agentDir, "helper-count"), "utf8"));
  } catch {
    return 0;
  }
}

const cachedModels = [{ id: "cached-model", name: "cached-model", provider: "litellm" }];

async function writeModelCache(agentDir: string, helperPath: string): Promise<void> {
  await writeFile(
    join(agentDir, "litellm-models.json"),
    JSON.stringify({
      baseUrl: "https://litellm.example.com",
      apiKeyFingerprint: fingerprint(`!${helperPath}`),
      fetchedAt: Date.now(),
      source: "model_info",
      models: cachedModels,
    }),
    "utf8",
  );
}

async function loadExtension(agentDir: string): Promise<(pi: TestPi) => Promise<void>> {
  vi.resetModules();
  vi.doMock("@earendil-works/pi-coding-agent", () => {
    type StoredEntry =
      | { type: "api_key"; key: string }
      | { type: "oauth"; access: string; expires: number; refresh: string; baseUrl?: string };

    class TestAuthStorage {
      constructor(private readonly authPath: string) {}

      static create(authPath: string): TestAuthStorage {
        return new TestAuthStorage(authPath);
      }

      private readData(): Record<string, StoredEntry> {
        try {
          return JSON.parse(readFileSync(this.authPath, "utf8")) as Record<string, StoredEntry>;
        } catch {
          return {};
        }
      }

      set(provider: string, credential: StoredEntry): void {
        const data = this.readData();
        data[provider] = credential;
        writeFileSync(this.authPath, JSON.stringify(data), "utf8");
      }

      async getApiKey(provider: string): Promise<string | undefined> {
        const entry = this.readData()[provider];
        if (entry?.type === "api_key") return process.env[entry.key] || entry.key;
        if (entry?.type === "oauth") return entry.access;
        return undefined;
      }
    }

    return { AuthStorage: TestAuthStorage, getAgentDir: () => agentDir };
  });
  const mod = await import("../src/index.js");
  return mod.default as unknown as (pi: TestPi) => Promise<void>;
}

function createPi(): TestPi {
  return {
    providers: [],
    commands: new Map(),
    handlers: new Map(),
    registerProvider(name: string, config: TestProviderConfig) {
      this.providers.push({ name, config });
    },
    registerCommand(name: string, command: TestCommand) {
      this.commands.set(name, command);
    },
    on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any) {
      this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
    },
  };
}

type TestPi = {
  providers: Array<{ name: string; config: TestProviderConfig }>;
  commands: Map<string, TestCommand>;
  handlers: Map<string, Array<(event: any, ctx?: any) => Promise<any> | any>>;
  registerProvider(name: string, config: TestProviderConfig): void;
  registerCommand(name: string, command: TestCommand): void;
  on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any): void;
};

afterEach(() => {
  for (const key of ENV_KEYS) {
    const original = ORIGINAL_ENV.get(key);
    if (original === undefined) process.env[key] = undefined;
    else process.env[key] = original;
  }
  vi.restoreAllMocks();
  vi.resetModules();
  vi.unmock("@earendil-works/pi-coding-agent");
});

describe("extension startup", () => {
  it("discovers with the resolved stored auth key before LITELLM_API_KEY", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({ litellm: { type: "api_key", key: "STORED_LITELLM_KEY" } }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.STORED_LITELLM_KEY = "stored-key";
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
      return jsonResponse(200, {
        data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
      });
    });

    const extension = await loadExtension(agentDir);
    await extension(createPi());

    expect(seenAuthHeaders).toEqual(["Bearer stored-key"]);
  });

  it("does not fetch on refresh when discovery timeout is zero", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
      }),
    );
    const notify = vi.fn();

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await pi.commands.get("litellm-refresh")?.handler([], { ui: { notify } });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(notify).toHaveBeenCalledWith("LiteLLM refresh disabled (LITELLM_DISCOVERY_TIMEOUT_MS=0)", "warning");
  });

  it("prompts during login and caches discovered models", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const seenRequests: Array<{ url: string; authorization: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({
        url,
        authorization: new Headers(init?.headers).get("authorization") ?? "",
      });
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "vidaimock-openai", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const promptMessages: string[] = [];
    const progress = vi.fn();
    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => {
        promptMessages.push(options.message);
        return promptMessages.length === 1 ? " http://127.0.0.1:4000/v1 " : " sk-login ";
      },
      onProgress: progress,
      signal: new AbortController().signal,
    });

    expect(promptMessages).toEqual(["Enter LiteLLM proxy URL (no trailing /v1):", "Enter API key:"]);
    expect(seenRequests).toEqual([{ url: "http://127.0.0.1:4000/model/info", authorization: "Bearer sk-login" }]);
    expect(credential).toMatchObject({
      access: "sk-login",
      refresh: "",
      baseUrl: "http://127.0.0.1:4000",
    });
    const cache = JSON.parse(await readFile(join(agentDir, "litellm-models.json"), "utf8")) as {
      models: Array<{ id: string }>;
    };
    expect(cache.models.map((model) => model.id)).toEqual(["vidaimock-openai"]);
    expect(progress).toHaveBeenCalledWith("LiteLLM: 1 models discovered (source: model_info)");
  });

  it("uses the login cache timestamp for later stale auto-refresh", async () => {
    const agentDir = await makeAgentDir();
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    delete process.env.LITELLM_DISCOVERY_TIMEOUT_MS;
    const loginTime = new Date("2026-05-01T00:00:00.000Z").getTime();
    vi.spyOn(Date, "now").mockReturnValue(loginTime);

    let callCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        callCount++;
        return jsonResponse(200, {
          data: [{ model_name: `vidaimock-openai-${callCount}`, model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    expect(callCount).toBe(0);

    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => (options.placeholder ? " http://127.0.0.1:4000/v1 " : " sk-login "),
      signal: new AbortController().signal,
    });
    expect(callCount).toBe(1);
    await writeFile(join(agentDir, "auth.json"), JSON.stringify({ litellm: { type: "oauth", ...credential } }), "utf8");

    vi.mocked(Date.now).mockReturnValue(loginTime + 25 * 60 * 60 * 1000);
    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler({ reason: "start" }, { sessionManager: { getSessionFile: () => undefined } });
    }

    await vi.waitFor(() => {
      expect(callCount).toBe(2);
      expect(pi.providers).toHaveLength(2);
    });
  });

  it("routes LITELLM_API_KEY_HELPER auth through the uncached OAuth hooks", async () => {
    const agentDir = await makeAgentDir();
    const first = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const second = makeJwt(Math.floor(Date.now() / 1000) + 7200);
    const helperPath = await writeHelper(agentDir, [first, second]);
    process.env.LITELLM_BASE_URL = "https://litellm.example.com/v1";
    process.env.LITELLM_API_KEY = "stale-token";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
      return jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] });
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    // Startup discovery still resolves a fresh helper token (one helper invocation).
    expect(seenAuthHeaders).toEqual([`Bearer ${first}`]);
    expect(await readHelperCount(agentDir)).toBe(1);

    // The provider key is never the `!helper` command (Pi caches `!command` keys for the process
    // lifetime); the literal env var is used as the static fallback instead.
    expect(pi.providers[0]?.config).toMatchObject({
      baseUrl: "https://litellm.example.com/v1",
      apiKey: "LITELLM_API_KEY",
    });

    // A command-backed OAuth credential is seeded so request-time auth refreshes via the helper.
    const auth = JSON.parse(await readFile(join(agentDir, "auth.json"), "utf8")) as {
      litellm: { type: string; access: string; refresh: string; expires: number; baseUrl?: string };
    };
    expect(auth.litellm).toMatchObject({ type: "oauth", refresh: `!${helperPath}`, expires: 0 });

    // The OAuth getApiKey hook re-runs the helper for an expired/opaque token (uncached).
    const refreshed = pi.providers[0]?.config.oauth?.getApiKey({
      access: "expired-token",
      refresh: `!${helperPath}`,
      expires: 0,
    });
    expect(refreshed).toBe(second);
    expect(await readHelperCount(agentDir)).toBe(2);
  });

  it("does not overwrite an existing /login OAuth credential with the env helper", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["helper-token"]);
    const stored = {
      type: "oauth",
      access: "logged-in-token",
      refresh: "real-refresh-token",
      expires: Date.now() + 3_600_000,
      baseUrl: "https://litellm.example.com",
    };
    await writeFile(join(agentDir, "auth.json"), JSON.stringify({ litellm: stored }), "utf8");
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    await extension(createPi());

    const auth = JSON.parse(await readFile(join(agentDir, "auth.json"), "utf8")) as { litellm: typeof stored };
    expect(auth.litellm).toEqual(stored);
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it.each([
    { name: "discovery is disabled", timeout: "0", fetches: false, helperRuns: 0 },
    { name: "fresh cache avoids discovery", fetches: false, helperRuns: 0 },
    { name: "helper output rotates before discovery fails", listModels: true, fetches: true, helperRuns: 1 },
  ])("uses cached helper-backed models when $name", async ({ timeout, listModels, fetches, helperRuns }) => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["rotated-token"]);
    const originalArgv = process.argv;
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    if (timeout) process.env.LITELLM_DISCOVERY_TIMEOUT_MS = timeout;
    if (listModels) process.argv = [...process.argv, "--list-models"];
    await writeModelCache(agentDir, helperPath);
    const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("temporary outage"));

    try {
      const extension = await loadExtension(agentDir);
      const pi = createPi();
      await extension(pi);
      expect(fetchMock).toHaveBeenCalledTimes(fetches ? 1 : 0);
      expect(await readHelperCount(agentDir)).toBe(helperRuns);
      expect(pi.providers[0]?.config.models).toEqual(cachedModels);
    } finally {
      process.argv = originalArgv;
    }
  });

  it("refreshes command-backed login credentials", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const now = new Date("2026-05-29T21:00:00.000Z").getTime();
    const first = makeJwt(Math.floor(now / 1000) + 60);
    const second = makeJwt(Math.floor(now / 1000) + 3600);
    const helperPath = await writeHelper(agentDir, [first, second, "opaque-token"]);
    vi.spyOn(Date, "now").mockReturnValue(now);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] }),
    );

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => (options.placeholder ? "https://litellm.example.com" : `!${helperPath}`),
    });
    const refreshed = await pi.providers[0]?.config.oauth?.refreshToken(credential!);
    const opaque = pi.providers[0]?.config.oauth?.getApiKey({ ...credential!, access: "opaque-old" });

    expect(credential?.access).toBe(first);
    expect(credential?.refresh).toBe(`!${helperPath}`);
    expect(credential?.expires).toBeLessThan((Math.floor(now / 1000) + 60) * 1000);
    expect(refreshed?.access).toBe(second);
    expect(opaque).toBe("opaque-token");
    expect(await readHelperCount(agentDir)).toBe(3);
  });

  it("uses a helper when stored OAuth credentials contain an expired token", async () => {
    const agentDir = await makeAgentDir();
    const now = new Date("2026-05-29T21:00:00.000Z").getTime();
    const expired = makeJwt(Math.floor(now / 1000) - 60);
    const fresh = makeJwt(Math.floor(now / 1000) + 3600);
    const helperPath = await writeHelper(agentDir, [fresh]);
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: {
          type: "oauth",
          access: expired,
          refresh: `!${helperPath}`,
          expires: Number.MAX_SAFE_INTEGER,
          baseUrl: "https://litellm.example.com",
        },
      }),
      "utf8",
    );
    vi.spyOn(Date, "now").mockReturnValue(now);
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
      return jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] });
    });

    const extension = await loadExtension(agentDir);
    await extension(createPi());

    expect(seenAuthHeaders).toEqual([`Bearer ${fresh}`]);
  });
});
