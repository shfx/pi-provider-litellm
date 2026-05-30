import { execSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { Api, AssistantMessage, Model, OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";
import type { ExtensionAPI, OAuthCredential, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { AuthStorage, getAgentDir } from "@earendil-works/pi-coding-agent";
import { fingerprint, readCache, writeCache } from "./cache.js";
import { setupLiteLLMCostTracking } from "./cost.js";
import { discoverModels, normalizeBaseUrl, shouldSuppressReasoningContent } from "./discover.js";
import { getSessionIdFromFile } from "./litellm.js";
import type { AuthFileEntry, CacheFile, DiscoveryOptions, DiscoveryResult, ResolvedCredentials } from "./types.js";

const PROVIDER_NAME = "litellm";
const ENV_BASE_URL = "LITELLM_BASE_URL";
const ENV_API_KEY = "LITELLM_API_KEY";
const ENV_API_KEY_HELPER = "LITELLM_API_KEY_HELPER";
const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
const ENV_OFFLINE = "LITELLM_OFFLINE";
const DEFAULT_TIMEOUT_MS = 5000;
const LOGIN_TIMEOUT_MS = 10_000;
const CACHE_FILENAME = "litellm-models.json";
const CACHE_STALE_MS = 24 * 60 * 60 * 1000;
const TOKEN_REFRESH_LEAD_MS = 5 * 60 * 1000;

type RefreshResult = { models: ProviderModelConfig[]; source: string };

function getAuthPath(): string {
  return join(getAgentDir(), "auth.json");
}

function getCachePath(): string {
  return join(getAgentDir(), CACHE_FILENAME);
}

async function readAuthEntry(): Promise<AuthFileEntry | undefined> {
  try {
    const raw = await readFile(getAuthPath(), "utf8");
    const parsed = JSON.parse(raw) as Record<string, AuthFileEntry>;
    return parsed?.[PROVIDER_NAME];
  } catch {
    return undefined;
  }
}

/**
 * When `LITELLM_API_KEY_HELPER` is the only credential source we must not hand the `!helper`
 * command to Pi's static provider API-key resolver: `!command` keys are documented as "cached for
 * process lifetime", so a long-running session would keep sending the first helper-issued bearer
 * token after it expires. Instead, seed a command-backed OAuth credential so every request resolves
 * through the uncached OAuth `getApiKey`/`refreshToken` hooks, which re-run the helper on demand.
 *
 * The seeded credential starts expired (`access: ""`, `expires: 0`) so no helper is executed at
 * startup (preserving the offline / disabled-discovery path); the first actual request triggers an
 * OAuth refresh that runs the helper and stores a fresh token.
 */
async function seedHelperOAuthCredential(baseUrl: string | undefined): Promise<void> {
  const helperCommand = getApiKeyHelperCommand();
  if (!helperCommand) return;

  const existing = await readAuthEntry();
  if (existing?.type === "oauth" && existing.refresh === helperCommand) {
    // A command-backed OAuth credential is already in place (seeded earlier or via `/login`); leave
    // its current token/expiry untouched so we do not discard a freshly refreshed token or rewrite
    // auth.json on every startup.
    return;
  }
  if (existing) {
    // Respect an explicit `/login litellm` or stored api_key entry that uses different credentials.
    return;
  }

  const credential: OAuthCredential = {
    type: "oauth",
    access: "",
    refresh: helperCommand,
    expires: 0,
    ...(baseUrl ? { baseUrl } : {}),
  };
  AuthStorage.create(getAuthPath()).set(PROVIDER_NAME, credential);
}

function cleanConfig(raw: string | undefined): string | undefined {
  const trimmed = raw?.trim();
  return trimmed && trimmed !== "undefined" ? trimmed : undefined;
}

function normalizeCommand(raw: string | undefined): string | undefined {
  const trimmed = cleanConfig(raw);
  if (!trimmed) return undefined;
  return trimmed.startsWith("!") ? trimmed : `!${trimmed}`;
}

function getApiKeyHelperCommand(): string | undefined {
  return normalizeCommand(process.env[ENV_API_KEY_HELPER]);
}

function executeApiKeyCommand(commandConfig: string): string {
  const command = commandConfig.startsWith("!") ? commandConfig.slice(1) : commandConfig;
  const output = execSync(command, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 10_000 }).trim();
  if (!output) throw new Error(`LiteLLM API key helper produced no output: ${command}`);
  return output;
}

function tokenExpiresAt(apiKey: string): number {
  const [, payload] = apiKey.split(".");
  if (!payload) return Number.MAX_SAFE_INTEGER;
  try {
    const claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as { exp?: unknown };
    return typeof claims.exp === "number"
      ? Math.max(Date.now(), claims.exp * 1000 - TOKEN_REFRESH_LEAD_MS)
      : Number.MAX_SAFE_INTEGER;
  } catch {
    return Number.MAX_SAFE_INTEGER;
  }
}

function shouldRefreshToken(apiKey: string): boolean {
  return tokenExpiresAt(apiKey) <= Date.now();
}

function getLiteLLMApiKey(credentials: OAuthCredentials): string {
  if (credentials.refresh.startsWith("!")) return executeApiKeyCommand(credentials.refresh);
  const command = shouldRefreshToken(credentials.access) ? getApiKeyHelperCommand() : undefined;
  return command ? executeApiKeyCommand(command) : credentials.access;
}

async function resolveCredentials({ executeHelpers = true } = {}): Promise<ResolvedCredentials> {
  const entry = await readAuthEntry();
  const envBase = cleanConfig(process.env[ENV_BASE_URL]);
  const envKey = cleanConfig(process.env[ENV_API_KEY]);
  const envHelperCommand = getApiKeyHelperCommand();
  const authBase = entry?.type === "oauth" ? entry.baseUrl?.trim() : undefined;
  const authKey =
    entry?.type === "oauth"
      ? (executeHelpers ? getLiteLLMApiKey(entry) : entry.access).trim()
      : entry?.type === "api_key"
        ? (await AuthStorage.create(getAuthPath()).getApiKey(PROVIDER_NAME, { includeFallback: false }))?.trim()
        : undefined;
  const apiKey =
    authKey || (executeHelpers && envHelperCommand ? executeApiKeyCommand(envHelperCommand) : undefined) || envKey;
  const apiKeyFingerprint =
    entry?.type === "oauth" && entry.refresh.startsWith("!")
      ? fingerprint(entry.refresh)
      : authKey
        ? fingerprint(authKey)
        : envHelperCommand
          ? fingerprint(envHelperCommand)
          : envKey
            ? fingerprint(envKey)
            : undefined;
  const rawBase = authBase || envBase;
  return {
    baseUrl: rawBase ? normalizeBaseUrl(rawBase) : undefined,
    apiKey: apiKey || undefined,
    apiKeyFingerprint,
  };
}

function getDiscoveryTimeoutMs(): number {
  const raw = process.env[ENV_TIMEOUT];
  if (raw === undefined) return DEFAULT_TIMEOUT_MS;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < 0) return DEFAULT_TIMEOUT_MS;
  return parsed;
}

function isOffline(): boolean {
  return process.env[ENV_OFFLINE] === "1";
}

function isListModelsMode(): boolean {
  return process.argv.includes("--list-models");
}

async function discoverWithFallback(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions,
): Promise<{ result: DiscoveryResult; warning?: string }> {
  try {
    return { result: await discoverModels(baseUrl, apiKey, options) };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      result: { models: [] as ProviderModelConfig[], source: "models_list" },
      warning: message,
    };
  }
}

async function loginLiteLLM(
  callbacks: OAuthLoginCallbacks,
  onCacheWrite?: (cache: CacheFile) => void,
): Promise<OAuthCredentials> {
  const rawBaseUrl = (
    await callbacks.onPrompt({
      message: "Enter LiteLLM proxy URL (no trailing /v1):",
      placeholder: "https://litellm.example.com",
    })
  ).trim();
  const apiKeyInput = (await callbacks.onPrompt({ message: "Enter API key:" })).trim();
  if (!rawBaseUrl || !apiKeyInput) throw new Error("Both base URL and API key are required");

  const baseUrl = normalizeBaseUrl(rawBaseUrl);
  const refresh = apiKeyInput.startsWith("!") ? apiKeyInput : "";
  const apiKey = refresh ? executeApiKeyCommand(refresh) : apiKeyInput;
  const { models, source } = await discoverModels(baseUrl, apiKey, {
    timeoutMs: LOGIN_TIMEOUT_MS,
    signal: callbacks.signal,
  });

  const cache: CacheFile = {
    baseUrl,
    apiKeyFingerprint: fingerprint(refresh || apiKey),
    fetchedAt: Date.now(),
    source,
    models,
  };
  await writeCache(getCachePath(), cache);
  onCacheWrite?.(cache);
  callbacks.onProgress?.(`LiteLLM: ${models.length} models discovered (source: ${source})`);

  return {
    access: apiKey,
    refresh,
    expires: tokenExpiresAt(apiKey),
    baseUrl,
  } as OAuthCredentials & { baseUrl: string };
}

async function refreshLiteLLM(credentials: OAuthCredentials): Promise<OAuthCredentials> {
  if (!credentials.refresh.startsWith("!")) return credentials;
  const access = executeApiKeyCommand(credentials.refresh);
  return { ...credentials, access, expires: tokenExpiresAt(access) };
}

function modifyLiteLLMModels(models: Model<Api>[], cred: OAuthCredentials): Model<Api>[] {
  const baseUrl = (cred as { baseUrl?: string }).baseUrl;
  if (!baseUrl) return models;
  return models.map((m) => (m.provider === PROVIDER_NAME ? { ...m, baseUrl: `${baseUrl}/v1` } : m));
}

function prepareLiteLLMRequestPayload(
  payload: Record<string, unknown>,
  modelId: string | undefined,
  sessionId: string | undefined,
): Record<string, unknown> | undefined {
  let next: Record<string, unknown> | undefined;
  const update = (key: string, value: unknown): void => {
    if (payload[key] !== undefined) return;
    next ??= { ...payload };
    next[key] = value;
  };

  if (modelId && shouldSuppressReasoningContent(modelId)) {
    update("include_reasoning", false);
    update("reasoning_content", false);
    update("merge_reasoning_content_in_choices", true);
    update("thinking", { type: "disabled" });
  }

  if (sessionId) {
    next ??= { ...payload };
    next.litellm_session_id = sessionId;
  }

  return next;
}

function normalizeThinkTags(message: AssistantMessage): AssistantMessage | undefined {
  if (message.provider !== PROVIDER_NAME || !shouldSuppressReasoningContent(message.model)) return;

  let changed = false;
  const content: AssistantMessage["content"] = [];
  const appendText = (text: string): void => {
    if (!text) return;
    const last = content.at(-1);
    if (last?.type === "text") {
      last.text += text;
      return;
    }
    content.push({ type: "text", text });
  };
  const appendThinking = (thinking: string): void => {
    if (!thinking) return;
    const last = content.at(-1);
    if (last?.type === "thinking") {
      last.thinking += thinking;
      return;
    }
    content.push({ type: "thinking", thinking });
  };

  for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
    const block = message.content[blockIndex];
    if (block.type !== "text") {
      content.push(block);
      continue;
    }

    let index = 0;
    while (index < block.text.length) {
      const start = block.text.indexOf("<think>", index);
      if (start === -1) {
        appendText(block.text.slice(index));
        break;
      }

      changed = true;
      appendText(block.text.slice(index, start));
      const thinkingStart = start + "<think>".length;
      const end = block.text.indexOf("</think>", thinkingStart);
      if (end === -1) {
        const isBeforeNonTextContent = message.content
          .slice(blockIndex + 1)
          .some((nextBlock) => nextBlock.type !== "text");
        if (isBeforeNonTextContent) appendThinking(block.text.slice(thinkingStart));
        else appendText(block.text.slice(thinkingStart));
        index = block.text.length;
        break;
      }

      appendThinking(block.text.slice(thinkingStart, end));
      index = end + "</think>".length;
    }
  }

  if (!changed) return;
  return { ...message, content };
}

export default async function (pi: ExtensionAPI): Promise<void> {
  let creds = await resolveCredentials({ executeHelpers: false });
  const cache = await readCache(getCachePath());
  let fp = creds.apiKeyFingerprint;
  let cacheFetchedAt = cache?.fetchedAt ?? 0;

  const cacheValid =
    cache !== null &&
    creds.baseUrl !== undefined &&
    fp !== undefined &&
    cache.baseUrl === creds.baseUrl &&
    cache.apiKeyFingerprint === fp;

  let models: ProviderModelConfig[] = cacheValid && cache ? cache.models : [];
  const shouldFetch =
    creds.baseUrl !== undefined &&
    fp !== undefined &&
    !isOffline() &&
    getDiscoveryTimeoutMs() > 0 &&
    (!cacheValid || isListModelsMode());

  if (shouldFetch) {
    creds = await resolveCredentials();
    fp = creds.apiKeyFingerprint;
  }

  if (shouldFetch && creds.baseUrl && creds.apiKey && fp) {
    const timeoutMs = getDiscoveryTimeoutMs();
    const { result, warning } = await discoverWithFallback(creds.baseUrl, creds.apiKey, {
      timeoutMs,
    });
    if (warning) {
      if (cacheValid && cache) {
        process.stderr.write(`LiteLLM: discovery failed (${warning}); using cached models.\n`);
        models = cache.models;
      } else {
        process.stderr.write(`LiteLLM: discovery failed (${warning}); registering provider with no models.\n`);
        models = [];
      }
    } else {
      models = result.models;
      const next: CacheFile = {
        baseUrl: creds.baseUrl,
        apiKeyFingerprint: fp,
        fetchedAt: Date.now(),
        source: result.source,
        models: result.models,
      };
      await writeCache(getCachePath(), next);
      cacheFetchedAt = next.fetchedAt;
      if (isListModelsMode()) {
        process.stderr.write(`LiteLLM: ${result.models.length} models discovered (source: ${result.source}).\n`);
      }
    }
  }

  const oauth = {
    name: "LiteLLM",
    login: (callbacks: OAuthLoginCallbacks) =>
      loginLiteLLM(callbacks, (next) => {
        cacheFetchedAt = next.fetchedAt;
      }),
    refreshToken: refreshLiteLLM,
    getApiKey: getLiteLLMApiKey,
    modifyModels: modifyLiteLLMModels,
  };

  // Route env-helper requests through the uncached OAuth hooks instead of the static `!command`
  // provider key (which Pi caches for the process lifetime). Seeding is idempotent and starts the
  // credential expired so the helper is only executed when a request actually needs a token.
  await seedHelperOAuthCredential(creds.baseUrl);

  function registerProvider(baseUrl: string | undefined, models: ProviderModelConfig[]): void {
    pi.registerProvider(PROVIDER_NAME, {
      baseUrl: baseUrl ? `${baseUrl}/v1` : "https://litellm.example.com/v1",
      // The helper command is never handed to Pi's static provider key resolver (it would be
      // cached for the process lifetime). When a helper is configured it is resolved per request
      // through the OAuth credential seeded above; otherwise we expose the literal
      // `LITELLM_API_KEY` env var.
      apiKey: ENV_API_KEY,
      api: "openai-completions",
      models,
      oauth,
    });
  }

  registerProvider(creds.baseUrl, models);

  const updateCosts = setupLiteLLMCostTracking(pi, models);

  let refreshInProgress: Promise<RefreshResult> | null = null;

  function discoveryDisabledReason(): string | null {
    if (isOffline()) return `${ENV_OFFLINE}=1`;
    if (getDiscoveryTimeoutMs() === 0) return `${ENV_TIMEOUT}=0`;
    return null;
  }

  async function refreshModelsAndCosts(): Promise<RefreshResult> {
    const fresh = await resolveCredentials();
    const freshFp = fresh.apiKeyFingerprint;
    if (!fresh.baseUrl || !fresh.apiKey || !freshFp) {
      throw new Error("no credentials. Run /login litellm or set env vars.");
    }
    const result = await discoverModels(fresh.baseUrl, fresh.apiKey, { timeoutMs: getDiscoveryTimeoutMs() });
    const now = Date.now();
    await writeCache(getCachePath(), {
      baseUrl: fresh.baseUrl,
      apiKeyFingerprint: freshFp,
      fetchedAt: now,
      source: result.source,
      models: result.models,
    });
    registerProvider(fresh.baseUrl, result.models);
    updateCosts(result.models);
    cacheFetchedAt = now;
    return { models: result.models, source: result.source };
  }

  function runRefresh(): Promise<RefreshResult> {
    refreshInProgress ??= refreshModelsAndCosts().finally(() => {
      refreshInProgress = null;
    });
    return refreshInProgress;
  }

  pi.registerCommand("litellm-refresh", {
    description: "Re-discover models from the LiteLLM proxy.",
    handler: async (_args, ctx) => {
      const disabledReason = discoveryDisabledReason();
      if (disabledReason) {
        ctx.ui.notify(`LiteLLM refresh disabled (${disabledReason})`, "warning");
        return;
      }
      try {
        const result = await runRefresh();
        ctx.ui.notify(`LiteLLM: ${result.models.length} models refreshed (source: ${result.source})`, "info");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ctx.ui.notify(`LiteLLM refresh failed: ${message}`, "error");
      }
    },
  });

  let sessionId: string | undefined;
  pi.on("session_start", (_event, ctx) => {
    sessionId = getSessionIdFromFile(ctx.sessionManager.getSessionFile());

    if (discoveryDisabledReason() || !cacheFetchedAt || Date.now() - cacheFetchedAt <= CACHE_STALE_MS) return;
    void runRefresh().catch(() => undefined);
  });

  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== PROVIDER_NAME) return;
    if (typeof event.payload !== "object" || event.payload === null) return;
    return prepareLiteLLMRequestPayload(event.payload as Record<string, unknown>, ctx.model?.id, sessionId);
  });

  pi.on("message_end", (event) => {
    if (event.message.role !== "assistant") return;
    const message = normalizeThinkTags(event.message as AssistantMessage);
    if (!message) return;
    return { message };
  });
}
