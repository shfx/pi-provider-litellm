import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function readWorkflow(): string {
  return readFileSync(resolve(repoRoot, ".github/workflows/litellm-smoke.yml"), "utf8");
}

describe("LiteLLM smoke workflow", () => {
  it("uses the GitHub Models smoke model as a complete LiteLLM model id", () => {
    const workflow = readWorkflow();

    expect(workflow).toContain("GH_MODELS_SMOKE_MODEL: ${{ vars.GH_MODELS_SMOKE_MODEL || 'openai/gpt-4o-mini' }}");
    expect(workflow).toContain("model: ${GH_MODELS_SMOKE_MODEL}");
    expect(workflow).not.toContain("model: openai/${GH_MODELS_SMOKE_MODEL}");
  });
});
