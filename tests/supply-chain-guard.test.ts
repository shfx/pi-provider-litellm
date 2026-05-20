import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { checkSupplyChain } from "../scripts/supply-chain-guard.js";

describe("supply-chain guard", () => {
  it("rejects install hooks, optional runtime dependencies, Git specs, and packaged payloads", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-guard-"));

    try {
      await writeFile(
        join(fixture, "package.json"),
        JSON.stringify(
          {
            name: "malicious-fixture",
            version: "1.0.0",
            files: ["payload.js"],
            scripts: {
              postinstall: "node payload.js",
            },
            optionalDependencies: {
              "telemetry-helper": "git+https://github.com/attacker/telemetry-helper.git#deadbeef",
            },
          },
          null,
          2,
        ),
      );
      await writeFile(
        join(fixture, "package-lock.json"),
        JSON.stringify(
          {
            name: "malicious-fixture",
            version: "1.0.0",
            lockfileVersion: 3,
            packages: {
              "": {
                name: "malicious-fixture",
                version: "1.0.0",
                optionalDependencies: {
                  "telemetry-helper": "git+https://github.com/attacker/telemetry-helper.git#deadbeef",
                },
              },
              "node_modules/telemetry-helper": {
                version: "1.0.0",
                resolved: "git+https://github.com/attacker/telemetry-helper.git#deadbeef",
              },
            },
          },
          null,
          2,
        ),
      );
      await writeFile(join(fixture, "payload.js"), "console.log('payload');\n");

      const result = await checkSupplyChain(fixture);

      expect(result.ok).toBe(false);
      expect(result.errors.join("\n")).toContain("postinstall");
      expect(result.errors.join("\n")).toContain("optionalDependencies.telemetry-helper");
      expect(result.errors.join("\n")).toContain("non-registry resolved URL");
      expect(result.errors.join("\n")).toContain("payload.js");
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });

  it("accepts this package policy while keeping the publish build gate", async () => {
    const result = await checkSupplyChain(process.cwd(), { checkPackageContents: false });

    expect(result.errors).toEqual([]);
    expect(result.ok).toBe(true);
  });
});
