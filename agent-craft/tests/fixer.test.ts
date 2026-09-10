import test from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { applyFixesToContent } from "../dist/engine/fixer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const badFixturePath = path.join(__dirname, "fixtures", "bad-component.tsx");

test("Fixer surgically repairs anti-patterns", () => {
  const badContent = fs.readFileSync(badFixturePath, "utf-8");
  const result = applyFixesToContent(badContent, "bad-component.tsx");

  assert.ok(result.fixesApplied >= 5, `Expected >= 5 fixes, got ${result.fixesApplied}`);

  // Fixed code checks
  assert.ok(
    result.fixedCode.includes("max-w-[800px] w-full"),
    "Fixed code must replace w-[800px] with responsive max-w"
  );

  assert.ok(
    result.fixedCode.includes("h-[100dvh]"),
    "Fixed code must upgrade h-screen to h-[100dvh]"
  );

  assert.ok(
    result.fixedCode.includes("p-4"),
    "Fixed code must snap p-[17px] to standard token p-4"
  );

  assert.ok(
    result.fixedCode.includes('<img alt=""'),
    "Fixed code must supply alt attribute for img"
  );

  assert.ok(
    result.fixedCode.includes("motion-reduce:animate-none"),
    "Fixed code must add motion-reduce guard to spinner"
  );
});
