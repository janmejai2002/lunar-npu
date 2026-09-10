import test from "node:test";
import assert from "node:assert/strict";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { auditFile, runAudit } from "../dist/engine/detector.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const badFixturePath = path.join(__dirname, "fixtures", "bad-component.tsx");
const goodFixturePath = path.join(__dirname, "fixtures", "good-component.tsx");

test("Detector flags all major slop defects on bad-component.tsx", () => {
  const violations = auditFile(badFixturePath);

  assert.ok(violations.length >= 8, `Expected at least 8 violations, found ${violations.length}`);

  const ruleIds = new Set(violations.map((v) => v.ruleId));

  // Layer 1: Slop
  assert.ok(ruleIds.has("slop/gradient-hero"), "Must detect purple-indigo gradient");
  assert.ok(ruleIds.has("slop/card-in-card"), "Must detect card-in-card nesting");
  assert.ok(ruleIds.has("slop/eyebrow-number"), "Must detect 01 eyebrow");
  assert.ok(ruleIds.has("slop/generic-copy"), "Must detect 99.9% Uptime / 100k devs");

  // Layer 2: Contrast & A11y
  assert.ok(ruleIds.has("a11y/wcag-aa-contrast"), "Must detect slate-400 contrast failure on white");
  assert.ok(ruleIds.has("a11y/image-alt"), "Must detect missing alt on img");
  assert.ok(ruleIds.has("a11y/outline-suppression"), "Must detect outline-none without ring");

  // Layer 3: Ergonomics & Viewport
  assert.ok(ruleIds.has("ergonomics/fixed-width"), "Must detect w-[800px]");
  assert.ok(ruleIds.has("ergonomics/tap-target"), "Must detect h-8 w-8 button");
  assert.ok(ruleIds.has("viewport/safari-jump"), "Must detect h-screen");

  // Layer 4: Tokens
  assert.ok(ruleIds.has("token/arbitrary-spacing"), "Must detect p-[17px]");

  // Layer 5: Motion
  assert.ok(ruleIds.has("perf/reduced-motion"), "Must detect animate-spin without reduced motion");
  assert.ok(ruleIds.has("perf/layout-animation"), "Must detect transition-all");
});

test("Detector passes good-component.tsx with 0 errors", () => {
  const violations = auditFile(goodFixturePath);
  const errors = violations.filter((v) => v.severity === "error");

  assert.equal(errors.length, 0, "Good component must have 0 errors");

  const report = runAudit(goodFixturePath);
  assert.ok(report.craftScore >= 90, `Good component must have high score, got ${report.craftScore}`);
});
