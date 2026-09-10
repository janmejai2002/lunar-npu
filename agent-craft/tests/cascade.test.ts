import test from "node:test";
import assert from "node:assert/strict";
import { CanvasHierarchyTracker, compositeAlpha } from "../dist/engine/canvas-tracker.js";
import { extractClassBranches } from "../dist/engine/cn-eval.js";
import { auditSnippet } from "../dist/engine/detector.js";

test("CanvasHierarchyTracker accurately inherits ancestor canvas background", () => {
  const code = `
    <div className="bg-zinc-950 p-8">
      <header className="border-b border-zinc-800">
        <h1 className="text-white">Dashboard</h1>
      </header>
      <main className="mt-4">
        <p className="text-slate-400">System operational and healthy.</p>
      </main>
    </div>
  `;

  const tracker = new CanvasHierarchyTracker(code);
  const pSurface = tracker.getSurfaceAtLine(7);

  // Line 7 (<p>) should inherit the parent div's dark background!
  assert.equal(pSurface.isDark, true);
  assert.ok(pSurface.resolvedColor.r <= 20, "Expected dark canvas color");

  // Auditing this snippet should NOT report a contrast failure on text-slate-400
  // because text-slate-400 on bg-zinc-950 has >6.0:1 contrast ratio!
  const violations = auditSnippet(code, "dashboard.tsx");
  const contrastFailures = violations.filter((v) => v.ruleId === "a11y/wcag-aa-contrast");
  assert.equal(contrastFailures.length, 0, "Expected 0 contrast failures inside dark ancestor container");
});

test("CanvasHierarchyTracker catches contrast failure on light canvas", () => {
  const code = `
    <div className="bg-white p-8">
      <p className="text-slate-400">This text is too light to read on white canvas.</p>
    </div>
  `;

  const violations = auditSnippet(code, "card.tsx");
  const contrastFailures = violations.filter((v) => v.ruleId === "a11y/wcag-aa-contrast");
  assert.ok(contrastFailures.length >= 1, "Expected contrast failure on light canvas for text-slate-400");
  assert.ok(contrastFailures[0].message.includes("2.56:1") || contrastFailures[0].message.includes("yields only"));
});

test("Porter-Duff alpha compositing calculates correct blended RGB", () => {
  const top = { r: 255, g: 255, b: 255 }; // white
  const bottom = { r: 0, g: 0, b: 0 };     // black
  const blended50 = compositeAlpha(top, 0.5, bottom);

  assert.equal(blended50.r, 128);
  assert.equal(blended50.g, 128);
  assert.equal(blended50.b, 128);
});

test("cn-eval decomposes conditional branches to prevent cross-branch collisions", () => {
  const rawClasses = `cn("px-4 py-2", isActive ? "bg-indigo-600 text-white" : "text-slate-400")`;
  const branches = extractClassBranches(rawClasses);

  assert.equal(branches.length, 2);

  const activeBranch = branches.find((b) => b.branchId.startsWith("true_"));
  const inactiveBranch = branches.find((b) => b.branchId.startsWith("false_"));

  assert.ok(activeBranch, "Expected true branch");
  assert.ok(inactiveBranch, "Expected false branch");

  // Active branch contains bg-indigo-600 and text-white
  assert.ok(activeBranch!.classes.includes("bg-indigo-600"));
  assert.ok(activeBranch!.classes.includes("text-white"));
  assert.ok(!activeBranch!.classes.includes("text-slate-400"), "text-slate-400 must NOT be in active branch");

  // Inactive branch contains text-slate-400 and NOT bg-indigo-600
  assert.ok(inactiveBranch!.classes.includes("text-slate-400"));
  assert.ok(!inactiveBranch!.classes.includes("bg-indigo-600"), "bg-indigo-600 must NOT bleed into inactive branch");
});
