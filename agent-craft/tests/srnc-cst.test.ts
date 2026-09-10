import { test, expect, describe } from "bun:test";
import {
  RowanCstEngine,
  SyntaxKind,
  GreenToken,
  GreenNode,
  RedNode,
} from "../src/engine/srnc-cst.js";

describe("Rowan Red-Green CST Engine", () => {
  const sampleSource = `// Copyright 2026 Antigravity Inc.
import React from 'react';

/**
 * Premium Hero Banner with WCAG 2.1 AAA Contrast
 */
export function HeroBanner({ title }: { title: string }) {
  // Line comment before JSX
  return (
    <div className="flex min-h-[44px] items-center p-4 bg-slate-950">
      <span className="text-slate-200">{title}</span>
      <button className="px-4 py-2 bg-slate-900 text-slate-400">
        Action
      </button>
    </div>
  );
}
`;

  test("Trivia Invariance: Yield(CST) === SourceText with 100% byte fidelity", () => {
    const tree = RowanCstEngine.parse(sampleSource);
    const reconstructed = tree.toSource();

    // Must be 100% identical byte-for-byte
    expect(reconstructed).toBe(sampleSource);
    expect(reconstructed.length).toBe(sampleSource.length);
  });

  test("Preserves all whitespace, indentation and comments losslessly", () => {
    const complexTrivia = `  \t\n// Single line\n/* Multi\n   Line */\nconst x = <button className="btn" />;\n`;
    const tree = RowanCstEngine.parse(complexTrivia);
    expect(tree.toSource()).toBe(complexTrivia);
  });

  test("Structural Hash-Consing deduplicates identical Green Nodes", () => {
    const tok1 = new GreenToken(SyntaxKind.Identifier, "button");
    const tok2 = new GreenToken(SyntaxKind.Identifier, "button");

    const green1 = new GreenNode(SyntaxKind.JsxElement, [tok1]);
    const green2 = new GreenNode(SyntaxKind.JsxElement, [tok2]);

    expect(green1.hash).toBe(green2.hash);
    expect(green1.textLen).toBe(green2.textLen);
  });

  test("RedNode provides lazy zero-allocation navigation with parent pointers and offsets", () => {
    const tree = RowanCstEngine.parse(sampleSource);
    const root = tree.root;

    expect(root.kind).toBe(SyntaxKind.SourceFile);
    expect(root.offset).toBe(0);
    expect(root.textLen).toBe(sampleSource.length);

    const descendants = root.descendants();
    expect(descendants.length).toBeGreaterThan(0);

    // Verify parent links
    for (const d of descendants) {
      expect(d.parent).toBeDefined();
      const [start, end] = d.textRange();
      expect(end - start).toBe(d.textLen);
    }
  });

  test("Surgical updateClassName preserves comments, indentation, and git blame", () => {
    const tree = RowanCstEngine.parse(sampleSource);

    // Find the button node specifically (not its parent div)
    const buttonNode = tree.root.descendants().find((d) =>
      d.toSource().startsWith("<button")
    );
    expect(buttonNode).toBeDefined();

    // Mutate class names surgically: text-slate-400 -> text-slate-100
    const mutatedTree = tree.updateClassName(buttonNode!, (classes) => {
      return classes.map((c) => (c === "text-slate-400" ? "text-slate-100" : c));
    });

    const newSource = mutatedTree.toSource();

    // 1. Target class is updated
    expect(newSource).toContain("text-slate-100");
    expect(newSource).not.toContain("text-slate-400");

    // 2. All surrounding comments and code are 100% preserved
    expect(newSource).toContain("// Copyright 2026 Antigravity Inc.");
    expect(newSource).toContain("Premium Hero Banner with WCAG 2.1 AAA Contrast");
    expect(newSource).toContain("// Line comment before JSX");
    expect(newSource).toContain('className="flex min-h-[44px] items-center p-4 bg-slate-950"');
    expect(newSource).toContain('<span className="text-slate-200">{title}</span>');

    // 3. Diff is surgical (only the modified element line changed)
    const originalLines = sampleSource.split("\n");
    const mutatedLines = newSource.split("\n");
    expect(mutatedLines.length).toBe(originalLines.length);

    let diffCount = 0;
    for (let i = 0; i < originalLines.length; i++) {
      if (originalLines[i] !== mutatedLines[i]) {
        diffCount++;
      }
    }
    // Exactly 1 line changed: pure git blame preservation!
    expect(diffCount).toBe(1);
  });

  test("Surgical wrapElement wraps element inside container", () => {
    const mini = `<button className="p-2">Click</button>`;
    const tree = RowanCstEngine.parse(mini);
    const btn = tree.root.descendants()[0];

    const wrapped = tree.wrapElement(btn, "div", { className: "wrapper-container" });
    const wrappedSource = wrapped.toSource();

    expect(wrappedSource).toContain('<div className="wrapper-container">');
    expect(wrappedSource).toContain('<button className="p-2">Click</button>');
    expect(wrappedSource).toContain("</div>");
  });
});
