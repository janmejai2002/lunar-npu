import { test, expect, describe } from "bun:test";
import { CassowarySimplexSolver } from "../src/engine/cassowary.js";
import { ApcaOklchOptimizer } from "../src/engine/apca-oklch.js";
import { RowanCstEngine } from "../src/engine/srnc-cst.js";

describe("Cassowary Linear Simplex Inequality Solver", () => {
  test("Enforces 44px touch target on undersized interactive element", () => {
    const res = CassowarySimplexSolver.solveLayout("btn_submit", 28, 24, 390, true);

    expect(res.satisfied).toBe(false);
    expect(res.solved.width).toBe(44);
    expect(res.solved.height).toBe(44);
    expect(res.deltas.dw).toBe(16);
    expect(res.deltas.dh).toBe(20);
    expect(res.suggestedClasses).toContain("min-w-[44px]");
    expect(res.suggestedClasses).toContain("min-h-[44px]");
    expect(res.latencyMs).toBeLessThan(1.0);
  });

  test("Maintains compliant dimensions without unnecessary deltas", () => {
    const res = CassowarySimplexSolver.solveLayout("btn_large", 64, 48, 390, true);

    expect(res.satisfied).toBe(true);
    expect(res.solved.width).toBe(64);
    expect(res.solved.height).toBe(48);
    expect(res.deltas.dw).toBe(0);
    expect(res.deltas.dh).toBe(0);
    expect(res.suggestedClasses.length).toBe(0);
    expect(res.latencyMs).toBeLessThan(1.0);
  });

  test("Enforces mobile viewport boundary limit (390px mobile)", () => {
    const res = CassowarySimplexSolver.solveLayout("card_wide", 450, 100, 390, false);

    expect(res.satisfied).toBe(false);
    // 390 - 32px safety padding = 358px max
    expect(res.solved.width).toBe(358);
    expect(res.suggestedClasses).toContain("max-w-full");
    expect(res.suggestedClasses).toContain("truncate");
  });

  test("Enforces constraints directly on Rowan CST RedNode", () => {
    const tree = RowanCstEngine.parse('<button className="text-sm">Save</button>');
    const btnNode = tree.root.descendants().find((d) => d.toSource().startsWith("<button"));
    expect(btnNode).toBeDefined();

    const { tree: updatedTree, result } = CassowarySimplexSolver.enforceCstElement(
      tree,
      btnNode!,
      { width: 30, height: 26 },
      390
    );

    expect(result.satisfied).toBe(false);
    const updatedSource = updatedTree.toSource();
    expect(updatedSource).toContain("min-w-[44px]");
    expect(updatedSource).toContain("min-h-[44px]");
    expect(updatedSource).toContain("text-sm");
  });
});

describe("APCA OKLCH Convex Optimizer", () => {
  test("Calculates APCA contrast value Lc accurately", () => {
    // Dark on light and light on dark both exceed 100 in absolute contrast
    const lcDarkOnLight = ApcaOklchOptimizer.calculateApca([0, 0, 0], [255, 255, 255]);
    expect(Math.abs(lcDarkOnLight)).toBeGreaterThan(100);

    const lcLightOnDark = ApcaOklchOptimizer.calculateApca([255, 255, 255], [0, 0, 0]);
    expect(Math.abs(lcLightOnDark)).toBeGreaterThan(100);
  });

  test("Optimizes low-contrast text along OKLCH gamut in <1ms", () => {
    // Low contrast slate gray text (#64748b) on dark background (#0f172a)
    const darkBg = "#0f172a";
    const lowContrastFg = "#64748b";

    // Warm-up JIT compile
    ApcaOklchOptimizer.optimizeContrast(lowContrastFg, darkBg, 60.0);

    const opt = ApcaOklchOptimizer.optimizeContrast(lowContrastFg, darkBg, 60.0);

    expect(opt.compliant).toBe(true);
    expect(Math.abs(opt.optimizedLc)).toBeGreaterThanOrEqual(60.0);
    expect(opt.optimizedOklch.l).toBeGreaterThan(0.5); // Lightness increased on dark bg
    expect(opt.latencyMs).toBeLessThan(1.0); // Strict sub-1ms requirement
  });

  test("Leaves already compliant contrast untouched", () => {
    const darkBg = "#020617";
    const highContrastFg = "#f8fafc";

    const opt = ApcaOklchOptimizer.optimizeContrast(highContrastFg, darkBg, 60.0);

    expect(opt.compliant).toBe(true);
    expect(opt.iterations).toBe(0);
    expect(opt.optimizedHex.toLowerCase()).toBe(highContrastFg.toLowerCase());
    expect(opt.latencyMs).toBeLessThan(1.0);
  });

  test("Enforces APCA contrast directly on Rowan CST RedNode", () => {
    const tree = RowanCstEngine.parse('<span className="text-slate-500 font-medium">Notice</span>');
    const spanNode = tree.root.descendants().find((d) => d.toSource().startsWith("<span"));
    expect(spanNode).toBeDefined();

    const { tree: updatedTree, result } = ApcaOklchOptimizer.enforceCstContrast(
      tree,
      spanNode!,
      "#64748b",
      "#0f172a",
      60.0
    );

    expect(result.compliant).toBe(true);
    const updatedSource = updatedTree.toSource();
    expect(updatedSource).toContain("text-slate-100");
    expect(updatedSource).toContain("font-medium");
  });
});
