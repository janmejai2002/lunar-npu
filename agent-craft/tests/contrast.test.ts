import test from "node:test";
import assert from "node:assert/strict";
import {
  parseHex,
  getRelativeLuminance,
  getContrastRatio,
  rgbToHex,
} from "../dist/color/luminance.js";
import { rgbToOklch, oklchToRgb, solveOklchContrast } from "../dist/color/oklch.js";

test("WCAG 2.1 relative luminance calculation", () => {
  const white = parseHex("#FFFFFF")!;
  const black = parseHex("#000000")!;
  assert.equal(getRelativeLuminance(white), 1.0);
  assert.equal(getRelativeLuminance(black), 0.0);

  // sRGB mid gray (#808080 = 128)
  const gray = parseHex("#808080")!;
  const lum = getRelativeLuminance(gray);
  assert.ok(lum > 0.21 && lum < 0.22, `Expected ~0.2158, got ${lum}`);
});

test("WCAG 2.1 contrast ratio extremes", () => {
  const white = parseHex("#FFFFFF")!;
  const black = parseHex("#000000")!;
  const ratio = getContrastRatio(white, black);
  assert.equal(Math.round(ratio), 21);

  // Self contrast is always 1.0
  assert.equal(getContrastRatio(white, white), 1.0);
});

test("OKLCH roundtrip conversion fidelity", () => {
  const original = parseHex("#3B82F6")!; // Tailwind Blue-500
  const oklch = rgbToOklch(original);
  const backToRgb = oklchToRgb(oklch);

  assert.ok(Math.abs(original.r - backToRgb.r) <= 2);
  assert.ok(Math.abs(original.g - backToRgb.g) <= 2);
  assert.ok(Math.abs(original.b - backToRgb.b) <= 2);
});

test("Algorithmic OKLCH contrast solver converges to >= 4.5:1", () => {
  const fg = parseHex("#94A3B8")!; // Slate 400
  const bg = parseHex("#FFFFFF")!; // White
  const initialRatio = getContrastRatio(fg, bg);
  assert.ok(initialRatio < 3.0, "Slate-400 on white must fail WCAG AA initially");

  const solved = solveOklchContrast(fg, bg, 4.5);
  assert.ok(
    solved.achievedRatio >= 4.5,
    `Solved ratio must be >= 4.5, got ${solved.achievedRatio}`
  );

  // Preserves hue within 5 degrees
  const origOklch = rgbToOklch(fg);
  assert.ok(
    Math.abs(origOklch.h - solved.solvedOklch.h) < 5,
    "Solver must preserve original perceptual color hue"
  );
});
