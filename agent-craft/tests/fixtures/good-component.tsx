import React from "react";

export function CraftExcellenceHero() {
  return (
    <section className="min-h-[100dvh] max-w-[1200px] w-full bg-zinc-950 text-zinc-100 p-6">
      <header className="pb-6 border-b border-zinc-800">
        <span className="text-xs uppercase tracking-widest text-emerald-400 font-mono">
          System Overview
        </span>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">
          Autonomous Silicon Acceleration Engine
        </h1>
      </header>

      {/* Clean Single Layer Card with High Contrast */}
      <div className="mt-6 border border-zinc-800 rounded-xl p-6 bg-zinc-900/50 shadow-sm">
        <p className="text-zinc-300 text-sm leading-relaxed">
          Deterministic execution at sub-15ms latency across local client devices.
        </p>
      </div>

      {/* Accessible Button with 44px Minimum Hit Area and Focus Visible Ring */}
      <div className="mt-6 flex items-center gap-4">
        <button
          aria-label="Inspect hardware status"
          className="min-h-[44px] px-5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-zinc-950 font-medium text-sm transition-transform duration-200 ease-out focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-950"
        >
          Initialize Telemetry
        </button>

        {/* Animation with Reduced Motion Fallback */}
        <div className="animate-spin motion-reduce:animate-none h-5 w-5 text-emerald-400">
          <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" stroke="currentColor" /></svg>
        </div>
      </div>

      {/* Properly labelled image */}
      <img src="https://example.com/diagram.svg" alt="Silicon architecture workflow" className="mt-8 rounded-lg" />
    </section>
  );
}
