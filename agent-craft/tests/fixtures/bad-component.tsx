import React from "react";

export function BadAISlopHero() {
  return (
    <div className="h-screen w-[800px] bg-white p-[17px]">
      {/* Cliché Section Eyebrow */}
      <div className="text-xs uppercase">
        <span>01 · CAPABILITIES</span>
      </div>

      <h1 className="text-4xl font-sans text-slate-400">
        Trusted by 100k devs with 99.9% Uptime
      </h1>

      {/* Nested Card-in-Card Anti-Pattern */}
      <div className="border border-zinc-200 rounded-xl p-6 shadow-purple-500/50">
        <div className="border border-zinc-200 rounded-lg p-4">
          <p className="text-slate-400">Nested card text with bad contrast</p>
        </div>
      </div>

      {/* Cliché Purple/Indigo Gradient Button with Sub-44px Touch Target */}
      <button className="bg-gradient-to-r from-purple-500 to-indigo-500 h-8 w-8 outline-none">
        <svg viewBox="0 0 24 24"><path d="M0 0h24v24H0z" /></svg>
      </button>

      {/* Missing alt on image */}
      <img src="https://example.com/avatar.png" />

      {/* Infinite animation missing reduced-motion */}
      <div className="animate-spin transition-all">
        Loading...
      </div>
    </div>
  );
}
