import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const binPath = path.join(__dirname, "..", "bin", "agent-craft.js");

test("MCP server connects over stdio and responds to tools/list", async () => {
  const child = spawn("node", [binPath, "mcp"], {
    stdio: ["pipe", "pipe", "inherit"],
  });

  const responses: any[] = [];
  let buffer = "";

  child.stdout.on("data", (chunk: Buffer) => {
    buffer += chunk.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.trim()) {
        try {
          responses.push(JSON.parse(line.trim()));
        } catch {
          // ignore non-json
        }
      }
    }
  });

  // Send initialize request
  const initReq = {
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "test-client", version: "1.0.0" },
    },
  };

  child.stdin.write(JSON.stringify(initReq) + "\n");

  // Send initialized notification
  const initializedNotif = {
    jsonrpc: "2.0",
    method: "notifications/initialized",
  };
  child.stdin.write(JSON.stringify(initializedNotif) + "\n");

  // Send tools/list request
  const listReq = {
    jsonrpc: "2.0",
    id: 2,
    method: "tools/list",
    params: {},
  };
  child.stdin.write(JSON.stringify(listReq) + "\n");

  // Wait for response
  await new Promise((resolve) => setTimeout(resolve, 800));

  child.kill();

  const listResp = responses.find((r) => r.id === 2);
  assert.ok(listResp, "Must receive tools/list response");
  assert.ok(listResp.result?.tools?.length >= 5, "Must expose at least 5 tools");

  const toolNames = listResp.result.tools.map((t: any) => t.name);
  assert.ok(toolNames.includes("craft_audit"), "Must include craft_audit");
  assert.ok(toolNames.includes("craft_fix"), "Must include craft_fix");
  assert.ok(toolNames.includes("craft_contrast"), "Must include craft_contrast");
  assert.ok(toolNames.includes("craft_tokens"), "Must include craft_tokens");
  assert.ok(toolNames.includes("craft_archetype"), "Must include craft_archetype");
});
