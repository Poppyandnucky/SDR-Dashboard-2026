import {
  CompareResponse,
  Preset,
  RunResponse,
  Scenario,
  ScenarioResult,
} from "./scenarios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const message =
      err?.error?.message || err?.detail?.error?.message || `API error ${res.status}`;
    throw new Error(message);
  }
  return res.json();
}

export async function fetchPresets(): Promise<Preset[]> {
  const data = await apiFetch<{ presets: Preset[] }>("/api/v1/presets");
  return data.presets;
}

export async function runScenario(scenario: Scenario): Promise<RunResponse> {
  return apiFetch<RunResponse>("/api/v1/scenarios/run", {
    method: "POST",
    body: JSON.stringify(scenario),
  });
}

export async function pollRun(runId: string): Promise<RunResponse> {
  return apiFetch<RunResponse>(`/api/v1/scenarios/${runId}`);
}

export async function waitForRun(runId: string, maxAttempts = 120): Promise<RunResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    const response = await pollRun(runId);
    if (response.status === "complete" || response.status === "failed") {
      return response;
    }
    await new Promise((r) => setTimeout(r, 5000));
  }
  throw new Error("Simulation timed out. Please try Quick mode or try again later.");
}

export async function compareScenarios(
  scenarioA: Scenario,
  scenarioB: Scenario,
  mode: "quick" | "robust" = "quick"
): Promise<CompareResponse> {
  return apiFetch<CompareResponse>("/api/v1/scenarios/compare", {
    method: "POST",
    body: JSON.stringify({ scenario_a: scenarioA, scenario_b: scenarioB, mode }),
  });
}

export type { ScenarioResult };
