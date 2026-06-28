"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import PillSelector from "@/components/PillSelector";
import ScenarioSummarySidebar from "@/components/design/ScenarioSummarySidebar";
import { useLocale } from "@/components/i18n/LocaleProvider";
import { runScenario, waitForRun } from "@/lib/api";
import { DEFAULT_SCENARIO, HSSIntensity, Scenario } from "@/lib/scenarios";
import { scenarioFromURLParams, scenarioToSearchParams } from "@/lib/url-state";

const HSS_OPTIONS = [
  { value: "off", label: "Off" },
  { value: "light", label: "Light", hint: "60–69%" },
  { value: "moderate", label: "Moderate", hint: "70–79%" },
  { value: "intensive", label: "Intensive", hint: "80–95%" },
];

function DesignContent() {
  const { t } = useLocale();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [scenario, setScenario] = useState<Scenario>(DEFAULT_SCENARIO);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTreatments, setShowTreatments] = useState(false);
  const [showCommunity, setShowCommunity] = useState(false);

  useEffect(() => {
    const fromUrl = scenarioFromURLParams(searchParams.get("s"));
    if (fromUrl) {
      setScenario(fromUrl);
      setShowTreatments(fromUrl.treatments.enabled);
      setShowCommunity(fromUrl.community.enabled);
    }
  }, [searchParams]);

  const update = useCallback((patch: Partial<Scenario>) => {
    setScenario((prev) => ({ ...prev, ...patch }));
  }, []);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      let response = await runScenario(scenario);
      if (response.status === "pending") {
        response = await waitForRun(response.run_id);
      }
      if (response.status === "failed" || !response.result) {
        throw new Error(response.error_message || "Simulation failed");
      }
      const params = scenarioToSearchParams(scenario);
      params.set("run_id", response.run_id);
      router.push(`/results?${params.toString()}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Run failed");
    } finally {
      setRunning(false);
    }
  };

  const totalYears = scenario.run.implementation_years + scenario.run.maintenance_years;

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-8 py-8">
      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div>
            <h1 className="font-display text-3xl mb-2">{t("design.title")}</h1>
            <p className="text-ink-muted text-sm">{t("design.subtitle")}</p>
          </div>

          {/* Pillar 1: HSS */}
          <section className="bg-card border border-border rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-display text-lg">{t("design.pillar1")}</h2>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={scenario.hss.enabled}
                  onChange={(e) =>
                    update({
                      hss: {
                        ...scenario.hss,
                        enabled: e.target.checked,
                        intensity: e.target.checked ? "moderate" : "off",
                      },
                    })
                  }
                />
                {t("design.enable")}
              </label>
            </div>
            {scenario.hss.enabled && (
              <PillSelector
                options={HSS_OPTIONS.filter((o) => o.value !== "off")}
                value={scenario.hss.intensity}
                onChange={(v) =>
                  update({ hss: { ...scenario.hss, intensity: v as HSSIntensity } })
                }
              />
            )}
            {scenario.hss.enabled && (
              <details className="mt-4 pt-4 border-t border-border-soft">
                <summary className="text-sm text-ink-muted cursor-pointer hover:text-ink">
                  {t("design.moreParams")}
                </summary>
                <div className="mt-4 grid md:grid-cols-2 gap-4 text-sm">
                  <label className="block">
                    <span className="text-ink-muted text-xs">4+ ANC rate (%)</span>
                    <input
                      type="range"
                      min={56}
                      max={95}
                      value={Math.round((scenario.hss.p_anc ?? 0.8) * 100)}
                      onChange={(e) =>
                        update({
                          hss: { ...scenario.hss, p_anc: Number(e.target.value) / 100 },
                        })
                      }
                      className="w-full mt-1"
                    />
                    <span className="num text-xs">{Math.round((scenario.hss.p_anc ?? 0.8) * 100)}%</span>
                  </label>
                  <label className="block">
                    <span className="text-ink-muted text-xs">L4/5 delivery (%)</span>
                    <input
                      type="range"
                      min={38}
                      max={95}
                      value={Math.round((scenario.hss.p_l45 ?? 0.68) * 100)}
                      onChange={(e) =>
                        update({
                          hss: { ...scenario.hss, p_l45: Number(e.target.value) / 100 },
                        })
                      }
                      className="w-full mt-1"
                    />
                    <span className="num text-xs">{Math.round((scenario.hss.p_l45 ?? 0.68) * 100)}%</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={scenario.hss.refer_enabled ?? true}
                      onChange={(e) =>
                        update({ hss: { ...scenario.hss, refer_enabled: e.target.checked } })
                      }
                    />
                    Referral network enabled
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={scenario.hss.transfer_enabled ?? true}
                      onChange={(e) =>
                        update({ hss: { ...scenario.hss, transfer_enabled: e.target.checked } })
                      }
                    />
                    Emergency transfer enabled
                  </label>
                </div>
              </details>
            )}
          </section>

          {/* Pillar 2: Treatments */}
          {!showTreatments ? (
            <button
              type="button"
              onClick={() => {
                setShowTreatments(true);
                update({ treatments: { ...scenario.treatments, enabled: true } });
              }}
              className="w-full border-2 border-dashed border-border rounded-xl p-6 text-ink-muted hover:border-intervention hover:text-intervention transition"
            >
              + {t("design.addTreatments")}
            </button>
          ) : (
            <section className="bg-card border border-border rounded-xl p-6">
              <h2 className="font-display text-lg mb-4">{t("design.pillar2")}</h2>
              <div className="grid grid-cols-2 gap-3">
                {(
                  [
                    ["pph_bundle", "PPH bundle"],
                    ["iv_iron", "IV iron"],
                    ["mgso4", "MgSO4"],
                    ["antibiotics", "Antibiotics"],
                    ["oxytocin", "Oxytocin"],
                    ["ultrasound", "Ultrasound"],
                  ] as const
                ).map(([key, label]) => (
                  <label key={key} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={!!scenario.treatments[key]}
                      onChange={(e) =>
                        update({
                          treatments: { ...scenario.treatments, enabled: true, [key]: e.target.checked },
                        })
                      }
                    />
                    {label}
                  </label>
                ))}
              </div>
            </section>
          )}

          {/* Pillar 3: Community */}
          {!showCommunity ? (
            <button
              type="button"
              onClick={() => {
                setShowCommunity(true);
                update({ community: { ...scenario.community, enabled: true } });
              }}
              className="w-full border-2 border-dashed border-border rounded-xl p-6 text-ink-muted hover:border-intervention hover:text-intervention transition"
            >
              + {t("design.addCommunity")}
            </button>
          ) : (
            <section className="bg-card border border-border rounded-xl p-6">
              <h2 className="font-display text-lg mb-4">{t("design.pillar3")}</h2>
              <div className="space-y-3">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={scenario.community.prompts.enabled}
                    onChange={(e) =>
                      update({
                        community: {
                          ...scenario.community,
                          enabled: true,
                          prompts: {
                            ...scenario.community.prompts,
                            enabled: e.target.checked,
                            adoption: e.target.checked ? 0.8 : 0,
                            chv_engagement: e.target.checked ? 0.8 : 0,
                          },
                        },
                      })
                    }
                  />
                  PROMPTS (wired)
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={scenario.community.mentors.enabled}
                    onChange={(e) =>
                      update({
                        community: {
                          ...scenario.community,
                          enabled: true,
                          mentors: {
                            ...scenario.community.mentors,
                            enabled: e.target.checked,
                            adoption: e.target.checked ? 0.8 : 0,
                          },
                        },
                      })
                    }
                  />
                  MENTORS (wired)
                </label>
                <label className="flex items-center gap-2 text-sm text-ink-muted">
                  <input
                    type="checkbox"
                    checked={scenario.community.fqa.enabled}
                    onChange={(e) =>
                      update({
                        community: {
                          ...scenario.community,
                          enabled: true,
                          fqa: { ...scenario.community.fqa, enabled: e.target.checked },
                        },
                      })
                    }
                  />
                  FQA <span className="text-[10px] text-warning">● UI only</span>
                </label>
                <label className="flex items-center gap-2 text-sm text-ink-muted">
                  <input
                    type="checkbox"
                    checked={scenario.community.pulse.enabled}
                    onChange={(e) =>
                      update({
                        community: {
                          ...scenario.community,
                          enabled: true,
                          pulse: { ...scenario.community.pulse, enabled: e.target.checked },
                        },
                      })
                    }
                  />
                  PULSE <span className="text-[10px] text-warning">● UI only</span>
                </label>
              </div>
            </section>
          )}

          {/* Run settings */}
          <section className="bg-card border border-border rounded-xl p-6">
            <h2 className="font-display text-lg mb-4">{t("design.runSettings")}</h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-ink-muted block mb-2">
                  {t("design.timeline", {
                    years: totalYears,
                    impl: scenario.run.implementation_years,
                    maint: scenario.run.maintenance_years,
                  })}
                </label>
                <input
                  type="range"
                  min={1}
                  max={6}
                  value={scenario.run.implementation_years}
                  onChange={(e) =>
                    update({
                      run: { ...scenario.run, implementation_years: Number(e.target.value) },
                    })
                  }
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-sm text-ink-muted block mb-2">{t("design.runMode")}</label>
                <PillSelector
                  options={[
                    { value: "quick", label: "Quick", hint: "~1 min" },
                    { value: "robust", label: "Robust", hint: "Multiple runs + CI" },
                  ]}
                  value={scenario.run.mode}
                  onChange={(v) =>
                    update({ run: { ...scenario.run, mode: v as "quick" | "robust" } })
                  }
                />
              </div>
            </div>
          </section>
        </div>

        <ScenarioSummarySidebar
          scenario={scenario}
          running={running}
          error={error}
          onNameChange={(name) => update({ name })}
          onRun={handleRun}
        />
      </div>
    </div>
  );
}

export default function DesignPage() {
  return (
    <Suspense fallback={<div className="p-8 text-ink-muted">Loading…</div>}>
      <DesignContent />
    </Suspense>
  );
}
