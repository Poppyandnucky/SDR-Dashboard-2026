import { HSSIntensity, Scenario } from "./scenarios";
import { DEFAULT_SCENARIO } from "./scenarios";

export type InterventionId =
  | "hss"
  | "pph_bundle"
  | "iv_iron"
  | "mgso4"
  | "antibiotics"
  | "oxytocin"
  | "ultrasound"
  | "prompts"
  | "mentors"
  | "fqa"
  | "pulse"
  | "referral_emt";

export type InterventionGroup = "supply" | "treatments" | "community";

export interface LibraryItem {
  id: InterventionId;
  name: string;
  group: InterventionGroup;
  description: string;
  wired: "wired" | "ui-only" | "partial";
}

export const INTERVENTION_LIBRARY: LibraryItem[] = [
  {
    id: "hss",
    name: "Health System Strengthening",
    group: "supply",
    description: "Facility capacity, training, and supply chain",
    wired: "wired",
  },
  {
    id: "pph_bundle",
    name: "PPH bundle",
    group: "treatments",
    description: "Postpartum haemorrhage treatment bundle",
    wired: "wired",
  },
  {
    id: "mgso4",
    name: "MgSO4 (Eclampsia)",
    group: "treatments",
    description: "Magnesium sulfate for eclampsia",
    wired: "wired",
  },
  {
    id: "iv_iron",
    name: "IV iron",
    group: "treatments",
    description: "Intravenous iron for anaemia",
    wired: "wired",
  },
  {
    id: "antibiotics",
    name: "Antibiotics",
    group: "treatments",
    description: "Antibiotics for maternal sepsis",
    wired: "wired",
  },
  {
    id: "oxytocin",
    name: "Oxytocin",
    group: "treatments",
    description: "Oxytocin for prolonged labour",
    wired: "wired",
  },
  {
    id: "ultrasound",
    name: "AI Ultrasound",
    group: "treatments",
    description: "Portable AI-assisted ultrasound",
    wired: "wired",
  },
  {
    id: "prompts",
    name: "PROMPTS",
    group: "community",
    description: "Community engagement via PROMPTS",
    wired: "wired",
  },
  {
    id: "mentors",
    name: "MENTORS",
    group: "community",
    description: "Mentorship sessions for providers",
    wired: "wired",
  },
  {
    id: "fqa",
    name: "FQA",
    group: "community",
    description: "Facility quality assessment",
    wired: "ui-only",
  },
  {
    id: "pulse",
    name: "PULSE",
    group: "community",
    description: "Pulse oximetry monitoring program",
    wired: "ui-only",
  },
  {
    id: "referral_emt",
    name: "Referral / EMT",
    group: "community",
    description: "Emergency medical transfer network",
    wired: "partial",
  },
];

export const GROUP_LABELS: Record<InterventionGroup, { label: string; dot: string }> = {
  supply: { label: "Supply (HSS)", dot: "bg-intervention" },
  treatments: { label: "Treatments", dot: "bg-warning" },
  community: { label: "Community (MOMISH)", dot: "bg-accent" },
};

export function hasIntervention(scenario: Scenario, id: InterventionId): boolean {
  switch (id) {
    case "hss":
      return scenario.hss.enabled && scenario.hss.intensity !== "off";
    case "pph_bundle":
      return !!scenario.treatments.pph_bundle;
    case "iv_iron":
      return !!scenario.treatments.iv_iron;
    case "mgso4":
      return !!scenario.treatments.mgso4;
    case "antibiotics":
      return !!scenario.treatments.antibiotics;
    case "oxytocin":
      return !!scenario.treatments.oxytocin;
    case "ultrasound":
      return !!scenario.treatments.ultrasound;
    case "prompts":
      return scenario.community.prompts.enabled;
    case "mentors":
      return scenario.community.mentors.enabled;
    case "fqa":
      return scenario.community.fqa.enabled;
    case "pulse":
      return scenario.community.pulse.enabled;
    case "referral_emt":
      return scenario.community.referral_emt.enabled;
  }
}

export function listActiveInterventions(scenario: Scenario): InterventionId[] {
  return INTERVENTION_LIBRARY.filter((item) => hasIntervention(scenario, item.id)).map(
    (item) => item.id
  );
}

export function applyIntervention(
  scenario: Scenario,
  id: InterventionId,
  opts?: { hssIntensity?: HSSIntensity }
): Scenario {
  const s = structuredClone(scenario);
  switch (id) {
    case "hss":
      s.hss = {
        ...s.hss,
        enabled: true,
        intensity: opts?.hssIntensity ?? (s.hss.intensity === "off" ? "moderate" : s.hss.intensity),
      };
      break;
    case "pph_bundle":
      s.treatments = { ...s.treatments, enabled: true, pph_bundle: true };
      break;
    case "iv_iron":
      s.treatments = { ...s.treatments, enabled: true, iv_iron: true };
      break;
    case "mgso4":
      s.treatments = { ...s.treatments, enabled: true, mgso4: true };
      break;
    case "antibiotics":
      s.treatments = { ...s.treatments, enabled: true, antibiotics: true };
      break;
    case "oxytocin":
      s.treatments = { ...s.treatments, enabled: true, oxytocin: true };
      break;
    case "ultrasound":
      s.treatments = { ...s.treatments, enabled: true, ultrasound: true };
      break;
    case "prompts":
      s.community = {
        ...s.community,
        enabled: true,
        prompts: {
          enabled: true,
          adoption: s.community.prompts.adoption ?? 0.8,
          chv_engagement: s.community.prompts.chv_engagement ?? 0.8,
          intervention_fidelity: s.community.prompts.intervention_fidelity ?? 0.75,
        },
      };
      break;
    case "mentors":
      s.community = {
        ...s.community,
        enabled: true,
        mentors: {
          enabled: true,
          adoption: s.community.mentors.adoption ?? 0.8,
          attendance: s.community.mentors.attendance ?? 0.8,
          fidelity: s.community.mentors.fidelity ?? 0.8,
        },
      };
      break;
    case "fqa":
      s.community = {
        ...s.community,
        enabled: true,
        fqa: { ...s.community.fqa, enabled: true },
      };
      break;
    case "pulse":
      s.community = {
        ...s.community,
        enabled: true,
        pulse: { ...s.community.pulse, enabled: true },
      };
      break;
    case "referral_emt":
      s.community = {
        ...s.community,
        enabled: true,
        referral_emt: {
          enabled: true,
          emt_participation: s.community.referral_emt.emt_participation ?? 0.7,
        },
      };
      break;
  }
  return s;
}

export function removeIntervention(scenario: Scenario, id: InterventionId): Scenario {
  const s = structuredClone(scenario);
  switch (id) {
    case "hss":
      s.hss = { ...s.hss, enabled: false, intensity: "off" };
      break;
    case "pph_bundle":
      s.treatments = { ...s.treatments, pph_bundle: false };
      if (!Object.values(s.treatments).some((v) => v === true)) s.treatments.enabled = false;
      break;
    case "iv_iron":
      s.treatments = { ...s.treatments, iv_iron: false };
      break;
    case "mgso4":
      s.treatments = { ...s.treatments, mgso4: false };
      break;
    case "antibiotics":
      s.treatments = { ...s.treatments, antibiotics: false };
      break;
    case "oxytocin":
      s.treatments = { ...s.treatments, oxytocin: false };
      break;
    case "ultrasound":
      s.treatments = { ...s.treatments, ultrasound: false };
      break;
    case "prompts":
      s.community = { ...s.community, prompts: { ...s.community.prompts, enabled: false } };
      break;
    case "mentors":
      s.community = { ...s.community, mentors: { ...s.community.mentors, enabled: false } };
      break;
    case "fqa":
      s.community = { ...s.community, fqa: { ...s.community.fqa, enabled: false } };
      break;
    case "pulse":
      s.community = { ...s.community, pulse: { ...s.community.pulse, enabled: false } };
      break;
    case "referral_emt":
      s.community = {
        ...s.community,
        referral_emt: { ...s.community.referral_emt, enabled: false },
      };
      break;
  }
  return s;
}

export function setHssIntensity(scenario: Scenario, intensity: HSSIntensity): Scenario {
  const s = structuredClone(scenario);
  s.hss = {
    ...s.hss,
    enabled: intensity !== "off",
    intensity,
  };
  return s;
}

export const QUICK_COMPARE_PRESETS: {
  label: string;
  a: Partial<Scenario>;
  b: Partial<Scenario>;
}[] = [
  {
    label: "Baseline vs HSS Intensive",
    a: { name: "Scenario A · Status quo", hss: { enabled: false, intensity: "off" } },
    b: { name: "Scenario B · HSS Intensive", hss: { enabled: true, intensity: "intensive" } },
  },
  {
    label: "HSS Moderate vs Combined",
    a: {
      name: "Scenario A · HSS Moderate",
      hss: { enabled: true, intensity: "moderate" },
    },
    b: {
      name: "Scenario B · Combined",
      hss: { enabled: true, intensity: "moderate" },
      treatments: { enabled: true, pph_bundle: true, mgso4: true },
      community: {
        enabled: true,
        prompts: { enabled: true, adoption: 0.6, chv_engagement: 0.6 },
        mentors: { enabled: false },
        fqa: { enabled: false, implementation: "low", influence_on_pulse: "low" },
        pulse: { enabled: false, implementation: "low" },
        referral_emt: { enabled: false },
      },
    },
  },
  {
    label: "MOMISH vs HSS + MOMISH",
    a: {
      name: "Scenario A · MOMISH only",
      community: {
        enabled: true,
        prompts: { enabled: true, adoption: 1, chv_engagement: 1, intervention_fidelity: 0.87 },
        mentors: { enabled: true, adoption: 0.8, attendance: 0.8, fidelity: 0.8 },
        fqa: { enabled: false, implementation: "low", influence_on_pulse: "low" },
        pulse: { enabled: false, implementation: "low" },
        referral_emt: { enabled: false },
      },
    },
    b: {
      name: "Scenario B · HSS + MOMISH",
      hss: { enabled: true, intensity: "moderate" },
      community: {
        enabled: true,
        prompts: { enabled: true, adoption: 0.8, chv_engagement: 0.8 },
        mentors: { enabled: true, adoption: 0.8 },
        fqa: { enabled: false, implementation: "low", influence_on_pulse: "low" },
        pulse: { enabled: false, implementation: "low" },
        referral_emt: { enabled: false },
      },
    },
  },
];

export function mergeScenario(base: Scenario, patch: Partial<Scenario>): Scenario {
  return {
    ...DEFAULT_SCENARIO,
    ...base,
    ...patch,
    hss: { ...DEFAULT_SCENARIO.hss, ...base.hss, ...patch.hss },
    treatments: { ...DEFAULT_SCENARIO.treatments, ...base.treatments, ...patch.treatments },
    community: {
      ...DEFAULT_SCENARIO.community,
      ...base.community,
      ...patch.community,
      prompts: {
        ...DEFAULT_SCENARIO.community.prompts,
        ...base.community?.prompts,
        ...patch.community?.prompts,
      },
      mentors: {
        ...DEFAULT_SCENARIO.community.mentors,
        ...base.community?.mentors,
        ...patch.community?.mentors,
      },
      fqa: {
        ...DEFAULT_SCENARIO.community.fqa,
        ...base.community?.fqa,
        ...patch.community?.fqa,
      },
      pulse: {
        ...DEFAULT_SCENARIO.community.pulse,
        ...base.community?.pulse,
        ...patch.community?.pulse,
      },
      referral_emt: {
        ...DEFAULT_SCENARIO.community.referral_emt,
        ...base.community?.referral_emt,
        ...patch.community?.referral_emt,
      },
    },
    run: { ...DEFAULT_SCENARIO.run, ...base.run, ...patch.run },
  };
}
