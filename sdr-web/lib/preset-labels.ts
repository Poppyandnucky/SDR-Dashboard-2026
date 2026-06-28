import { TranslateFn } from "./i18n";

const PRESET_I18N_KEY: Record<string, string> = {
  "status-quo": "statusQuo",
  "hss-intensive": "hssIntensive",
  momish: "momish",
  combined: "combined",
};

export interface PresetDisplay {
  name: string;
  subtitle: string;
  description: string;
}

export function getPresetDisplay(presetId: string, t: TranslateFn): PresetDisplay | null {
  const key = PRESET_I18N_KEY[presetId];
  if (!key) return null;
  return {
    name: t(`presetCards.${key}.name`),
    subtitle: t(`presetCards.${key}.subtitle`),
    description: t(`presetCards.${key}.description`),
  };
}
