import { KENYA_COUNTY_PATHS } from "@/lib/data/kenya-county-paths";

export interface CountyMeta {
  id: string;
  name: string;
  calibrated: boolean;
  population?: number;
  available?: string;
}

export const COUNTY_STORAGE_KEY = "sdr_county";
export const DEFAULT_COUNTY_ID = "kakamega";

/** Counties shown in the map / selector (matches API meta). */
export const COUNTIES: CountyMeta[] = [
  { id: "kakamega", name: "Kakamega", calibrated: true, population: 1_872_000 },
  { id: "kisumu", name: "Kisumu", calibrated: false, available: "Q3 2026" },
  { id: "nairobi", name: "Nairobi", calibrated: false, available: "Q4 2026" },
  { id: "bungoma", name: "Bungoma", calibrated: false, available: "Q4 2026" },
];

export function getCountyById(id: string): CountyMeta | undefined {
  return COUNTIES.find((c) => c.id === id);
}

export function countyDisplayName(id: string): string {
  const c = getCountyById(id);
  if (c) return `${c.name} County`;
  const fromMap = id
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
  return `${fromMap} County`;
}

export function getCountyLabel(id: string): string {
  const meta = getCountyById(id);
  if (meta) return meta.name;
  const path = KENYA_COUNTY_PATHS.find((p) => p.id === id);
  return path?.name ?? id;
}

export function isCountySelectable(id: string): boolean {
  return getCountyById(id)?.calibrated === true;
}

export function getStoredCountyId(): string {
  if (typeof window === "undefined") return DEFAULT_COUNTY_ID;
  const raw = localStorage.getItem(COUNTY_STORAGE_KEY);
  return raw && isCountySelectable(raw) ? raw : DEFAULT_COUNTY_ID;
}

export function storeCountyId(id: string): void {
  if (typeof window === "undefined") return;
  if (isCountySelectable(id)) {
    localStorage.setItem(COUNTY_STORAGE_KEY, id);
  }
}
