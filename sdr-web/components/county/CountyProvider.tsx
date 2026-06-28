"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  CountyMeta,
  COUNTIES,
  DEFAULT_COUNTY_ID,
  getCountyById,
  getStoredCountyId,
  isCountySelectable,
  storeCountyId,
} from "@/lib/counties";

interface CountyContextValue {
  countyId: string;
  county: CountyMeta;
  setCountyId: (id: string) => boolean;
  comingSoonCountyId: string | null;
  clearComingSoon: () => void;
}

const CountyContext = createContext<CountyContextValue | null>(null);

export function CountyProvider({ children }: { children: React.ReactNode }) {
  const [countyId, setCountyIdState] = useState(DEFAULT_COUNTY_ID);
  const [comingSoonCountyId, setComingSoonCountyId] = useState<string | null>(null);

  useEffect(() => {
    setCountyIdState(getStoredCountyId());
  }, []);

  const setCountyId = useCallback((id: string): boolean => {
    if (!isCountySelectable(id)) {
      setComingSoonCountyId(id);
      return false;
    }

    setCountyIdState(id);
    storeCountyId(id);
    setComingSoonCountyId(null);
    return true;
  }, []);

  const county = useMemo(
    () => getCountyById(countyId) ?? COUNTIES[0],
    [countyId]
  );

  const value = useMemo(
    () => ({
      countyId,
      county,
      setCountyId,
      comingSoonCountyId,
      clearComingSoon: () => setComingSoonCountyId(null),
    }),
    [countyId, county, setCountyId, comingSoonCountyId]
  );

  return <CountyContext.Provider value={value}>{children}</CountyContext.Provider>;
}

export function useCounty() {
  const ctx = useContext(CountyContext);
  if (!ctx) {
    throw new Error("useCounty must be used within CountyProvider");
  }
  return ctx;
}
