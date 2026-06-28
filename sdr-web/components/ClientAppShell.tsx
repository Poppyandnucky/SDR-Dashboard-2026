"use client";

import OnboardingModal from "@/components/modals/OnboardingModal";
import TopNav from "@/components/TopNav";
import { CountyProvider } from "@/components/county/CountyProvider";
import { LocaleProvider } from "@/components/i18n/LocaleProvider";

export default function ClientAppShell({ children }: { children: React.ReactNode }) {
  return (
    <LocaleProvider>
      <CountyProvider>
        <TopNav />
        <main>{children}</main>
        <OnboardingModal />
      </CountyProvider>
    </LocaleProvider>
  );
}
