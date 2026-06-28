import { Fraunces, Instrument_Sans } from "next/font/google";
import ClientAppShell from "@/components/ClientAppShell";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
});

const instrument = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-instrument",
  display: "swap",
});

export const metadata = {
  title: "Kenya Maternal Health Decision Tool",
  description: "Design and compare maternal health scenarios for Kakamega County",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${fraunces.variable} ${instrument.variable}`}>
      <body className="font-sans antialiased min-h-screen">
        <ClientAppShell>{children}</ClientAppShell>
      </body>
    </html>
  );
}
