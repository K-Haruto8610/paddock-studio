import type { Metadata } from "next";
import "./globals.css";
import "./playback.css";

export const metadata: Metadata = {
  title: "Paddock Studio｜GⅠ展開予想",
  description: "馬柱を学習した展開モデルと、ぬいぐるみの3DコースでJRA平地GⅠの流れを読む。",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body className="antialiased">{children}</body>
    </html>
  );
}
