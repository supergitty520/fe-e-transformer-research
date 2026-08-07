import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "刚度约束和信息熵对深度学习的反作用及引申",
  description:
    "以FE-E深层传播实验为起点，交互展示稳定约束的反作用、传播观测器与教育引申。",
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
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
