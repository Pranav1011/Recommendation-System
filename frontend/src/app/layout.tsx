import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Movie Recommender | AI-Powered Recommendations",
  description:
    "Discover your next favorite movie with our AI-powered recommendation engine. Rate movies and get personalized suggestions based on your taste.",
  keywords: ["movies", "recommendations", "AI", "machine learning", "film"],
  authors: [{ name: "Sai Pranav Krovvidi" }],
  openGraph: {
    title: "Movie Recommender",
    description: "AI-powered movie recommendations based on your preferences",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
