import React from "react";
import Navbar from "../components/landing/Navbar";
import HeroSection from "../components/landing/HeroSection";
import FeaturesSection from "../components/landing/FeaturesSection";
import DimensionsSection from "../components/landing/DimensionsSection";
import FooterSection from "../components/landing/FooterSection";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="flex-1">
        <HeroSection />
        <FeaturesSection />
        <DimensionsSection />
      </main>
      <FooterSection />
    </div>
  );
}