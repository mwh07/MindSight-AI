import React from "react";
// import { useQuery } from "@tanstack/react-query";
// import { base44 } from "@/api/base44Client";
import { Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { ArrowLeft, Download, RefreshCw } from "lucide-react";
import { motion } from "framer-motion";

import Navbar from "../components/landing/Navbar";
import FooterSection from "../components/landing/FooterSection";
import ScoreGauge from "../components/results/ScoreGauge";
import RadarChartCard from "../components/results/RadarChart";
import BarChartCard from "../components/results/BarChartCard";
import RecommendationCard from "../components/results/RecommendationCard";
// import { DIMENSION_INFO, getSeverityLevel } from "../lib/questionnaire-data";

export default function Results() {
  const urlParams = new URLSearchParams(window.location.search);
  const id = urlParams.get("id");

  const { data: assessment, isLoading } = useQuery({
    queryKey: ["assessment", id],
    queryFn: () => base44.entities.Assessment.filter({ id }),
    enabled: !!id,
  });

  const record = assessment?.[0];

  if (isLoading) {
    return (
      <div className="flex min-h-screen flex-col">
        <Navbar />
        <main className="page-shell flex flex-1 items-center justify-center">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card-surface p-10 flex flex-col items-center max-w-sm text-center"
          >
            <div className="relative mb-6">
              <div className="absolute inset-0 bg-primary/20 rounded-full animate-ping"></div>
              <div className="relative h-12 w-12 animate-spin rounded-full border-4 border-primary/30 border-t-primary" />
            </div>
            <h2 className="text-xl font-semibold text-foreground">Analyzing Results</h2>
            <p className="text-sm text-muted-foreground mt-2">Please wait while we compile your assessment data...</p>
          </motion.div>
        </main>
        <FooterSection compact />
      </div>
    );
  }

  if (!record) {
    return (
      <div className="flex min-h-screen flex-col">
        <Navbar />
        <main className="page-shell flex flex-1 items-center justify-center">
          <motion.div 
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            className="card-surface p-10 flex flex-col items-center max-w-md text-center"
          >
            <div className="h-16 w-16 bg-rose-500/10 rounded-2xl flex items-center justify-center mb-6 border border-rose-500/20">
              <span className="text-rose-500">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
              </span>
            </div>
            <h2 className="font-display text-2xl font-bold text-foreground">Assessment Not Found</h2>
            <p className="mt-3 text-muted-foreground leading-relaxed">
              We couldn't locate the requested assessment. It may have been deleted or the link might be invalid.
            </p>
            <Link to="/assessment" className="w-full mt-8">
              <Button className="w-full rounded-full shadow-md hover:shadow-lg transition-all h-11">
                Take New Assessment
              </Button>
            </Link>
          </motion.div>
        </main>
        <FooterSection compact />
      </div>
    );
  }

  const scores = record.scores || {};
  const predictions = record.predictions || {};

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="page-shell flex-1">
        <div className="page-container-medium">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8 bg-card border border-border/50 p-6 rounded-2xl shadow-sm"
        >
          <div>
            <h1 className="text-3xl font-display font-bold text-foreground">Your Results</h1>
            <p className="text-muted-foreground mt-1 font-medium">
              Assessment completed · Age: {record.age}
            </p>
          </div>
          <div className="flex gap-3">
            <Link to="/assessment">
              <Button variant="outline" className="rounded-full gap-2 hover:bg-secondary/80">
                <RefreshCw className="w-4 h-4" /> Retake
              </Button>
            </Link>
            <Link to="/dashboard">
              <Button className="rounded-full gap-2 shadow-md hover:shadow-lg transition-all">
                <ArrowLeft className="w-4 h-4" /> Dashboard
              </Button>
            </Link>
          </div>
        </motion.div>

        {/* Score Gauges */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mb-8"
        >
          {Object.entries(DIMENSION_INFO).map(([key, info]) => (
            <ScoreGauge
              key={key}
              label={info.label}
              score={scores[key] || 0}
              maxScore={info.maxScore}
              severity={getSeverityLevel(key, scores[key] || 0)}
              color={info.color}
            />
          ))}
        </motion.div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <RadarChartCard scores={scores} />
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <BarChartCard scores={scores} />
          </motion.div>
        </div>

        {/* Recommendations */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
          <RecommendationCard predictions={predictions} scores={scores} />
        </motion.div>
        </div>
      </main>
      <FooterSection compact />
    </div>
  );
}