// src/pages/Dashboard.jsx
import { mindsightAPI } from "../lib/apiClient";
import React, { useState, useMemo, useEffect } from "react";
import { useLocation, Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { 
  Plus, Brain, Calendar, Heart, Activity, Shield, Moon,
  Users, Briefcase, Stethoscope, AlertCircle, CheckCircle,
  FileText, Info, BarChart3
} from "lucide-react";
import { motion } from "framer-motion";
import { format } from "date-fns";

import Navbar from "../components/landing/Navbar";
import FooterSection from "../components/landing/FooterSection";
import ScoreGauge from "../components/results/ScoreGauge";
import RadarChart from "../components/results/RadarChart";
import BarChartCard from "../components/results/BarChartCard";
import RecommendationCard from "../components/results/RecommendationCard";

export default function Dashboard() {
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  // --- FETCH LATEST REPORT ---
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await mindsightAPI.latestReport();
        console.log('📊 Fetched latest report:', response.data);
        setData(response.data);
        localStorage.setItem('lastAssessmentResult', JSON.stringify(response.data));
      } catch (err) {
        console.error('❌ Failed to fetch latest report:', err);
        const savedData = localStorage.getItem('lastAssessmentResult');
        if (savedData) {
          try {
            const parsed = JSON.parse(savedData);
            console.log('📊 Using data from localStorage as fallback');
            setData(parsed);
          } catch (e) {
            setError('No assessment data found. Please complete an assessment first.');
          }
        } else {
          setError('No assessment data found. Please complete an assessment first.');
        }
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, [location.state]);

  // Save to localStorage
  useEffect(() => {
    if (data) {
      localStorage.setItem('lastAssessmentResult', JSON.stringify(data));
    }
  }, [data]);

  // --- EXTRACT DATA (directly from JSON) ---
  const domainScores = data?.domain_scores || {};
  const age = data?.age || 'N/A';
  const sex = data?.sex || 'Not specified';
  const idNo = data?.id_no || 'MS-ANONYMOUS';
  const schemaVersion = data?.schema_version || '3.8';
  const plainLanguageSummary = data?.plain_language_summary || null;

  // --- Domain-specific data ---
  const personalityData = useMemo(() => {
    const placement = domainScores.domain_1_personality?.placement || {};
    return Object.entries(placement).map(([key, value]) => ({
      trait: key.replace(/_/g, " ").toUpperCase(),
      value: value,
    }));
  }, [domainScores]);

  const selfEsteemData = domainScores.domain_2_self_esteem?.placement || {};
  const moodData = domainScores.domain_3_mood_sleep?.placement || {};
  const digitalData = domainScores.domain_4_digital_and_social?.placement || 
                      domainScores.domain_4_multitask?.placement || {};
  const burnoutData = domainScores.domain_5_occupational_burnout?.placement || {};
  const clinicalData = domainScores.domain_6_severe_clinical?.placement || {};

  // --- Top contributors for bar charts (all domains) ---
  const allContributors = useMemo(() => {
    const domains = [
      { key: 'domain_1_personality', label: 'Personality' },
      { key: 'domain_2_self_esteem', label: 'Self-Esteem' },
      { key: 'domain_3_mood_sleep', label: 'Mood & Sleep' },
      { key: 'domain_4_digital_and_social', label: 'Digital & Social' },
      { key: 'domain_5_occupational_burnout', label: 'Occupational Health' },
      { key: 'domain_6_severe_clinical', label: 'Clinical Severity' },
    ];
    return domains.map(({ key, label }) => {
      const domain = domainScores[key];
      if (!domain) return null;
      const contributors = (domain.top_contributors || []).map(item => ({
        name: item.display_name || item.feature,
        value: item.contribution || 0,
        direction: item.direction,
        relative_magnitude: item.relative_magnitude || 0,
      }));
      return { label, key, contributors, summary: domain.domain_summary, severity: domain.severity_tier };
    }).filter(Boolean);
  }, [domainScores]);

  // --- Existing bar chart data (used in the original layout) ---
  const burnoutChartData = useMemo(() => {
    return (domainScores.domain_5_occupational_burnout?.top_contributors || []).map(item => ({
      name: item.display_name || item.feature,
      value: item.contribution,
      direction: item.direction
    }));
  }, [domainScores]);

  const clinicalChartData = useMemo(() => {
    return (domainScores.domain_6_severe_clinical?.top_contributors || []).map(item => ({
      name: item.display_name || item.feature,
      value: item.contribution,
      direction: item.direction
    }));
  }, [domainScores]);

  const selfEsteemContributors = useMemo(() => {
    return (domainScores.domain_2_self_esteem?.top_contributors || []).map(item => ({
      name: item.display_name || item.feature,
      value: item.contribution,
      direction: item.direction
    }));
  }, [domainScores]);

  // --- Domain status cards ---
  const domainCards = useMemo(() => {
    return [
      {
        id: "domain_1_personality",
        name: "Personality",
        icon: Brain,
        status: "Completed",
        color: "bg-indigo-100 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400",
        description: "5 traits analyzed"
      },
      {
        id: "domain_2_self_esteem",
        name: "Self-Esteem",
        icon: Shield,
        status: selfEsteemData.classification || "Unknown",
        color: selfEsteemData.classification === "Normal" 
          ? "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400"
          : "bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400",
        description: `${selfEsteemData.score || 0}/${selfEsteemData.max_possible_score || 40}`
      },
      {
        id: "domain_3_mood_sleep",
        name: "Mood & Sleep",
        icon: Moon,
        status: moodData.severity_label || "Unknown",
        color: moodData.phq9_sum <= 4 
          ? "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400"
          : moodData.phq9_sum <= 9 
          ? "bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400"
          : "bg-rose-100 text-rose-600 dark:bg-rose-900/30 dark:text-rose-400",
        description: `PHQ-9: ${moodData.phq9_sum || 0}`
      },
      {
        id: "domain_4_digital_and_social",
        name: "Digital & Social",
        icon: Users,
        status: digitalData.classification || "Unknown",
        color: "bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400",
        description: `IA: ${digitalData.predicted_total_internet_addiction || 0}`
      },
      {
        id: "domain_5_occupational_burnout",
        name: "Occupational Health",
        icon: Briefcase,
        status: burnoutData.burnout_tier_label || "Unknown",
        color: burnoutData.burnout_index <= 4 
          ? "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400"
          : burnoutData.burnout_index <= 6 
          ? "bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400"
          : "bg-rose-100 text-rose-600 dark:bg-rose-900/30 dark:text-rose-400",
        description: `Index: ${burnoutData.burnout_index || 0}`
      },
      {
        id: "domain_6_severe_clinical",
        name: "Clinical Severity",
        icon: Stethoscope,
        status: clinicalData.predicted_condition_label || "Unknown",
        color: clinicalData.predicted_condition_code === 1 
          ? "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400"
          : "bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400",
        description: `Code: ${clinicalData.predicted_condition_code || 0}`
      },
    ];
  }, [selfEsteemData, moodData, digitalData, burnoutData, clinicalData]);

  // --- Recommendations (using consistent fields) ---
  const getRecommendations = () => {
    const recs = [];
    const personality = domainScores.domain_1_personality?.placement || {};
    if (personality.extraversion < -0.2) recs.push("Consider social activities to gradually increase extraversion");
    if (personality.emotional_stability < 0.3) recs.push("Practice emotional regulation techniques and mindfulness");
    if (personality.conscientiousness < 0.3) recs.push("Use organizational tools and set small daily goals");
    if (selfEsteemData.score < 20) recs.push("Practice positive self-affirmations and challenge negative thoughts");
    if (moodData.phq9_sum >= 5) {
      recs.push("Consider speaking with a mental health professional about mood symptoms");
      recs.push("Establish a consistent sleep schedule and exercise routine");
    }
    if ((digitalData.loneliness_score || 0) > 50) recs.push("Consider professional support for loneliness and social connection");
    if ((digitalData.predicted_total_internet_addiction || 0) > 50) recs.push("Set screen time limits and take regular digital breaks");
    if (burnoutData.burnout_index > 4) {
      recs.push("Prioritize work-life balance and set clear boundaries");
      recs.push("Take regular breaks and practice stress management techniques");
    }
    return recs.length > 0 ? recs : [
      "Maintain your current healthy habits and continue monitoring",
      "Regular self-care and mindfulness practices are recommended"
    ];
  };

  const getSeverity = () => {
    const scores = {
      mood: moodData.phq9_sum || 0,
      burnout: burnoutData.burnout_index || 0,
      self_esteem: selfEsteemData.score || 30,
      loneliness: digitalData.loneliness_score || 0,
    };
    if (scores.mood >= 10 || scores.burnout > 6 || scores.loneliness > 50) return "high";
    if (scores.mood >= 5 || scores.burnout > 4 || scores.self_esteem < 20 || scores.loneliness > 35) return "moderate";
    return "low";
  };

  // --- RENDERING ---
  if (isLoading) {
    return (
      <div className="flex min-h-screen flex-col">
        <Navbar />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-muted-foreground">Loading assessment results...</p>
          </div>
        </main>
        <FooterSection compact />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex min-h-screen flex-col">
        <Navbar />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <AlertCircle className="w-16 h-16 text-amber-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Assessment Data</h3>
            <p className="text-muted-foreground mb-6">{error || 'Please complete an assessment to see results.'}</p>
            <Link to="/assessment">
              <Button>Take Assessment</Button>
            </Link>
          </div>
        </main>
        <FooterSection compact />
      </div>
    );
  }

  // --- MAIN RENDER ---
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="page-shell flex-1">
        <div className="page-container">
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8"
        >
          <div>
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-primary/10 rounded-xl border border-primary/20">
                <Brain className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-3xl md:text-4xl font-display font-bold text-foreground tracking-tight">
                  Mental Health Assessment Dashboard
                </h1>
                <p className="text-muted-foreground text-sm mt-1">
                  ID: <span className="font-medium text-foreground">{idNo}</span> · {age} yrs · {sex} · v{schemaVersion}
                </p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="px-4 py-2 bg-background/50 backdrop-blur-sm shadow-sm">
              <Calendar className="w-4 h-4 mr-2 text-primary" />
              {format(new Date(), "MMM d, yyyy")}
            </Badge>
            <Link to="/assessment">
              <Button className="rounded-full gap-2 shadow-md hover:shadow-lg transition-all">
                <Plus className="w-4 h-4" /> New Assessment
              </Button>
            </Link>
          </div>
        </motion.div>

        {/* Domain Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {domainCards.map((card, i) => {
            const Icon = card.icon;
            return (
              <motion.div
                key={card.id}
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: i * 0.05 }}
              >
                <Card className="card-surface-hover h-full">
                  <CardContent className="p-5">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${card.color} bg-opacity-40`}>
                          <Icon className="w-5 h-5" />
                        </div>
                        <div>
                          <div className="text-sm font-semibold text-foreground">{card.name}</div>
                          <div className="text-xs text-muted-foreground mt-0.5">{card.description}</div>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-xs shadow-sm bg-background/50">
                        {card.status}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Domain Summaries Section */}
        {allContributors.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="mb-8"
          >
            <h2 className="text-xl font-display font-bold mb-4 flex items-center gap-2">
              <Info className="w-5 h-5 text-primary" /> Domain Summaries
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {allContributors.map((item, idx) => (
                <Card key={idx} className="card-surface-hover">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold text-foreground">{item.label}</h4>
                      <Badge variant="outline" className="text-xs">Severity: {item.severity}</Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
                      {item.summary}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </motion.div>
        )}

        {/* Main Content Grid (charts) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            <Card className="card-surface h-full">
              <CardContent className="p-0">
                <RadarChart
                  data={personalityData}
                  title="Personality Profile"
                  description="Big Five personality traits assessment"
                  color="hsl(var(--primary))"
                  height={340}
                />
              </CardContent>
            </Card>
          </div>
          <div>
            <Card className="card-surface h-full">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-display text-foreground">Self-Esteem</CardTitle>
                <p className="text-sm text-muted-foreground">Rosenberg Scale</p>
              </CardHeader>
              <CardContent className="flex flex-col items-center justify-center pt-4">
                <ScoreGauge
                  value={selfEsteemData.score || 0}
                  maxValue={selfEsteemData.max_possible_score || 40}
                  label={`${selfEsteemData.score || 0}/${selfEsteemData.max_possible_score || 40}`}
                  sublabel={selfEsteemData.classification || "Unknown"}
                  color={selfEsteemData.score >= 20 ? "hsl(var(--chart-5))" : "hsl(var(--chart-4))"}
                  size={180}
                />
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Mood & Sleep Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6 mb-8">
          <Card className="card-surface-hover">
            <CardContent className="p-5 sm:p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-primary/10 rounded-xl border border-primary/20">
                  <Heart className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">PHQ-9 Score</p>
                  <p className="text-3xl font-display font-bold text-foreground mt-1">{moodData.phq9_sum || 0}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="card-surface-hover">
            <CardContent className="p-5 sm:p-6">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-xl border ${moodData.phq9_sum <= 4 ? 'bg-emerald-500/10 border-emerald-500/20' : moodData.phq9_sum <= 9 ? 'bg-amber-500/10 border-amber-500/20' : 'bg-rose-500/10 border-rose-500/20'}`}>
                  <Activity className={`w-6 h-6 ${moodData.phq9_sum <= 4 ? 'text-emerald-600 dark:text-emerald-400' : moodData.phq9_sum <= 9 ? 'text-amber-600 dark:text-amber-400' : 'text-rose-600 dark:text-rose-400'}`} />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Depression Level</p>
                  <p className="text-xl font-semibold text-foreground mt-1">{moodData.severity_label || "Unknown"}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="card-surface-hover">
            <CardContent className="p-5 sm:p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-accent/10 rounded-xl border border-accent/20">
                  <Moon className="w-6 h-6 text-accent" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Sleep Duration</p>
                  <p className="text-3xl font-display font-bold text-foreground mt-1">{moodData.calculated_sleep_duration_hours || 0}h</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Digital & Social and Burnout Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className="card-surface">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-display text-foreground">Digital & Social Profile</CardTitle>
              <p className="text-sm text-muted-foreground">Internet Addiction & Loneliness Metrics</p>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-foreground">Internet Addiction</span>
                    <span className="font-semibold text-primary">{digitalData.predicted_total_internet_addiction || 0}/100</span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2.5 overflow-hidden shadow-inner">
                    <div 
                      className="bg-primary h-full rounded-full transition-all duration-1000"
                      style={{ width: `${Math.min((digitalData.predicted_total_internet_addiction || 0), 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-foreground">Loneliness Score</span>
                    <span className="font-semibold text-accent">{digitalData.loneliness_score || 0}/100</span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2.5 overflow-hidden shadow-inner">
                    <div 
                      className="bg-accent h-full rounded-full transition-all duration-1000"
                      style={{ width: `${Math.min((digitalData.loneliness_score || 0), 100)}%` }}
                    />
                  </div>
                </div>
                <div className="pt-2">
                  <Badge variant="secondary" className="font-medium">
                    {digitalData.classification || "Unknown"}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-surface">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-display text-foreground">Occupational Burnout</CardTitle>
              <p className="text-sm text-muted-foreground">Workplace Stress & Burnout Assessment</p>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between mb-5">
                <div>
                  <span className="text-sm font-medium text-muted-foreground">Burnout Index</span>
                  <p className="text-4xl font-display font-bold text-foreground mt-1">{burnoutData.burnout_index || 0}</p>
                </div>
                <Badge className={
                  burnoutData.burnout_index <= 4 
                    ? "bg-emerald-500/10 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300 border-emerald-200"
                    : burnoutData.burnout_index <= 6 
                    ? "bg-amber-500/10 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300 border-amber-200"
                    : "bg-rose-500/10 text-rose-700 dark:bg-rose-500/20 dark:text-rose-300 border-rose-200"
                }>
                  {burnoutData.burnout_tier_label || "Unknown"}
                </Badge>
              </div>
              <div className="w-full bg-secondary rounded-full h-2.5 overflow-hidden shadow-inner">
                <div 
                  className={`h-full rounded-full transition-all duration-1000 ${
                    burnoutData.burnout_index <= 4 
                      ? "bg-emerald-500" 
                      : burnoutData.burnout_index <= 6 
                      ? "bg-amber-500" 
                      : "bg-rose-500"
                  }`}
                  style={{ width: `${Math.min((burnoutData.burnout_index || 0) / 10 * 100, 100)}%` }}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Contributing Factors Charts (existing) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <Card className="card-surface">
            <CardContent className="p-0">
              <BarChartCard
                data={selfEsteemContributors}
                title="Self-Esteem Contributing Factors"
                description="Top contributing items to self-esteem score"
                dataKey="value"
                nameKey="name"
                color="hsl(var(--primary))"
                height={300}
              />
            </CardContent>
          </Card>
          <Card className="card-surface">
            <CardContent className="p-0">
              <BarChartCard
                data={burnoutChartData}
                title="Burnout Contributing Factors"
                description="Key factors affecting occupational burnout"
                dataKey="value"
                nameKey="name"
                color="hsl(var(--accent))"
                height={300}
              />
            </CardContent>
          </Card>
        </div>

        {/* Clinical Severity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card className="card-surface">
            <CardContent className="p-0">
              <BarChartCard
                data={clinicalChartData}
                title="Clinical Severity Contributors"
                description="Top contributing factors to clinical profile"
                dataKey="value"
                nameKey="name"
                color="hsl(var(--chart-3))"
                height={300}
              />
            </CardContent>
          </Card>
          <Card className="card-surface">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-display text-foreground">Clinical Summary</CardTitle>
              <p className="text-sm text-muted-foreground">Severity assessment overview</p>
            </CardHeader>
            <CardContent className="flex flex-col items-center justify-center h-[280px]">
              <div className="text-center">
                <div className={`p-5 rounded-full ${
                  clinicalData.predicted_condition_code === 1 
                    ? "bg-emerald-500/10 border border-emerald-500/20" 
                    : "bg-amber-500/10 border border-amber-500/20"
                } inline-block mb-5 shadow-sm`}>
                  {clinicalData.predicted_condition_code === 1 ? (
                    <CheckCircle className="w-12 h-12 text-emerald-600 dark:text-emerald-400" />
                  ) : (
                    <AlertCircle className="w-12 h-12 text-amber-600 dark:text-amber-400" />
                  )}
                </div>
                <h3 className="text-2xl font-bold text-foreground">{clinicalData.predicted_condition_label || "Unknown"}</h3>
                <p className="text-sm font-medium text-muted-foreground mt-2 bg-secondary/50 px-3 py-1 rounded-full inline-block">
                  Diagnostic Code: {clinicalData.predicted_condition_code || 0}
                </p>
                {clinicalData.anomaly_review_flag && (
                  <div className="mt-4">
                    <Badge variant="outline" className="bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30 px-3 py-1 shadow-sm">
                      <AlertCircle className="w-3 h-3 mr-1.5 inline-block" /> Anomaly Review Recommended
                    </Badge>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* --- Detailed Contributing Factors (all domains) --- */}
<div className="mt-8">
  <h2 className="text-xl font-display font-bold mb-4 flex items-center gap-2">
    <BarChart3 className="w-5 h-5 text-primary" /> Detailed Contributing Factors
  </h2>
  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
    {allContributors.map((item) => {
      if (!item.contributors.length) return null;
      return (
        <Card key={item.key} className="card-surface">
          <CardHeader className="pb-2">
            <CardTitle className="text-md font-display text-foreground">{item.label}</CardTitle>
            <p className="text-xs text-muted-foreground">Top contributors (raw contribution & relative magnitude)</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {item.contributors.slice(0, 4).map((c, idx) => (
                <div key={idx}>
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-foreground">{c.name}</span>
                    <span className="text-muted-foreground">
                      {c.value.toFixed(3)} ({c.relative_magnitude.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-1.5 mt-0.5 overflow-hidden">
                    <div 
                      className="bg-primary h-full rounded-full transition-all duration-700"
                      style={{ width: `${Math.min(c.relative_magnitude, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      );
    })}
  </div>
</div>
        {/* Recommendations */}
        <div className="mt-8">
          <h2 className="text-xl font-display font-bold mb-4">Personalized Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <RecommendationCard
              domain="personality"
              severity={getSeverity()}
              recommendations={getRecommendations().slice(0, 4)}
            />
            <RecommendationCard
              domain="mood_sleep"
              score={moodData.phq9_sum || 0}
              severity={moodData.phq9_sum <= 4 ? "low" : moodData.phq9_sum <= 9 ? "moderate" : "high"}
              recommendations={[
                moodData.phq9_sum >= 5 ? "Consider speaking with a mental health professional" : "Continue maintaining your current mood management strategies",
                moodData.calculated_sleep_duration_hours < 7 ? "Aim for 7-9 hours of quality sleep per night" : "Maintain your healthy sleep schedule",
                "Practice daily mindfulness or meditation for 10-15 minutes"
              ]}
            />
          </div>
        </div>

        {/* Plain Language Summary (bottom) */}
        {plainLanguageSummary && plainLanguageSummary.full_text && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-8"
          >
            <Card className="card-surface border-l-4 border-l-primary overflow-hidden">
              <CardContent className="py-4 bg-primary/5">
                <div className="flex items-start gap-3">
                  <FileText className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className="font-semibold text-foreground">Plain Language Summary</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed mt-1 whitespace-pre-line">
                      {plainLanguageSummary.full_text}
                    </p>
                    {plainLanguageSummary.recommend_professional_help && (
                      <Badge className="mt-2 bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30">
                        <AlertCircle className="w-3 h-3 mr-1.5 inline-block" /> Professional consultation recommended
                      </Badge>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Footer */}
        <div className="mt-8 pt-8 border-t border-border/50">
          <div className="flex flex-wrap gap-4 justify-between items-center">
            <div className="text-sm text-muted-foreground">
              Assessment ID: {idNo} · Schema: v{schemaVersion}
            </div>
            <div className="flex gap-3">
              <Button variant="outline" size="sm" className="hover:bg-primary/10 hover:text-primary hover:border-primary/30 transition-all">
                <Brain className="w-4 h-4 mr-2" /> Export Data
              </Button>
              <Button variant="outline" size="sm" className="hover:bg-accent/10 hover:text-accent hover:border-accent/30 transition-all">
                <Activity className="w-4 h-4 mr-2" /> Print Report
              </Button>
            </div>
          </div>
        </div>
        </div>
      </main>
      <FooterSection compact />
    </div>
  );
}