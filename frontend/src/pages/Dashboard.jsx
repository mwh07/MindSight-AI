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
  FileText, Info, BarChart3, ChevronDown, ChevronUp
} from "lucide-react";
import { motion } from "framer-motion";
import { format } from "date-fns";

import Navbar from "../components/landing/Navbar";
import FooterSection from "../components/landing/FooterSection";
import ScoreGauge from "../components/results/ScoreGauge";
import RadarChart from "../components/results/RadarChart";
import RecommendationCard from "../components/results/RecommendationCard";

export default function Dashboard() {
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [activeDomainKey, setActiveDomainKey] = useState("domain_1_personality");

  const handleExportData = () => {
    if (!data) return;
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(data, null, 2)
    )}`;
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", jsonString);
    downloadAnchor.setAttribute("download", `mindsight_profile_${data.id_no || "MS-ANONYMOUS"}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

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
      { key: 'domain_1_personality', label: 'Personality', icon: Brain, color: "bg-indigo-100 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400" },
      { key: 'domain_2_self_esteem', label: 'Self-Esteem', icon: Shield, color: "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400" },
      { key: 'domain_3_mood_sleep', label: 'Mood & Sleep', icon: Moon, color: "bg-rose-100 text-rose-600 dark:bg-rose-900/30 dark:text-rose-400" },
      { key: 'domain_4_digital_and_social', label: 'Digital & Social', icon: Users, color: "bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400" },
      { key: 'domain_5_occupational_burnout', label: 'Occupational Health', icon: Briefcase, color: "bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400" },
      { key: 'domain_6_severe_clinical', label: 'Clinical Severity', icon: Stethoscope, color: "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400" },
    ];
    return domains.map(({ key, label, icon, color }) => {
      const domain = domainScores[key];
      if (!domain) return null;
      const contributors = (domain.top_contributors || []).map(item => ({
        name: item.display_name || item.feature,
        value: item.contribution || 0,
        direction: item.direction,
        relative_magnitude: item.relative_magnitude || 0,
      }));
      return { label, key, icon, color, contributors, summary: domain.domain_summary, severity: domain.severity_tier };
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
                  MINDSIGHT Dashboard
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

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {/* Digital & Social Profile */}
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

          {/* Occupational Burnout */}
          <Card className="card-surface">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-display text-foreground">Occupational Burnout</CardTitle>
              <p className="text-sm text-muted-foreground">Workplace Stress & Burnout Assessment</p>
            </CardHeader>
            <CardContent className="flex flex-col justify-between h-[200px]">
              <div>
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
              </div>
            </CardContent>
          </Card>

          {/* Clinical Summary */}
          <Card className="card-surface">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-display text-foreground">Clinical Summary</CardTitle>
              <p className="text-sm text-muted-foreground">Severity assessment overview</p>
            </CardHeader>
            <CardContent className="flex flex-col justify-between h-[200px] pb-4">
              <div className="flex items-start gap-4">
                <div className={`p-2.5 rounded-xl ${
                  clinicalData.predicted_condition_code === 1 
                    ? "bg-emerald-500/10 border border-emerald-500/20" 
                    : "bg-amber-500/10 border border-amber-500/20"
                } inline-block shadow-sm`}>
                  {clinicalData.predicted_condition_code === 1 ? (
                    <CheckCircle className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
                  ) : (
                    <AlertCircle className="w-6 h-6 text-amber-600 dark:text-amber-400" />
                  )}
                </div>
                <div>
                  <h4 className="text-md font-bold text-foreground">{clinicalData.predicted_condition_label || "Unknown"}</h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    Diagnostic Code: {clinicalData.predicted_condition_code || 0}
                  </p>
                </div>
              </div>
              {clinicalData.anomaly_review_flag && (
                <div>
                  <Badge variant="outline" className="bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30 px-2 py-0.5 shadow-sm text-xs">
                    <AlertCircle className="w-3 h-3 mr-1 inline-block" /> Anomaly Review Recommended
                  </Badge>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* --- Combined Domain Summaries & Detailed Contributing Factors (Dynamic Tabbed Panel) --- */}
        {allContributors.length > 0 && (() => {
          const activeDomain = allContributors.find(d => d.key === activeDomainKey) || allContributors[0];
          const ActiveIcon = activeDomain.icon || Info;
          return (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="mt-8"
            >
              <h2 className="text-2xl font-display font-bold mb-6 flex items-center gap-2 text-foreground">
                <BarChart3 className="w-6 h-6 text-primary" /> Domain Summaries & Contributing Factors
              </h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Side: Sidebar menu tabs */}
                <div className="flex flex-col gap-3">
                  {allContributors.map((item) => {
                    const TabIcon = item.icon || Info;
                    const isActive = activeDomainKey === item.key;
                    return (
                      <div
                        key={item.key}
                        onClick={() => setActiveDomainKey(item.key)}
                        className={`p-4 flex items-center justify-between cursor-pointer rounded-xl border transition-all duration-200 ${
                          isActive 
                            ? "bg-primary/5 border-primary/40 text-primary shadow-sm ring-1 ring-primary/20" 
                            : "bg-card hover:bg-secondary/20 border-border/60 text-foreground"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${item.color} bg-opacity-40`}>
                            <TabIcon className="w-4 h-4" />
                          </div>
                          <span className="font-semibold text-sm">{item.label}</span>
                        </div>
                        {item.severity && (
                          <Badge variant="outline" className={`text-[10px] uppercase font-bold py-0.5 px-2 bg-background/50 border-primary/10 ${
                            isActive ? "text-primary border-primary/20" : "text-muted-foreground"
                          }`}>
                            {item.severity?.toLowerCase() === 'descriptive' 
                            ? item.severity
                            : `RISK : ${item.severity}`}
                          </Badge>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Right Side: Active tab details display */}
                <div className="lg:col-span-2">
                  <Card className="card-surface h-full overflow-hidden border-primary/10 shadow-md">
                    <CardHeader className="p-5 border-b border-border/40 bg-secondary/5">
                      <div className="flex items-center gap-4">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${activeDomain.color} bg-opacity-40`}>
                          <ActiveIcon className="w-5 h-5" />
                        </div>
                        <div>
                          <CardTitle className="font-display font-bold text-xl text-foreground">
                            {activeDomain.label} Details
                          </CardTitle>
                          <p className="text-xs text-muted-foreground mt-0.5">Comprehensive view of diagnostics and model drivers</p>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Summary narrative */}
                        <div>
                          <h5 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-1.5">
                            <Info className="w-4 h-4 text-primary" /> Overview Summary
                          </h5>
                          <p className="text-sm text-foreground leading-relaxed whitespace-pre-line bg-secondary/10 p-5 rounded-xl border border-border/30">
                            {activeDomain.summary || "No description summary available for this domain."}
                          </p>
                        </div>

                        {/* Top Drivers / Contributors */}
                        <div>
                          <h5 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-1.5">
                            <BarChart3 className="w-4 h-4 text-accent" /> Key Influencing Drivers
                          </h5>
                          {activeDomain.contributors.length > 0 ? (
                            <div className="space-y-4 bg-secondary/10 p-5 rounded-xl border border-border/30">
                              {activeDomain.contributors.map((c, idx) => (
                                <div key={idx}>
                                  <div className="flex justify-between items-center text-xs">
                                    <span className="font-medium text-foreground">{c.name}</span>
                                    <span className="text-muted-foreground font-mono text-[10px]">
                                      Direction: <span className={c.direction === "+" ? "text-rose-500 font-bold" : "text-emerald-500 font-bold"}>{c.direction}</span> ({c.relative_magnitude.toFixed(1)}%)
                                    </span>
                                  </div>
                                  <div className="w-full bg-secondary rounded-full h-2 mt-2 overflow-hidden">
                                    <div 
                                      className={`h-full rounded-full transition-all duration-700 ${c.direction === "+" ? 'bg-rose-500' : 'bg-emerald-500'}`}
                                      style={{ width: `${Math.min(c.relative_magnitude, 100)}%` }}
                                    />
                                  </div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p className="text-sm text-muted-foreground italic">No contributing factors available for this domain.</p>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </motion.div>
          );
        })()}

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

        {/* Clinical Synthesis Narrative / Plain Language Summary */}
        {plainLanguageSummary && plainLanguageSummary.full_text && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-8"
          >
            <Card className="card-surface border-l-4 border-l-primary overflow-hidden bg-gradient-to-br from-card to-primary/5 shadow-md">
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-primary/10 rounded-xl border border-primary/20 flex-shrink-0">
                    <FileText className="w-6 h-6 text-primary" />
                  </div>
                  <div className="space-y-3 flex-1">
                    <div>
                      <h3 className="font-display font-bold text-xl text-foreground">Plain Language Summary</h3>
                      <p className="text-xs text-muted-foreground mt-0.5">A clear, non-technical overview of the assessment findings.</p>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line bg-background/50 p-4 rounded-xl border border-border/30">
                      {plainLanguageSummary.full_text}
                    </p>
                    {plainLanguageSummary.recommend_professional_help && (
                      <div className="pt-1">
                        <Badge className="bg-amber-500/10 hover:bg-amber-500/15 text-amber-700 dark:text-amber-400 border border-amber-500/20 px-3 py-1 text-xs shadow-sm">
                          <AlertCircle className="w-3.5 h-3.5 mr-1.5 inline-block text-amber-600 dark:text-amber-400" /> Clinical Recommendation: Professional consultation advised
                        </Badge>
                      </div>
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
              <Button 
                variant="default"
                className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold shadow-md transition-all px-5 py-2 rounded-xl flex items-center justify-center border-0"
                onClick={handleExportData}
              >
                <Brain className="w-4 h-4 mr-2" /> Export Data
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                className="hover:bg-accent/10 hover:text-accent hover:border-accent/30 transition-all"
                onClick={() => window.print()}
              >
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