import React from "react";
import { Card, CardContent } from "../components/ui/card";
import { Brain, Layers, GitBranch, BarChart3, Shield, Cpu, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Button } from "../components/ui/button";

import Navbar from "../components/landing/Navbar";
import FooterSection from "../components/landing/FooterSection";

const steps = [
  { icon: Layers, title: "70-Item Input Data", desc: "Responses are collected via the IMP-70 framework covering 6 psychological domains." },
  { icon: GitBranch, title: "Parallel Extraction", desc: "Each of the six domains independently extracts its required subset of features without contamination." },
  { icon: Brain, title: "Decoupled Inference", desc: "Domain-specific algorithms (GRM, XGBoost, Random Forest, K-Means) execute in parallel." },
  { icon: Cpu, title: "Outlier Detection", desc: "Isolation Forests independently flag atypical clinical response patterns to ensure data integrity." },
  { icon: BarChart3, title: "Profile Aggregation", desc: "Independent predictions are merged and severity levels are cross-referenced dynamically." },
  { icon: Shield, title: "Diagnostic Reporting", desc: "A synthesized global profile and plain-language narrative are generated into a final report." },
];

const team = [
  { name: "Md Waki Hasmi", id: "10200122057", photoPath: "/photos/Waki.jpg", fitClass: "object-cover object-center scale-[1.02]" },
  { name: "Abhibrata Nandy", id: "10200122045", photoPath: "/photos/abhi.jpg", fitClass: "object-cover object-center" },
  { name: "Arpan Pal", id: "10200122043", photoPath: "/photos/arpan.jpg", fitClass: "object-cover object-center" },
];

export default function About() {
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="flex-1 pb-20">
        <div className="page-shell">
          {/* Hero */}
          <div className="page-container-medium mb-16 sm:mb-24 text-center">
            <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6 shadow-sm border border-primary/20">
                <Brain className="w-8 h-8 text-primary" />
              </div>
              <h1 className="section-heading">About MINDSIGHT</h1>
              <p className="section-subheading-center mt-6">
                A Unified Multi-Domain Psychological Diagnostic Profiling System designed to deliver clinically-grounded, multi-dimensional mental health assessments with robust analytical precision.
              </p>
            </motion.div>
          </div>

          {/* Architecture Pipeline */}
          <div className="page-container-medium mb-24">
            <h2 className="text-2xl font-display font-bold text-center mb-10 text-foreground">Pipeline Execution Workflow</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
              {steps.map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 15 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.1 }}
                >
                  <Card className="h-full card-surface-hover relative overflow-hidden group">
                    <CardContent className="p-6 sm:p-8">
                      <div className="absolute -top-4 -right-4 text-[6rem] font-bold text-primary/5 font-display select-none transition-transform group-hover:scale-110">
                        {i + 1}
                      </div>
                      <div className="relative z-10">
                        <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-5 group-hover:bg-primary/20 transition-colors">
                          <step.icon className="w-6 h-6 text-primary" />
                        </div>
                        <h3 className="font-semibold text-lg mb-2 text-foreground">{step.title}</h3>
                        <p className="text-sm text-muted-foreground leading-relaxed">{step.desc}</p>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Key Innovations */}
        <div className="bg-secondary/40 border-y border-border/50 py-20">
          <div className="page-container-narrow">
            <h2 className="text-2xl font-display font-bold text-center mb-10 text-foreground">Key Technical Innovations</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
              {[
                "Parallel inference pipelines for independent domain evaluation",
                "Strict data isolation architecture to prevent cross-domain feature leakage",
                "Item Response Theory (GRM) for nuanced latent trait measurement",
                "Monotonic XGBoost modeling (Ordinal & Quantile) for occupational health",
                "Dual Random Forest regressors isolating distinct behavioral cross-impacts",
                "Multinomial Logistic Regression paired with Isolation Forests for cluster analysis",
                "Unsupervised K-Means clustering mapping distinct mood and sleep phenotypes",
                "Rigorous deterministic scoring aligned with validated psychometric scales",
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.05 }}
                  className="flex items-start gap-3 p-4 rounded-xl bg-card border border-border/60 shadow-sm hover:border-primary/20 transition-colors"
                >
                  <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                    <div className="w-2 h-2 rounded-full bg-primary" />
                  </div>
                  <span className="text-sm font-medium text-foreground">{item}</span>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Team */}
        <div className="page-container-medium py-24">
          <h2 className="text-2xl font-display font-bold text-center mb-3 text-foreground">Research Team</h2>
          <p className="text-center text-muted-foreground mb-12 max-w-lg mx-auto">
            Under the guidance of Prof. Manju Biswas & Dr. Sourav Banerjee
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 sm:gap-8 max-w-3xl mx-auto">
            {team.map((member, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 15 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.1 }}
              >
                <Card className="card-surface-hover text-center h-full pt-8">
                  <CardContent className="p-6">
                    <div className="w-28 h-28 rounded-full bg-secondary/80 flex items-center justify-center mx-auto mb-5 border-4 border-background shadow-lg overflow-hidden relative">
                      {/* Photo Placeholder */}
                      <img 
                        src={member.photoPath} 
                        alt={member.name} 
                        className={`w-full h-full relative z-10 ${member.fitClass || "object-cover object-top"}`} 
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }} 
                      />
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-primary/5 text-muted-foreground/60 z-0">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mb-1"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                        <span className="text-[10px] font-medium uppercase tracking-widest">Photo</span>
                      </div>
                    </div>
                    <h3 className="font-semibold text-base text-foreground">{member.name}</h3>
                    <p className="text-sm text-muted-foreground mt-1 font-mono bg-secondary/50 inline-block px-2 py-0.5 rounded-md">{member.id}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="text-center px-4">
          <Link to="/assessment">
            <Button size="lg" className="rounded-full px-10 text-base h-12 gap-2 shadow-md hover:shadow-lg transition-shadow">
              Try the Assessment <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>
        </div>
      </main>
      <FooterSection />
    </div>
  );
}