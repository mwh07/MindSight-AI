import React from "react";
import { Link } from "react-router-dom";
import { Button } from "../ui/button";
import { ArrowRight, Shield, Brain, Activity, CheckCircle } from "lucide-react";
import { motion } from "framer-motion";
import { Card } from "../ui/card";

export default function HeroSection() {
  return (
    <section className="relative min-h-[90vh] flex flex-col items-center justify-center pt-24 pb-16">
      <div className="page-container relative z-10 px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="flex justify-center mb-6"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary text-secondary-foreground border border-border/50 text-sm font-medium shadow-sm">
              <Shield className="w-4 h-4 text-primary" />
              <span>Academic Research Project</span>
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1, ease: "easeOut" }}
            className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-display font-bold leading-[1.1] tracking-tight text-foreground"
          >
            Clinical-Grade <br className="hidden sm:block" />
            <span className="text-primary">Mental Health</span> Assessment
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" }}
            className="mt-6 text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed"
          >
            MINDSIGHT leverages a Unified Multi-Domain Psychological Diagnostic Profiling System to provide comprehensive, 
            multidimensional psychological insights with rigorous reliability.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3, ease: "easeOut" }}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link to="/assessment">
              <Button size="lg" className="rounded-full px-8 text-base h-12 gap-2 shadow-md">
                Start Assessment <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
            <Link to="/about">
              <Button variant="outline" size="lg" className="rounded-full px-8 text-base h-12 bg-background/50 backdrop-blur-sm">
                View Methodology
              </Button>
            </Link>
          </motion.div>

          {/* Academic Trust Indicators */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4, ease: "easeOut" }}
            className="mt-16 sm:mt-20 max-w-3xl mx-auto"
          >
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
              {[
                { icon: Brain, label: "6 Psychological Domains" },
                { icon: Activity, label: "70-Item Clinical Battery" },
                { icon: CheckCircle, label: "v3.8 Architecture" },
              ].map((stat, i) => (
                <Card key={i} className="card-surface p-4 flex items-center justify-center gap-3 bg-card/60 backdrop-blur-sm border-border/40">
                  <stat.icon className="w-5 h-5 text-primary opacity-80" />
                  <span className="text-sm font-medium text-foreground">{stat.label}</span>
                </Card>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}