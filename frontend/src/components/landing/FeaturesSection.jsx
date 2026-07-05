import React from "react";
import { Card, CardContent } from "../ui/card";
import { Brain, ShieldCheck, BarChart3, Layers, Eye, Cpu } from "lucide-react";
import { motion } from "framer-motion";

const features = [
  {
    icon: Layers,
    title: "Parallel Domain Architecture",
    description: "Evaluates six core psychological domains independently without cross-contamination.",
  },
  {
    icon: Brain,
    title: "Multi-Model Orchestration",
    description: "Utilizes XGBoost, Random Forest, K-Means, and Multinomial Logit tailored to each domain.",
  },
  {
    icon: ShieldCheck,
    title: "Psychometric Validity",
    description: "Uses validated Item Response Theory (GRM) to measure underlying personality traits.",
  },
  {
    icon: Eye,
    title: "Strict Data Isolation",
    description: "Enforces rigid boundaries, ensuring answers from one domain don't artificially inflate another.",
  },
  {
    icon: BarChart3,
    title: "Global Synthesis",
    description: "Compiles all domain-specific risk scores into a unified, easy-to-read clinical summary.",
  },
  {
    icon: Cpu,
    title: "Robust Outlier Detection",
    description: "Automatically flags inconsistent or anomalous response patterns to maintain data integrity.",
  },
];

export default function FeaturesSection() {
  return (
    <section className="page-shell bg-secondary/20">
      <div className="page-container">
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="section-heading">Framework Architecture</h2>
          <p className="section-subheading-center">
            MINDSIGHT integrates advanced machine learning with established clinical psychology methodologies.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
          {features.map((feature, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 15 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.1 }}
            >
              <Card className="h-full card-surface-hover group border-border/50">
                <CardContent className="p-6 sm:p-8">
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-5 group-hover:bg-primary/20 transition-colors">
                    <feature.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground mb-3">{feature.title}</h3>
                  <p className="text-muted-foreground text-sm leading-relaxed">{feature.description}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}