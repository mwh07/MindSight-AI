import React from "react";
import { Card, CardContent } from "../ui/card";
import { Brain, Shield, Moon, Users, Briefcase, Stethoscope } from "lucide-react";
import { motion } from "framer-motion";

const dimensions = [
  { icon: Brain, title: "Personality", desc: "Big Five traits assessment via Graded Response Model (GRM)", color: "bg-indigo-500/10 text-indigo-500" },
  { icon: Shield, title: "Self-Esteem", desc: "Demographic norm-referenced scoring for clinical self-worth", color: "bg-sky-500/10 text-sky-500" },
  { icon: Moon, title: "Mood & Sleep", desc: "Unsupervised K-Means clustering identifying distinct phenotypes", color: "bg-emerald-500/10 text-emerald-500" },
  { icon: Users, title: "Digital & Social Risk", desc: "Cross-impact analysis for depression risk via Random Forest", color: "bg-purple-500/10 text-purple-500" },
  { icon: Briefcase, title: "Occupational Burnout", desc: "Monotonic ordinal and quantile mapping via XGBoost", color: "bg-amber-500/10 text-amber-500" },
  { icon: Stethoscope, title: "Clinical Screening", desc: "Multinomial Logit classification & Anomaly Detection", color: "bg-rose-500/10 text-rose-500" },
];

export default function DimensionsSection() {
  return (
    <section className="page-shell">
      <div className="page-container">
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="section-heading">6 Psychological Domains</h2>
          <p className="section-subheading-center">
            Comprehensive assessment across all major mental health vectors using clinically validated instruments.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 sm:gap-6">
          {dimensions.map((dim, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.96 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.05 }}
            >
              <Card className="card-surface-hover border-border/50 group h-full">
                <CardContent className="p-6 flex items-start gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${dim.color} bg-opacity-50 group-hover:bg-opacity-100 transition-all`}>
                    <dim.icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground text-base mb-1">{dim.title}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">{dim.desc}</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}