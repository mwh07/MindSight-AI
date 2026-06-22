import React from "react";
import { motion } from "framer-motion";
import { 
  Lightbulb,
  Heart,
  Brain,
  Shield,
  Moon,
  Users,
  Briefcase,
  Stethoscope,
  ArrowRight,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";

const SEVERITY_BADGE = {
  low: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
  moderate: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
  high: "bg-rose-100 text-rose-700 dark:bg-rose-900 dark:text-rose-300",
  normal: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
  mild: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
  severe: "bg-rose-100 text-rose-700 dark:bg-rose-900 dark:text-rose-300",
};

const SEVERITY_ICON = {
  low: {
    container: "bg-emerald-100 dark:bg-emerald-900/30",
    icon: "text-emerald-600 dark:text-emerald-400",
  },
  normal: {
    container: "bg-emerald-100 dark:bg-emerald-900/30",
    icon: "text-emerald-600 dark:text-emerald-400",
  },
  moderate: {
    container: "bg-amber-100 dark:bg-amber-900/30",
    icon: "text-amber-600 dark:text-amber-400",
  },
  mild: {
    container: "bg-amber-100 dark:bg-amber-900/30",
    icon: "text-amber-600 dark:text-amber-400",
  },
  high: {
    container: "bg-rose-100 dark:bg-rose-900/30",
    icon: "text-rose-600 dark:text-rose-400",
  },
  severe: {
    container: "bg-rose-100 dark:bg-rose-900/30",
    icon: "text-rose-600 dark:text-rose-400",
  },
};

function resolveSeverityKey(severity) {
  const key = severity?.toLowerCase();
  if (key && SEVERITY_BADGE[key]) return key;
  return "moderate";
}

const ICON_MAP = {
  personality: Brain,
  self_esteem: Shield,
  mood_sleep: Moon,
  digital_and_social: Users,
  occupational_burnout: Briefcase,
  severe_clinical: Stethoscope,
};

export default function RecommendationCard({
  domain,
  score,
  severity = "moderate",
  recommendations = [],
  className = "",
}) {
  const Icon = ICON_MAP[domain] || Heart;
  const severityKey = resolveSeverityKey(severity);
  const severityBadge = SEVERITY_BADGE[severityKey];
  const severityIcon = SEVERITY_ICON[severityKey];
  const severityLabel = severity?.charAt(0).toUpperCase() + severity?.slice(1) || "Moderate";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className={`border-border/50 hover:shadow-lg transition-all ${className}`}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${severityIcon.container}`}>
                <Icon className={`w-5 h-5 ${severityIcon.icon}`} />
              </div>
              <div>
                <CardTitle className="text-base font-display">
                  {domain.replace(/_/g, " ").replace("domain ", "").toUpperCase()}
                </CardTitle>
                <div className="flex items-center gap-2 mt-0.5">
                  <Badge className={severityBadge}>
                    {severityLabel}
                  </Badge>
                  {score !== undefined && (
                    <span className="text-xs text-muted-foreground">
                      Score: {typeof score === 'number' ? score.toFixed(1) : score}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <Lightbulb className="w-5 h-5 text-amber-500" />
          </div>
        </CardHeader>
        <CardContent>
          {recommendations.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">Recommendations:</p>
              <ul className="space-y-1.5">
                {recommendations.map((rec, index) => (
                  <motion.li
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-start gap-2 text-sm"
                  >
                    <ArrowRight className="w-3 h-3 text-primary mt-1 flex-shrink-0" />
                    <span>{rec}</span>
                  </motion.li>
                ))}
              </ul>
            </div>
          )}
          {recommendations.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No specific recommendations at this time. Continue monitoring your well-being.
            </p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}