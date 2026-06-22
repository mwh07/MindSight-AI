import React from "react";
import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

export default function QuestionCard({ question, scale, scaleValues, value, onChange, index }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="bg-card rounded-xl border border-border/50 p-5 hover:border-primary/20 transition-all"
    >
      <p className="text-sm font-medium leading-relaxed mb-4">
        <span className="text-primary font-semibold mr-2">Q{index + 1}.</span>
        {question.text}
      </p>
      <div className="flex flex-wrap gap-2">
        {scale.map((label, i) => (
          <button
            key={i}
            onClick={() => onChange(question.id, scaleValues[i])}
            className={cn(
              "px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              value === scaleValues[i]
                ? "bg-primary text-primary-foreground border-primary shadow-md shadow-primary/20"
                : "bg-secondary/50 text-muted-foreground border-border hover:border-primary/30 hover:bg-secondary"
            )}
          >
            {label}
          </button>
        ))}
      </div>
    </motion.div>
  );
}