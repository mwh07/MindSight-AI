import React from "react";
import { motion } from "framer-motion";

export default function ProgressBar({ current, total, sectionName }) {
  const percentage = Math.round((current / total) * 100);

  return (
    <div className="mb-8">
      <div className="flex items-end justify-between mb-2">
        <div>
          <span className="text-sm font-semibold tracking-wide text-foreground uppercase">{sectionName}</span>
          <div className="text-xs font-medium text-muted-foreground mt-0.5">{current} of {total} questions answered</div>
        </div>
        <span className="text-sm font-bold text-primary bg-primary/10 px-2 py-0.5 rounded-md">{percentage}%</span>
      </div>
      <div className="h-2.5 bg-secondary/80 rounded-full overflow-hidden shadow-inner">
        <motion.div
          className="h-full bg-primary rounded-full relative"
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <div className="absolute top-0 right-0 bottom-0 left-0 bg-white/20" style={{ clipPath: 'polygon(0 0, 100% 0, 95% 100%, 0% 100%)' }} />
        </motion.div>
      </div>
    </div>
  );
}