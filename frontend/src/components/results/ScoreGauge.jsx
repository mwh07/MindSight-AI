import React from "react";
import { motion } from "framer-motion";

export default function ScoreGauge({ 
  value, 
  maxValue = 10, 
  label, 
  sublabel, 
  color = "#6366f1",
  size = 120,
  showValue = true,
  className = ""
}) {
  const percentage = Math.min((value / maxValue) * 100, 100);
  const center = size / 2;
  const radius = size / 2 - 10;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <div className={`flex flex-col items-center ${className}`}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="transform -rotate-90" width={size} height={size}>
          <circle
            cx={center}
            cy={center}
            r={radius}
            stroke="hsl(var(--secondary))"
            strokeWidth="8"
            fill="none"
            opacity="0.3"
          />
          <motion.circle
            cx={center}
            cy={center}
            r={radius}
            stroke={color}
            strokeWidth="8"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.5, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {showValue && (
            <motion.span
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
              className="text-2xl font-bold"
              style={{ color }}
            >
              {typeof value === 'number' ? value.toFixed(1) : value}
            </motion.span>
          )}
          {label && (
            <span className="text-xs text-muted-foreground mt-0.5 text-center px-2">
              {label}
            </span>
          )}
        </div>
      </div>
      {sublabel && (
        <span className="text-sm font-medium mt-2">{sublabel}</span>
      )}
    </div>
  );
}