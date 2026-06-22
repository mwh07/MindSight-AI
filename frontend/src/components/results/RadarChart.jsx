import React from "react";
import {
  Radar,
  RadarChart as RechartsRadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

export default function RadarChart({ 
  data, 
  title, 
  description,
  dataKey = "value",
  nameKey = "trait",
  color = "#6366f1",
  domain = [-1, 1],
  height = 300,
  className = ""
}) {
  return (
    <Card className={`border-border/50 ${className}`}>
      {title && (
        <CardHeader>
          <CardTitle className="text-lg font-display">{title}</CardTitle>
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </CardHeader>
      )}
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <RechartsRadarChart data={data}>
            <PolarGrid stroke="hsl(var(--border))" />
            <PolarAngleAxis 
              dataKey={nameKey} 
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            />
            <PolarRadiusAxis 
              domain={domain} 
              tickCount={5}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            />
            <Radar
              name="Score"
              dataKey={dataKey}
              stroke={color}
              fill={color}
              fillOpacity={0.3}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: "hsl(var(--card))", 
                border: "1px solid hsl(var(--border))",
                borderRadius: "8px",
                fontSize: "12px"
              }}
            />
          </RechartsRadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}