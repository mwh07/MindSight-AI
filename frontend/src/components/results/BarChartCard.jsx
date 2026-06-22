import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

const CHART_COLORS = [
  "#6366f1",
  "#8b5cf6",
  "#a855f7",
  "#d946ef",
  "#ec4899",
  "#f43f5e",
  "#fb7185",
  "#f97316",
  "#f59e0b",
  "#eab308",
];

export default function BarChartCard({
  data,
  title,
  description,
  dataKey = "value",
  nameKey = "name",
  layout = "vertical",
  color = "#6366f1",
  height = 300,
  className = "",
  barSize = 30,
  useColors = false,
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
          <BarChart 
            data={data} 
            layout={layout}
            margin={{ top: 10, right: 10, left: layout === 'vertical' ? 80 : 10, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            {layout === 'vertical' ? (
              <>
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis 
                  dataKey={nameKey} 
                  type="category" 
                  tick={{ fontSize: 11 }}
                  width={80}
                />
              </>
            ) : (
              <>
                <XAxis dataKey={nameKey} tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
              </>
            )}
            <Tooltip 
              contentStyle={{ 
                backgroundColor: "hsl(var(--card))", 
                border: "1px solid hsl(var(--border))",
                borderRadius: "8px",
                fontSize: "12px"
              }}
            />
            <Bar 
              dataKey={dataKey} 
              fill={color} 
              barSize={barSize}
              radius={[4, 4, 0, 0]}
            >
              {useColors && data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={CHART_COLORS[index % CHART_COLORS.length]} 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
} 