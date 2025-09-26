import * as React from 'react';
import { ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';

const sample = Array.from({ length: 20 }, (_, i) => ({ t: i, cpu: 20 + Math.round(Math.sin(i / 2) * 10) }));

export function MetricsChart({ metrics }: { metrics?: any }) {
  const data = metrics?.cpuSeries || sample;
  return (
    <div className="rounded-xl border p-4">
      <div className="mb-2 text-sm font-medium">CPU Utilization</div>
      <div style={{ width: '100%', height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Line type="monotone" dataKey="cpu" stroke="#3b82f6" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// TEST: Renders recharts line
