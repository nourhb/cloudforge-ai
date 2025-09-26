import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function AIInsights() {
  const insights = [
    'Predicted CPU spike in 10 minutes. Consider scaling to 3 replicas.',
    'Anomaly detected in error logs of backend service.',
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="list-disc pl-5 text-sm">
          {insights.map((i, idx) => (
            <li key={idx}>{i}</li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

// TEST: Renders insights
