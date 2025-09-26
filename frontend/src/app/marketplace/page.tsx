"use client";

import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function MarketplacePage() {
  const workers = [
    { id: 1, name: 'echo-api', description: 'Echoes your input', version: '1.0.0' },
    { id: 2, name: 'cpu-forecast', description: 'Forecast CPU usage', version: '1.0.0' },
  ];
  return (
    <div className="mx-auto max-w-6xl p-6">
      <h1 className="text-3xl font-bold mb-6">API Marketplace</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {workers.map(w => (
          <Card key={w.id}>
            <CardHeader>
              <CardTitle>{w.name}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 mb-2">{w.description}</p>
              <p className="text-xs text-gray-500">v{w.version}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

// TEST: Static list renders
