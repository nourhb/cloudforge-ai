import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function SystemStatus({ health, isLoading }: { health: any; isLoading: boolean }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>System Status</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="text-sm text-gray-500">Loading...</div>
        ) : (
          <div className="text-sm">
            <div className="flex justify-between">
              <span>API</span>
              <span className={health?.status === 'ok' ? 'text-green-600' : 'text-yellow-600'}>{health?.status || 'unknown'}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// TEST: Displays status
