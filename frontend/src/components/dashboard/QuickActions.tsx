import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function QuickActions() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Quick Actions</CardTitle>
      </CardHeader>
      <CardContent className="flex gap-2">
        <Button>Deploy Worker</Button>
        <Button variant="outline">Run Backup</Button>
        <Button>View Metrics</Button>
      </CardContent>
    </Card>
  );
}

// TEST: Renders buttons
