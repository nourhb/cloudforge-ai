import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function RecentActivity() {
  const items = [
    { id: 1, text: 'Deployed worker echo-api', time: '2m ago' },
    { id: 2, text: 'Backup completed', time: '1h ago' },
    { id: 3, text: 'AI predicted CPU spike', time: '3h ago' },
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Activity</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2 text-sm">
          {items.map(i => (
            <li key={i.id} className="flex justify-between">
              <span>{i.text}</span>
              <span className="text-gray-500">{i.time}</span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

// TEST: Renders list
