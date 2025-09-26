import { Controller, Get, Header, Req } from '@nestjs/common';
import type { Request } from 'express';

// Simple in-memory counters for demo metrics
let requestCount = 0;
let startTime = Date.now();

@Controller('/metrics')
export class MetricsController {
  @Get()
  @Header('Content-Type', 'text/plain; version=0.0.4')
  getMetrics(@Req() req: Request): string {
    // increment a simple counter
    requestCount += 1;
    const up = 1;
    const uptimeSeconds = Math.floor((Date.now() - startTime) / 1000);

    // Minimal Prometheus exposition format
    // See: https://prometheus.io/docs/instrumenting/exposition_formats/
    const lines: string[] = [];
    lines.push('# HELP app_up 1 if the service is up');
    lines.push('# TYPE app_up gauge');
    lines.push(`app_up ${up}`);

    lines.push('# HELP app_request_total Total number of metric scrapes');
    lines.push('# TYPE app_request_total counter');
    lines.push(`app_request_total ${requestCount}`);

    lines.push('# HELP app_uptime_seconds Service uptime in seconds');
    lines.push('# TYPE app_uptime_seconds gauge');
    lines.push(`app_uptime_seconds ${uptimeSeconds}`);

    return lines.join('\n') + '\n';
  }
}
