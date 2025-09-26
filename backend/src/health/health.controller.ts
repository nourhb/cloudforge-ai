import { Controller, Get } from '@nestjs/common';

@Controller('/health')
export class HealthController {
  @Get()
  getHealth() {
    return { status: 'ok' };
  }
}

// TEST: GET /health returns {status:'ok'}
