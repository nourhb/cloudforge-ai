import { Body, Controller, HttpException, HttpStatus, Post } from '@nestjs/common';
import axios from 'axios';

@Controller('/api/iac')
export class IacController {
  @Post('generate')
  async generate(@Body() body: { prompt: string }) {
    try {
      const prompt = body?.prompt || '';
      const base = process.env.AI_URL || 'http://127.0.0.1:5001';
      const client = axios.create({ timeout: 2500 });
      let res;
      try {
        res = await client.post(`${base}/ai/iac/generate`, { prompt });
      } catch (e) {
        // single retry in case of transient startup delays
        res = await client.post(`${base}/ai/iac/generate`, { prompt });
      }
      return res.data;
    } catch (e: any) {
      // Fallback minimal YAML to avoid blocking the flow when AI service is down
      const yaml = [
        'apiVersion: v1',
        'kind: Service',
        'metadata:',
        '  name: backend',
        '  labels:',
        '    app: cloudforge-ai',
        '    component: backend',
        'spec:',
        '  type: ClusterIP',
        '  selector:',
        '    app: cloudforge-ai',
        '    component: backend',
        '  ports:',
        '    - name: http',
        '      port: 4000',
        '      targetPort: 4000',
      ].join('\n');
      return { ok: true, yaml, fallback: true };
    }
  }
}
