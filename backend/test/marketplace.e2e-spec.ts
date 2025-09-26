import { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import * as request from 'supertest';
import * as fs from 'fs';
import * as path from 'path';
import { AppModule } from '../src/app.module';

describe('Marketplace E2E', () => {
  let app: INestApplication;

  beforeAll(async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleRef.createNestApplication();
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  it('GET /api/marketplace/list returns ok and items array', async () => {
    const res = await request(app.getHttpServer()).get('/api/marketplace/list');
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('ok', true);
    expect(Array.isArray(res.body.items)).toBe(true);
  });

  it('POST /api/marketplace/upload accepts a small file and returns item', async () => {
    const tmpFile = path.join(__dirname, 'tmp-upload.txt');
    fs.writeFileSync(tmpFile, 'hello');
    const res = await request(app.getHttpServer())
      .post('/api/marketplace/upload')
      .field('name', 'echo-api')
      .field('runtime', 'python:3.12')
      .attach('file', tmpFile);

    expect(res.status).toBe(201);
    expect(res.body).toHaveProperty('ok', true);
    expect(res.body).toHaveProperty('item');
    expect(res.body.item.name).toBe('echo-api');

    fs.unlinkSync(tmpFile);
  });
});
