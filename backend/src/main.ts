import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import helmet from 'helmet';
import * as compression from 'compression';
import * as expressWinston from 'express-winston';
import * as winston from 'winston';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, {
    logger: ['log', 'error', 'warn', 'debug', 'verbose'],
  });
  app.use(helmet());
  app.use(compression());
  app.enableCors({ origin: '*'});

  app.use(
    expressWinston.logger({
      transports: [new winston.transports.Console()],
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.timestamp(),
        winston.format.json()
      ),
    })
  );

  await app.listen(4000);
}
bootstrap().catch((err) => {
  // eslint-disable-next-line no-console
  console.error('Bootstrap error', err);
  process.exit(1);
});

// TEST: App boots and listens on 4000
