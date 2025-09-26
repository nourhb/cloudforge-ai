# Frontend Dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./
RUN npm ci || yarn install --frozen-lockfile || pnpm i --frozen-lockfile
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "start"]

# TEST: Builds Next.js 15.5.4 image
