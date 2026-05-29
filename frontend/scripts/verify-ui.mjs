import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd(), '..');
const outputDir = path.join(root, '.ely_eye', 'proofs');
await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const consoleErrors = [];
page.on('console', (message) => {
  if (message.type() === 'error') {
    consoleErrors.push(message.text());
  }
});

await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
await page.getByRole('heading', { name: 'Local VLM Context OS' }).waitFor();
await page.getByLabel('system status').getByText('Qwen/Qwen3.5-9B').waitFor();
await page.getByRole('heading', { name: 'Adapter Matrix' }).waitFor();
await page.getByRole('heading', { name: 'PRD Proof Suite' }).waitFor();
await page.getByText(/\d+\/\d+ passed/).waitFor();
await page.screenshot({ path: path.join(outputDir, 'ely-eye-dashboard.png'), fullPage: true });
const status = await page.locator('.kpi-grid').innerText();
await browser.close();

if (consoleErrors.length > 0) {
  throw new Error(consoleErrors.join('\n'));
}

console.log(JSON.stringify({ screenshot: path.join(outputDir, 'ely-eye-dashboard.png'), status }, null, 2));
