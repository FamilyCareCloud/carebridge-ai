// Validate recorded, actual GPU responses against a checked-out backend contract.
// This script never starts a backend server or writes service data.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const [reportPath, backendPath, outputPath, corpusPath] = process.argv.slice(2);
if (!reportPath || !backendPath || !outputPath) {
  throw new Error('Usage: node scripts/check_backend_contract.mjs REPORT BACKEND_REPO NEW_OUTPUT [CORPUS]');
}
const raw = await readFile(reportPath);
const report = JSON.parse(raw);
if (!report.finished_at) throw new Error('Evaluation is incomplete');
const corpusRaw = await readFile(corpusPath ?? new URL('../evaluation/translation-corpus-v1.json', import.meta.url));
const corpusHash = createHash('sha256').update(corpusRaw).digest('hex');
if (report.corpus_sha256 !== corpusHash) throw new Error('Corpus does not match the evaluated fixture');
const corpus = JSON.parse(corpusRaw.toString('utf8'));
const contractPath = resolve(backendPath, 'src/carebridge-contract.js');
const { validateDraft, validateReview, evidenceSegments } = await import(pathToFileURL(contractPath));
const result = {
  schema: 'carebridge.backend-compatibility.v1',
  evaluated_at: new Date().toISOString(),
  evaluation_sha256: createHash('sha256').update(raw).digest('hex'),
  corpus_sha256: corpusHash,
  backend_revision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: backendPath, encoding: 'utf8' }).trim(),
  backend_contract_sha256: createHash('sha256').update(await readFile(contractPath)).digest('hex'),
  drafts: [], reviews: [], failures: [], skipped_failed_requests: [],
};
const check = (type, id, callback) => {
  try { callback(); result[type].push(id); }
  catch (error) { result.failures.push({ type, id, code: error.code ?? error.name }); }
};
for (const entry of report.translations ?? []) {
  if (entry.http_status !== 200) { result.skipped_failed_requests.push(entry.id); continue; }
  const source = corpus.translation_cases.find(c => c.id === entry.id)?.source;
  if (source === undefined) throw new Error(`Unknown translation case: ${entry.id}`);
  check('drafts', entry.id, () => validateDraft(entry.response, source));
}
for (const entry of report.qa_cases ?? []) {
  if (entry.http_status !== 200) { result.skipped_failed_requests.push(entry.id); continue; }
  const fixture = corpus.qa_cases.find(c => c.id === entry.id);
  const source = corpus.translation_cases.find(c => c.id === fixture?.base_id)?.source;
  if (source === undefined) throw new Error(`Unknown QA case: ${entry.id}`);
  const expected = { translation: { segments: evidenceSegments(source).map(evidence => ({ evidence })) } };
  check('reviews', entry.id, () => validateReview(entry.response, expected));
}
for (const entry of report.workflow_cases ?? []) {
  if (entry.http_status !== 200 || !entry.response.draft) continue;
  const value = entry.response;
  check('drafts', entry.id, () => validateDraft(value.draft, value.draft.source.text));
  if (value.review) check('reviews', entry.id, () => validateReview(value.review, value.draft));
}
if (report.schema === 'carebridge.targeted-audit-evaluation.v1') {
  for (const entry of report.results) {
    if (entry.http_status !== 200) { result.skipped_failed_requests.push(entry.id); continue; }
    const source = entry.draft?.source?.text;
    if (!corpus.translation_cases.some(c => c.source === source)) throw new Error(`Unknown recheck source: ${entry.id}`);
    // Only generated drafts are service outputs. QA request fixtures intentionally
    // contain unchecked/mutated content and are not publishable backend drafts.
    if (entry.kind === 'generated') check('drafts', entry.id, () => validateDraft(entry.draft, source));
    check('reviews', entry.id, () => validateReview(entry.response, entry.draft));
  }
}
if (!result.drafts.length || !result.reviews.length) throw new Error('No responses checked');
result.all_checked_responses_compatible = result.failures.length === 0;
await writeFile(outputPath, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ drafts: result.drafts.length, reviews: result.reviews.length, failures: result.failures, skipped: result.skipped_failed_requests.length }));
process.exitCode = result.failures.length ? 1 : 0;
