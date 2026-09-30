import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
const require = createRequire(pathToFileURL(process.argv[2]));
const { Tokenizer } = require('@huggingface/tokenizers');
const { serializeState } = await import(new URL('./sequence.js', pathToFileURL(process.argv[2])));
const { Laya } = await import(pathToFileURL(process.argv[2]).href);
const laya = await Laya.load({modelDir: process.argv[3], executionProviders: ['cpu'],
  sessionOptions: {intraOpNumThreads: 4}});
const tokenizer = new Tokenizer(JSON.parse(await readFile(`${process.argv[3]}/tokenizer/tokenizer.json`, 'utf8')),
  JSON.parse(await readFile(`${process.argv[3]}/tokenizer/tokenizer_config.json`, 'utf8')));
console.log(JSON.stringify({ready: true}));
const input = createInterface({input: process.stdin});
for await (const line of input) {
  try {
    const state = JSON.parse(line);
    const stateTokens = tokenizer.encode(serializeState(state), {add_special_tokens: false}).ids.length;
    // Do not let the vendor silently truncate candidates or conflicting OCR evidence.
    if (stateTokens > laya.config.max_len - laya.config.head_max_len - 4) {
      throw new Error(`Evidence exceeds Laya context budget (${stateTokens} state tokens); visual fallback required`);
    }
    const result = await laya.systemOne(state, {membership: {type: 'choice',
      instructions: 'Decide whether the detected product belongs to any target catalog SKU. OCR is imperfect. Similarities are not probabilities. A shared brand alone does not identify a variant. Missing text is not exclusion evidence. Use uncertain when evidence conflicts or is insufficient.',
      criteria: {include: 'Reliable packaging text and visual evidence support a target product.',
        exclude: 'Reliable packaging text identifies a product outside all targets.',
        uncertain: 'Evidence is weak, ambiguous, or conflicting.'}}});
    console.log(JSON.stringify({...result.answers.membership, input_tokens: result.usage.input_tokens}));
  } catch (error) { console.log(JSON.stringify({error: String(error)})); }
}
await laya.close();
