# Try decision providers on the same evidence

Jev remains the default. Text adapters are also available for Microsoft
Decision-1 through Microsoft Foundry and OpenAI Decisions with `gpt-6-luna`.
The alternative adapters have API-free contract and pipeline tests; this project
has not yet measured their live driving-scene accuracy, latency or cost.

All three receive the same readable script and lateral/longitudinal definitions.
Numerical facts and camera images remain local. A request contains two independent
Choice questions, or just the lateral question with `--lateral-only`. The sample
and scene formats, context window and merge behavior remain shared.

## Install and configure

```bash
uv sync --frozen --extra decisions
```

Configure credentials in your shell or an ignored local `.env`. See
[.env.example](../.env.example); do not put real keys in a command saved to Git.

| Provider | Configuration | Service |
| --- | --- | --- |
| `jev` | `TYPESAFE_API_KEY`; optional `--model` | TypeSafe System One |
| `microsoft` | `MICROSOFT_DECISION_ENDPOINT`, `MICROSOFT_DECISION_DEPLOYMENT`, and `MICROSOFT_DECISION_API_KEY` or `MICROSOFT_DECISION_ENTRA_TOKEN` | Your existing Microsoft Foundry deployment |
| `openai` | `OPENAI_API_KEY`; model defaults to `gpt-6-luna` | OpenAI Decisions |

For Microsoft, `--endpoint` overrides the resource endpoint and `--model`
overrides the deployment name. The request uses that deployment name; results
also record the underlying model name returned by Foundry. Configure a deployed
resource you can access; the CLI does not create Azure resources or refresh tokens.
Only HTTPS endpoints without embedded credentials, query strings or fragments
are accepted. OpenAI uses its dedicated Decisions endpoint, without a Responses
API fallback. Redirects do not forward credentials.

Sources: [Microsoft Foundry documentation](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision)
and [OpenAI documentation](https://developers.openai.com/api/docs/guides/decisions).
Model availability, input limits, quotas and account access depend on the service.

## Start with a short trial

Use an existing prepared folder with `meta/` and `steps/`. The original input
scripts—including any explicitly reviewed evidence corrections—are reused.

Inspect the configuration and selected sample count without a key or API call:

```bash
uv run --extra decisions jevsceneminer classify out/boston \
  --provider microsoft --from-s 30 --to-s 50 --dry-run
uv run --extra decisions jevsceneminer classify out/boston \
  --provider openai --from-s 30 --to-s 50 --dry-run
```

The Microsoft endpoint and deployment must already be configured, even for its
preview. Character counts are reported; they do not certify that an input fits
the service's token limit. Inputs are never silently truncated.

The following commands **call the paid services**. Run them when you are ready:

```bash
uv run --extra decisions jevsceneminer classify out/boston \
  --provider microsoft --out out/boston-microsoft-trial \
  --from-s 30 --to-s 50 --maneuver-tail 2

uv run --extra decisions jevsceneminer classify out/boston \
  --provider openai --out out/boston-openai-trial \
  --from-s 30 --to-s 50 --maneuver-tail 2
```

Use fresh output directories, outside the source folder. A fork copies prepared
evidence and ancillary indexes, retains compatible caches, and removes old raw
answers from its prepared rows. A bounded trial retains only the requested NOW
samples; each sample keeps its complete original past/future script. Unclassified
time outside a trial remains unknown. Source files and GT are not changed.

For a full run, omit `--from-s` and `--to-s`. Resume an interrupted trial by
classifying its result directory directly. That folder records provider, requested
model/deployment and endpoint; completed answers are cached as they arrive.
When switching provider in an existing run, `--out NEW_FOLDER` is required.

Make a matching Jev reference using existing compatible caches, with no API:

```bash
uv run jevsceneminer classify out/boston --out out/boston-jev-reference \
  --from-s 30 --to-s 50 --cache-only --maneuver-tail 2
```

`--cache-only` fails before requesting inference if any selected cache key is
missing or incompatible. Changing the script, questions, requested model,
provider or Microsoft endpoint can invalidate compatibility. Legacy Jev cache
keys and old result files remain readable.

## Review each model's results

```bash
uv run jevsceneminer view /path/to/boston --gt /path/to/boston-gt \
  --sensor-root /path/to/sensor_blobs --port 8650 \
  --run jev=/path/to/boston \
  --run microsoft=/path/to/boston-microsoft-trial \
  --run openai=/path/to/boston-openai-trial
```

For `view --run`, use absolute paths for result folders outside the base run.
Relative paths are resolved against the base run, as in the existing viewer.
For example, replace the folders above with their absolute paths on your machine.
Whole result folders provide both scenes and raw answers. Legacy scene-only
folders still work; they cannot supply raw probabilities from another run.

Choose a result provider/model in the header. Its merged scene, trajectory colors
and raw probability panels change together. All configured scene bars remain
available for comparison. Camera/BEV and GT editing retain their existing behavior.

## Compare saved runs without inference

```bash
uv run jevsceneminer compare \
  --run jev=out/boston-jev-reference \
  --run microsoft=out/boston-microsoft-trial \
  --run openai=out/boston-openai-trial \
  --gt out/boston-gt --out out/provider-comparison
```

This writes `report.json` and `report.md`. It checks identical readable scripts
at common timestamps, matching cadence, and question-definition hashes when
available. Different question hashes are rejected; older runs without hashes
are explicitly marked unverified. A full reference can be compared with short
trials on their shared timestamps.

Agreement uses only explicitly labeled GT inside those common inference cells.
Unreviewed time is excluded and missing predictions remain unknown. Partial GT
and short trials do not establish full-log precision/recall or true maneuver
boundary accuracy. Use independent, complete evaluation intervals for those
claims; see [evaluation](evaluation.md).

Runtime reports distinguish requested deployment/model, resolved model, fresh
answers, cache reuse, inference wall time and reported token usage. Missing usage
produces unknown cost. Microsoft pricing is unknown by default; configure both
`--input-price` and `--output-price` in USD per million tokens if you know the
applicable rates. Jev 1.x and OpenAI `gpt-6-luna` use dated input-price estimates;
new model IDs need explicit rates. Estimates cover successful fresh responses,
exclude unreported failed-attempt billing, and assume no API cache discount.
Cached runs and different sample sets are not equivalent latency measurements.

The alternatives may have different probability behavior. Evaluate merge
thresholds on held-out driving examples rather than assuming that a general
benchmark winner is best for these scenes.
