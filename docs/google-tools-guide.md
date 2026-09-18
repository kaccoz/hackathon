# Using the supplied Google tools

## What the participant pack confirms

The pack confirms:

- A team-specific Google Cloud environment and project.
- A Google workshop on the Gemini Enterprise suite.
- Google Antigravity access instructions for registered workshop participants.

The pack does not explicitly guarantee a particular AI Studio model, Stitch entitlement, quota, GPU, or Vertex AI service. Verify these after logging into the assigned project and keep the solution functional without them.

## Recommended division of responsibility

| Google tool | Use it for | Do not depend on it for |
|---|---|---|
| Gemini in AI Studio | Explaining unfamiliar code, drafting tests, reviewing functions, UI copy, and prototyping small app components | Producing the official sensor predictions |
| Antigravity | Repository-aware coding assistance, issue implementation, and code review during the workshop | A step that only one teammate knows how to reproduce |
| Stitch | Rapid wireframes and interaction prototypes | The final trained model or data pipeline |
| Cloud Run | Hosting the final Streamlit/web prototype | Training against the full 5.5 GB Rail dataset on every request |
| Cloud Storage | Optional shared storage for approved datasets and model artifacts | Source control or credential storage |
| Vertex AI Workbench/Colab | Experiments if the local machine is constrained | A requirement for running final inference |

## Recommended AI model usage

- Use a stable Flash model available in the supplied environment for routine coding, summarization, and UI text. At the time of planning, Gemini 3.8 Flash is Google's current stable long-horizon coding/agent model.
- Use Gemini 3.1 Pro Preview only for difficult one-off reasoning if it is available. Preview availability can change, so do not make it a runtime dependency.
- Give the model small functions, schemas, error messages, or aggregated features—not the entire 5.5 GB Rail folder.
- Ask for tests and explanations together with generated code.
- Run and review every generated change before merging.

## Rail-specific use

Good prompts:

```text
Explain this feature-extraction function to a beginner. Check whether odd axle-box
positions are consistently treated as Side I and even positions as Side II. Identify
test cases I should add, but do not change the official output schema.
```

```text
Here is a cross-validation confusion matrix and a table of aggregated feature values.
Suggest three testable hypotheses for why Side I is being classified as Normal.
Do not infer anything from the hidden test set.
```

Avoid:

```text
Here are 129 columns of raw numbers. Guess whether this is Side I or Side II.
```

The LLM lacks a validated learned mapping from these organizer signals to the official classes. The scikit-learn pipeline should make the prediction.

## Cloud deployment order

1. Confirm the assigned project and permissions during the workshop.
2. Deploy a minimal Streamlit hello-world service to Cloud Run early.
3. Connect one saved subsystem model, preferably Rail.
4. Confirm file upload and CSV download through the hosted URL.
5. Add other subsystems one at a time.
6. Keep the trained model inside the deployment artifact or load it from a controlled bucket.
7. Do not copy the 5.5 GB training set into the runtime image; inference only needs the saved model and an uploaded file.
8. Test the hosted URL in an incognito browser before recording the video.

## Security and reliability

- Never commit cloud credentials, API keys, team passkeys, or `.env` files.
- Do not call Gemini directly from browser-side JavaScript with an exposed API key.
- Set a spending/quota alert if the project permits it.
- Preserve a fully local CSV-generation command in case the hosted demo is unavailable.
- Record the demo only after the deployed application and downloads work end to end.

