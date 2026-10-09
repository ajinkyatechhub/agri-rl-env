# 🌱 agri-rl-env

**Agriculture RL environment built with Scale AI's [AgentEnv Framework](https://www.agentenvframework.com/).**

An AI agent works on a small farm: it reads the spray rule book, inspects crop fields, and decides for each field whether to **spray** or mark it **healthy**. An AI judge then grades every decision against an answer key, and the result is a **reward score** between 0 and 1.

> ✅ **Status:** working end-to-end — first successful run scored **1.0 / 1.0**.

---

## 📖 What is an RL environment?

An RL (Reinforcement Learning) environment is a **practice world** where an AI tries a task and receives a **score (reward)** that says how well it did. AI labs run environments like this thousands of times and use the rewards to train better models.

| RL concept | In this project |
|---|---|
| **Environment / world** | `farm/server.py` — the farm |
| **State** | 3 fields with crop, disease and severity (%) |
| **Actions** | 5 tools the agent can call |
| **Agent** | `student/` — an AI that calls the tools |
| **Reward** | `teacher/` — an AI judge that scores the work |

---

## 🧪 The task

The agent must make exactly **one decision per field**, following the rule book:

> *Spray a field only if it has a disease **and** the disease is **10% or more**.*

| Field | Crop | Disease | Severity | Correct action |
|---|---|---|---|---|
| F-1 | Tomato | Early blight | 30% | **Spray** |
| F-2 | Cotton | — | 0% | **Healthy** |
| F-3 | Chilli | Powdery mildew | 5% | **Healthy** 🪤 *(trap: has a disease, but below 10%)* |

### Scoring rubric

| Criterion | Weight |
|---|---|
| F-1 was sprayed | +1 |
| F-2 was marked healthy | +1 |
| F-3 was marked healthy (keep watching) | +1 |
| A field was sprayed that must not be sprayed | **−2** (penalty) |

The final score is a weighted average, so a perfect run scores **1.0**.

---

## 🏗️ Architecture

```
                    ┌──────────────────────── AgentEnv task (task.json) ───────────────────────┐
                    │                                                                          │
  data/farm.json ──►│ 1. deploy_env ──► 2. load_artifact ──► 3. deploy_agent ──► 4. prompt_agent│
                    │      🏫 farm          📋 fields+rules       🧒 student         ✍️ works     │
                    │                                                                  │       │
                    │                         6. teardown ◄── 5. rubrics_verifier ◄────┘       │
                    │                            🧹 clean         👩‍🏫 teacher → ⭐ score          │
                    └──────────────────────────────────────────────────────────────────────────┘

  🧒 student ──(MCP tool calls)──► 🚪 gateway ──► 🏫 farm server
  🧒 student / 👩‍🏫 teacher ──(OpenAI-compatible API)──► 🤖 Google Gemini
```

Every part runs in its own **Docker container** on the local machine.

---

## 📂 Project structure

```
agri-rl-env/
├── .agentenv/
│   └── config.toml        # where the AI model lives (Gemini endpoint); key comes from the shell
├── data/
│   └── farm.json          # the world's data: rule book + 3 fields
├── farm/
│   ├── server.py          # the environment: data plane + 5 MCP tools
│   └── Dockerfile
├── student/
│   ├── agent.py           # the agent: tool-calling loop, records its trajectory
│   └── Dockerfile
├── teacher/
│   ├── agent.py           # the judge: reads the trajectory, grades against the rubric
│   └── Dockerfile
├── task.json              # the exam: 6 steps + the scoring rubric
└── README.md
```

### Tools the agent can use

| Tool | What it does |
|---|---|
| `farm_read_rules` | Read the spray rule book |
| `farm_list_fields` | List all fields and their status |
| `farm_check_field` | Inspect one field's disease and severity |
| `farm_spray` | Spray a field (a reason note is required) |
| `farm_mark_healthy` | Mark a field as healthy (a reason note is required) |

The world enforces its own rules: it refuses unknown fields, a second decision on the same field, and actions without a note.

---

## 🛠️ Tech stack

| Layer | Technology |
|---|---|
| RL framework | [AgentEnv Framework](https://github.com/scaleapi/agentenv-framework) (Scale AI) |
| Language | Python 3.12 |
| Containers | Docker Desktop |
| Tool protocol | MCP (Model Context Protocol) |
| Agent protocol | A2A (Agent-to-Agent) |
| AI model | Google Gemini via its OpenAI-compatible API (`gemini-3.5-flash-lite`, free tier) |
| Python libraries | `agentenv-framework-protocol`, `mcp`, `openai` |
| Dev machine | MacBook (Apple Silicon M5), VS Code, `uv` |

---

## ✅ Prerequisites

- macOS on Apple Silicon (instructions below use `--platform linux/arm64`; remove it on Intel)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — running
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- AgentEnv — `uv tool install agentenv-framework`
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/)

Check the install:

```bash
agent-env run hello        # should print "passed"
```

---

## 🚀 How to run

Run everything from inside the `agri-rl-env` folder.

### 1. Set the API key (every new terminal)

```bash
export litellm_api_key=YOUR_GEMINI_KEY
```

> 🔐 Never put the key in a file or commit it. `config.toml` only references it as `secret:litellm_api_key`.

### 2. Register AgentEnv's built-in helpers (once per machine)

```bash
agent-env env service-db put --id default-db --platform linux/arm64
agent-env env gateway put --id default --platform linux/arm64
```

### 3. Register the environment and both agents

```bash
agent-env env mcp-server put --id farm --dockerfile farm/Dockerfile --platform linux/arm64
agent-env a2a-agent put --id student --dockerfile student/Dockerfile --skip-validation --platform linux/arm64
agent-env a2a-agent put --id teacher --dockerfile teacher/Dockerfile --skip-validation --platform linux/arm64
```

### 4. Register the data and the task

```bash
agent-env artifact environment put --id farm-data --environment-name farm \
  --description "Tiny farm: 3 fields and the spray rule book" data/farm.json
agent-env artifact environment-universe put --id farm-world --environment-artifact farm-data
agent-env task create task.json --id farm-exam
```

### 5. Run the exam

```bash
agent-env task run --id farm-exam --output-dir out
```

Expected end of the output:

```
score (verifier_id=farm-marks): 1.0
Task completed!
```

### 6. Read the results

```bash
grep -E '"(id|score|result)"' out/farm-exam_*.json
```

### What to re-run after a change

| You changed | Re-run |
|---|---|
| `farm/server.py` | step 3 (first line), then step 5 |
| `data/farm.json` | step 4 (first two lines), then step 5 |
| `task.json` | step 4 (last line), then step 5 |
| `student/agent.py` / `teacher/agent.py` | the matching line of step 3, then step 5 |

Each `put` creates a new **version**; runs use the latest version, and old versions stay available.

---

## 📊 Results

| Run | Model | Tool calls | Score |
|---|---|---|---|
| First successful run | `gemini-3.5-flash-lite` | 8 | **1.0** |

| Criterion | Result |
|---|---|
| F-1 sprayed | ✅ |
| F-2 marked healthy | ✅ |
| F-3 marked healthy (trap) | ✅ |
| No wrong spray | ✅ |

---

## 🧯 Troubleshooting (lessons learned)

| Problem | Cause | Fix |
|---|---|---|
| `local registry ... did not answer /v2/ on port 5000` | macOS **AirPlay Receiver** uses port 5000 | System Settings → General → AirDrop & Handoff → turn **AirPlay Receiver** off, then remove the stuck registry container |
| `A2A agent ... did not serve /.well-known/agent.json within 300s` | Agent file was empty/unsaved when the image was built | Save the file (turn on VS Code **Auto Save**), re-run the `a2a-agent put` |
| `Failed to parse judge response as JSON` | The model wrapped its JSON in ```` ```json ```` fences | `teacher/agent.py` strips code fences before returning |
| `429` / `GenerateRequestsPerDayPerProjectPerModel-FreeTier` | Gemini free tier allows ~20 requests/day per model (one run ≈ 10–14 requests) | Wait for the daily reset, switch to another free model, or use a paid key / local model |
| `Missing option '--description'` / `No such option '--project-id'` | CLI options differ between AgentEnv versions | Check `agent-env <command> --help` |
| `exec format error` | Image built for the wrong CPU | Add `--platform linux/arm64` on Apple Silicon |

---

## 🗺️ Roadmap

- [x] Basic environment: 3 fields, 5 tools, AI judge — **score 1.0**
- [ ] Edge-case trap: a field at exactly 10% severity
- [ ] Weather tool + "don't spray before rain" rule
- [ ] Code-based end-state verifier (`env_outcome_verifier`) alongside the AI judge
- [ ] Virtual clock and triggers (changing weather mid-run)
- [ ] Realistic synthetic agriculture data (farmers, dealers, products, stock)
- [ ] Multiple apps composed into one world (farm + weather + SMS)

---

## 🙏 Acknowledgements

- [AgentEnv Framework](https://www.agentenvframework.com/) by Scale AI (Apache-2.0)
- [Google Gemini API](https://ai.google.dev/)

## 👤 Author

**Sushant Shere**
