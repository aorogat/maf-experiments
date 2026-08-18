# Pinned experiment versions

Canonical pin file for the paper footnote (“Versions are pinned in the artifact”):

**[`artifacts/experiment_versions.json`](../artifacts/experiment_versions.json)**

## AutoGen

| Distribution | Version | Role |
|---|---|---|
| `autogen-agentchat` | 0.7.5 | Evaluated AutoGen API (all experiments) |
| `autogen-core` | 0.7.5 | Dependency of agentchat |
| `autogen-ext` | 0.7.5 | Model clients / Chroma memory |

Install: `pip install -r requirements-autogen.lock`

**Do not install AG2** (`ag2` / `pyautogen`, import name `autogen`). MASBench does not import it. If both stacks are present, reviewers cannot tell which line was evaluated.
