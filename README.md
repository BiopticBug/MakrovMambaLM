# MarkovMambaLM

A research prototype combining a **Selective State Space Model (Mamba)** with a **differentiable discrete Markov state mechanism**, built to explore whether forcing information through an explicit discrete bottleneck produces more interpretable internal representations than a plain continuous SSM state — without needing extra probing or clustering tools after the fact.

This is a from-scratch, CUDA-free implementation intended for **development, debugging, and interpretability auditing**, not large-scale training.

---

## What this actually shows (and what it doesn't)

This project was built, debugged, and evaluated carefully, including negative results:

- ✅ **Interpretability**: the discrete Markov states align with known task phases (e.g. "reading input" vs. "waiting" vs. "retrieving") noticeably more cleanly than a k-means clustering of a vanilla Mamba block's own continuous hidden state, across all layers tested.
- ✅ **No performance cost**: models with and without the Markov mechanism reach comparable task accuracy.
- ❌ **No speed advantage**: a promising single-run result suggesting the Markov mechanism accelerates learning did **not replicate** across multiple random seeds — this was tested rigorously and rejected rather than reported as a positive result.
- ❌ **Associative recall (content-addressable memory)** was not solved by either architecture at the scale tested here (~240K–440K parameters, CPU).

See [`docs/findings.md`](docs/findings.md) *(optional, if you keep your own notes there)* for the full experimental trail.

---

## Architecture
<img width="1536" height="1024" alt="Makrov Architecture" src="https://github.com/user-attachments/assets/4db3f929-21ec-454d-9cf6-719dd34c6425" />


**How it works, in short:**

1. **Token + positional embeddings** feed into a stack of `MarkovMambaBlock` layers (pre-norm + residual, matching standard Transformer/Mamba practice).
2. Inside each block, two things happen in parallel:
   - The **`MarkovStateTracker`** looks at the current token and the previous (soft) state, and produces a new discrete state distribution via Gumbel-Softmax — a differentiable stand-in for a hard discrete choice.
   - The **`MambaBlock`** runs its normal selective-scan recurrence.
3. The Markov state's embedding is **injected into the Mamba block right after its causal convolution, before the input-dependent Δ/B/C projections** — this lets the discrete state directly shape the SSM's selectivity at each timestep, rather than just being tacked on as an afterthought.
4. State distributions from every layer are retained and returned, so they can be inspected, logged, or visualized directly — this is the interpretability hook that motivates the whole design.

For the full reasoning behind each design choice (why Gumbel-Softmax, why inject at that specific point, why pre-norm), see the docstrings in `src/markov_state.py` and `src/markov_mamba_block.py` — they're written to be read as documentation, not just code.

---

## Project structure

```
markov-mamba-project/
├── assets/
│   └── architecture.svg        # architecture diagram (shown above)
├── src/
│   ├── mamba_block.py          # pure-PyTorch selective SSM block
│   ├── markov_state.py         # Gumbel-Softmax discrete state tracker
│   ├── markov_mamba_block.py   # combines the two above
│   ├── markov_mamba_model.py   # full stacked language model
│   ├── vanilla_mamba_model.py  # parameter-matched baseline (no Markov state)
│   ├── synthetic_tasks.py      # copying / associative recall / selective copying tasks
│   └── audit_states.py         # debugging & interpretability inspection tool
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## Setup — Local (any OS, CPU)

This is the recommended path for debugging and auditing, since none of the code requires a GPU.

```bash
# 1. Clone or copy this project, then from its root directory:
python -m venv venv

# 2. Activate the environment
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows (PowerShell/cmd)

# 3. Install dependencies (CPU build of PyTorch is installed by default)
pip install -r requirements.txt

# 4. Verify the setup by running each module's self-test
cd src
python mamba_block.py
python markov_state.py
python markov_mamba_block.py
python markov_mamba_model.py
python vanilla_mamba_model.py
python synthetic_tasks.py

# 5. Run the debugging/auditing tool
python audit_states.py
```

If you hit a `DLL load failed` error on Windows when importing `torch`, install the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) and retry — this is unrelated to this project's code and is a general PyTorch-on-Windows requirement.

---

## Setup — Local machine with an NVIDIA GPU

The code in this repo runs correctly on GPU with **zero changes** — `torch` will use CUDA automatically once tensors/models are moved to it. To take advantage of a GPU:

```bash
# Install the CUDA build of PyTorch instead of the CPU build.
# Check https://pytorch.org/get-started/locally/ for the exact command
# matching your CUDA version, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt   # installs the remaining (non-torch) dependencies
```

Then, in any script, move the model and inputs to `cuda`:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MarkovMambaLM(vocab_size=1000, d_model=64, num_layers=4).to(device)
input_ids = input_ids.to(device)
```

**Note:** this repo implements the Mamba selective scan as a plain sequential Python loop (no fused CUDA kernels), by design — it's meant to be transparent and easy to modify for debugging, not fast. For large-scale training, consider porting the validated architecture to the official [`mamba-ssm`](https://github.com/state-spaces/mamba) CUDA kernels once the design is finalized.

---

## Setup — Docker (any system, consistent environment)

Docker gives you an identical environment regardless of host OS, without touching your system Python install. This uses the **CPU** build of PyTorch, matching the local CPU setup above.

### Build the image

```bash
docker build -t markov-mamba .
```

### Run the default audit tool

```bash
docker run --rm markov-mamba
```

### Run a specific script (e.g. training or a different self-test)

```bash
docker run --rm markov-mamba python markov_mamba_model.py
docker run --rm markov-mamba python synthetic_tasks.py
```

### Run interactively (for debugging / exploring inside the container)

```bash
docker run --rm -it markov-mamba /bin/bash
```

### Mount a local folder to save outputs (e.g. training histories, JSON logs)

```bash
docker run --rm -v "$(pwd)/outputs:/app/src/outputs" markov-mamba python markov_mamba_model.py
```

### GPU-enabled Docker (optional, requires NVIDIA Container Toolkit on the host)

The provided `Dockerfile` uses CPU PyTorch for portability. For a GPU-enabled image, swap the base image and PyTorch install:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# ... then install Python, pip, and:
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Run with:
```bash
docker run --rm --gpus all markov-mamba-gpu
```

This requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host machine.

---

## Using this for debugging / auditing during development

The core practical value of this architecture is that it exposes **named, inspectable discrete states** at every layer — no clustering or probing step required, unlike a plain Mamba model's opaque continuous hidden state.

`src/audit_states.py` is the entry point for this:

```bash
cd src
python audit_states.py
```

This prints, for a batch of input tokens:
- The **discrete state index** chosen at every position, for every layer
- An **entropy** measure (how spread out the state usage is — low entropy can indicate the tracker has collapsed to one dominant mode)
- A **dominant-state fraction** warning if one state is being used for more than 50% of all assignments (a sign something may be wrong, e.g. an undertrained or miscalibrated Gumbel-Softmax temperature)

To audit a specific scenario (e.g. a real batch from one of your training runs, or a synthetic task), swap the random `input_ids` in `audit_states.py`'s `__main__` block for your own batch:

```python
from synthetic_tasks import generate_copying_task
input_ids, target_ids = generate_copying_task(batch_size=8, seq_len=32, vocab_size=50, num_tokens_to_copy=4)
audit_batch(model, input_ids)
```

This is intended as a lightweight, always-available diagnostic — run it any time you change the architecture, the training task, or the Gumbel-Softmax temperature, to sanity-check that the discrete states are behaving reasonably before investing more compute in a full training run.

---

## A note on scope

This is a research/learning project, not a production-ready library. The Mamba implementation is unoptimized (no fused kernels, sequential Python loop for the scan) by design, to keep it transparent and easy to modify. Treat performance numbers accordingly — the value here is in the interpretability mechanism and the debugging workflow around it, not raw speed or benchmark scores.
