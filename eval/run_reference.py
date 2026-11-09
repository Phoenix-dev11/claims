"""Run the subnet's canonical extraction with the published reference profile.

Usage: run_reference.py PDF PAPER_ID TITLE OUTPUT (with CLAIMS_REPO on PYTHONPATH)
"""

import os
import sys
from pathlib import Path

import miner.agent_v1 as agent_package
from miner.agent_v1.config import AgentV1Config
from miner.agent_v1.runner import AgentV1Runner


def main() -> None:
    pdf, paper_id, title, output = sys.argv[1:5]
    output_path = Path(output)
    config = AgentV1Config.from_env()
    config.runtime = "dspy-react"
    config.model = "openrouter/openrouter/deepseek/deepseek-v4-flash-0731"
    config.pdf_reader = "pdf-inspector"
    config.provider = "openrouter"
    config.api_base = os.environ["CLAIMS_INFERENCE_BASE_URL"]
    config.api_key = os.environ["CLAIMS_INFERENCE_TOKEN"]
    config.temperature = 0.0
    config.max_tokens = int(os.environ.get("REF_MAX_TOKENS", "8192"))
    config.max_agent_iters = 6
    config.max_repair_attempts = 3
    config.skill_dir = Path(agent_package.__file__).parent / "skills" / "compiler"
    artifact = AgentV1Runner(config).run_from_pdf(
        Path(pdf), output_dir=output_path.parent / "reference-run",
        paper_override={"paper_id": paper_id, "title": title, "source_url": "", "source_sha256": ""},
    )
    output_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
