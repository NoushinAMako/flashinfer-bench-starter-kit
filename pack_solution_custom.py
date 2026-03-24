"""
Custom solution packer for FlashInfer-Bench.

This script manually creates a Solution object from source files.
"""

import json
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import Solution, BuildSpec, SourceFile


def load_config():
    """Load configuration from config.toml."""
    config_path = Path("config.toml")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution():
    """Pack solution files into solution.json."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory
    if language == "triton":
        source_dir_str = build_config.get("source_dir", "solution/triton")
        source_dir = Path(source_dir_str)
        files_to_pack = ["kernel.py"]
    elif language == "cuda":
        source_dir_str = build_config.get("source_dir", "solution/cuda")
        source_dir = Path(source_dir_str)
        files_to_pack = ["kernel.cu", "kernel.cpp"]
    else:
        raise ValueError(f"Unsupported language: {language}")

    # Read source files
    sources = []
    for filename in files_to_pack:
        file_path = source_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        with open(file_path, "r") as f:
            content = f.read()

        sources.append(
            SourceFile(
                path=filename,
                content=content
            )
        )

    # Create build spec
    dps = build_config.get("destination_passing_style", True)
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=dps,
    )

    # Create solution
    solution = Solution(
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
        spec=spec,
        sources=sources,
    )

    # Write to file
    output_path = Path("solution.json")
    with open(output_path, "w") as f:
        f.write(solution.model_dump_json(indent=2))

    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")
    print(f"  Sources: {', '.join(files_to_pack)}")

    return output_path


if __name__ == "__main__":
    pack_solution()
