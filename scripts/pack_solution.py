"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def load_config(config_path: Path = None) -> dict:
    """Load configuration from config.toml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.toml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None, config_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config(config_path)

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Use the directory containing config.toml as the base for resolving paths
    if config_path is not None:
        base_dir = Path(config_path).parent
    else:
        base_dir = PROJECT_ROOT

    # Determine source directory: respect source_dir in config, else use default
    if "source_dir" in build_config:
        source_dir = base_dir / "solution" / build_config["source_dir"]
    elif language == "triton":
        source_dir = base_dir / "solution" / "triton"
    elif language == "cuda":
        source_dir = base_dir / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    dps = build_config.get("destination_passing_style", True)
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=dps,
    )

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    # Write to output file
    if output_path is None:
        output_path = base_dir / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: <config_dir>/solution.json)"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: repo root config.toml)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output, args.config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
