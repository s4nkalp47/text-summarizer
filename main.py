import argparse

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

STAGES = ["ingestion", "transformation", "training", "evaluation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dialogue summarizer pipeline.")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override smoke_test.enabled in params.yaml (--smoke / --no-smoke).",
    )
    return parser.parse_args()


def run_stage(stage: str, config_manager: ConfigurationManager) -> None:
    get_config = {
        "ingestion": config_manager.get_data_ingestion_config,
        "transformation": config_manager.get_data_transformation_config,
        "training": config_manager.get_model_trainer_config,
        "evaluation": config_manager.get_model_evaluation_config,
    }[stage]
    logger.info(f">>>>> Stage '{stage}' started <<<<<")
    logger.info(f"Config: {get_config()}")
    logger.info(f"Stage '{stage}': not implemented")


def main() -> None:
    args = parse_args()
    try:
        config_manager = ConfigurationManager(smoke=args.smoke)
        for stage in STAGES if args.stage == "all" else [args.stage]:
            run_stage(stage, config_manager)
    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
