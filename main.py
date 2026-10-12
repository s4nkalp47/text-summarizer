import argparse

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger
from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from textSummarizer.pipeline.stage_02_data_transformation import DataTransformationPipeline
from textSummarizer.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from textSummarizer.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

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


PIPELINES = {
    "ingestion": DataIngestionPipeline,
    "transformation": DataTransformationPipeline,
    "training": ModelTrainerPipeline,
    "evaluation": ModelEvaluationPipeline,
}


def run_stage(stage: str, config_manager: ConfigurationManager) -> None:
    logger.info(f">>>>> Stage '{stage}' started <<<<<")
    PIPELINES[stage](config_manager).main()
    logger.info(f">>>>> Stage '{stage}' completed <<<<<")


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
