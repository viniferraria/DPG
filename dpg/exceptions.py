"""Exceptions raised by the DPG public API."""

from typing import Any


class DPGError(Exception):
    """Base class for all DPG-specific errors."""


class DPGValidationError(DPGError, ValueError):
    """Raised when an API argument or input is invalid."""

    @classmethod
    def empty_feature_names(cls) -> "DPGValidationError":
        return cls("Feature names cannot be empty.")

    @classmethod
    def sample_feature_count(cls, actual: int, expected: int) -> "DPGValidationError":
        return cls(f"Sample has {actual} features, expected {expected}.")

    @classmethod
    def missing_local_input(cls) -> "DPGValidationError":
        return cls("Either local_explanation or sample must be provided.")

    @classmethod
    def invalid_path_indices(cls) -> "DPGValidationError":
        return cls("path_indices must reference valid path indices.")

    @classmethod
    def invalid_max_samples(cls) -> "DPGValidationError":
        return cls("max_samples must be a positive integer when provided.")

    @classmethod
    def positive_learner_count(cls) -> "DPGValidationError":
        return cls("Number of learners must be positive.")

    @classmethod
    def empty_evaluation_input(cls) -> "DPGValidationError":
        return cls("X must contain at least one sample for faithfulness evaluation.")

    @classmethod
    def mismatched_evaluation_length(cls, argument: str) -> "DPGValidationError":
        return cls(f"{argument} length must match the number of evaluated samples.")

    @classmethod
    def mismatched_y_true_length(cls) -> "DPGValidationError":
        return cls.mismatched_evaluation_length("y_true")

    @classmethod
    def mismatched_sample_ids_length(cls) -> "DPGValidationError":
        return cls.mismatched_evaluation_length("sample_ids")

    @classmethod
    def graphviz_type_required(cls) -> "DPGValidationError":
        return cls("Input must be a Graphviz Digraph object.")

    @classmethod
    def graph_node_not_found(cls, node_id: str) -> "DPGValidationError":
        return cls(f"Node {node_id} was not found in the graph.")

    @classmethod
    def directory_required(cls, folder_path: str) -> "DPGValidationError":
        return cls(f"Path {folder_path} is not a valid directory.")

    @classmethod
    def invalid_label_mode(cls) -> "DPGValidationError":
        return cls("label_mode must be one of: full, wrapped, short.")

    @classmethod
    def invalid_readability(cls) -> "DPGValidationError":
        return cls("readability must be one of: compact, normal, presentation.")

    @classmethod
    def invalid_plot_configuration(cls) -> "DPGValidationError":
        return cls("The plot requires either no attribute/clusters, clusters, or a node metric.")

    @classmethod
    def missing_dpg_metrics(cls) -> "DPGValidationError":
        return cls("dpg_metrics is required to plot communities.")

    @classmethod
    def missing_communities(cls) -> "DPGValidationError":
        return cls("dpg_metrics must include 'Communities' or 'Clusters' to plot communities.")

    @classmethod
    def no_matching_communities(cls) -> "DPGValidationError":
        return cls("No nodes matched communities or cluster labels.")

    @classmethod
    def no_predicate_labels(cls) -> "DPGValidationError":
        return cls("No predicate labels are available to compute LRC scores.")

    @classmethod
    def feature_importances_required(cls) -> "DPGValidationError":
        return cls("Model must expose feature_importances_.")

    @classmethod
    def dataframe_required(cls) -> "DPGValidationError":
        return cls("X_df must be a pandas DataFrame with named columns.")

    @classmethod
    def explanation_graph_required(cls, context: str = "") -> "DPGValidationError":
        suffix = f" for {context}" if context else ""
        return cls(f"explanation.graph is required{suffix}.")

    @classmethod
    def class_path_graph_required(cls) -> "DPGValidationError":
        return cls.explanation_graph_required("class-path analysis")

    @classmethod
    def node_metrics_columns_required(cls) -> "DPGValidationError":
        return cls("node_metrics must contain Node and Label columns.")

    @classmethod
    def non_empty_heatmap_required(cls) -> "DPGValidationError":
        return cls("heat_df must be a non-empty class-feature count DataFrame.")


class DPGModelError(DPGValidationError):
    """Raised when a model cannot be used by DPG."""

    @classmethod
    def invalid_ensemble(cls) -> "DPGModelError":
        return cls("Model must be a tree-based ensemble with fitted estimators.")

    @classmethod
    def unsupported_model(cls, model_name: str, available_models: list[str]) -> "DPGModelError":
        available = ", ".join(available_models)
        return cls(f"Unsupported model '{model_name}'. Available models: {available}.")


class DPGConfigurationError(DPGValidationError):
    """Raised when DPG configuration or theme settings are invalid."""

    @classmethod
    def missing_parameter(cls, parameter: str) -> "DPGConfigurationError":
        return cls(f"DPG configuration is missing required parameter '{parameter}'.")

    @classmethod
    def missing_perc_var(cls) -> "DPGConfigurationError":
        return cls.missing_parameter("perc_var")

    @classmethod
    def missing_decimal_threshold(cls) -> "DPGConfigurationError":
        return cls.missing_parameter("decimal_threshold")

    @classmethod
    def missing_n_jobs(cls) -> "DPGConfigurationError":
        return cls.missing_parameter("n_jobs")

    @classmethod
    def unsupported_graph_mode(
        cls, mode: Any, supported_modes: list[str]
    ) -> "DPGConfigurationError":
        supported = ", ".join(supported_modes)
        return cls(f"Unsupported graph construction mode '{mode}'. Supported modes: {supported}.")

    @classmethod
    def unknown_palette(cls, palette: str) -> "DPGConfigurationError":
        return cls(
            f"Unknown palette '{palette}'. Expected one of: default, extended, olive."
        )

    @classmethod
    def unknown_theme(cls, theme: str) -> "DPGConfigurationError":
        return cls(f"Unknown theme '{theme}'. Expected one of: dpg, legacy.")

    @classmethod
    def invalid_yaml(cls, cause: BaseException) -> "DPGConfigurationError":
        return cls(f"Invalid YAML configuration: {cause!s}")


class DPGGraphError(DPGError):
    """Raised when a DPG graph cannot be constructed or interpreted."""

    @classmethod
    def no_paths(cls, perc_var: Any, decimal_threshold: Any) -> "DPGGraphError":
        return cls(
            "No decision paths remain after filtering with "
            f"perc_var={perc_var!r} and decimal_threshold={decimal_threshold!r}. "
            "Relax these configuration values and try again."
        )


class DPGNotFittedError(DPGValidationError):
    """Raised when an explainer operation requires a fitted graph."""

    @classmethod
    def require_fit(cls, next_step: str) -> "DPGNotFittedError":
        return cls(f"DPGExplainer is not fitted. Call {next_step}.")

    @classmethod
    def for_graph(cls) -> "DPGNotFittedError":
        return cls.require_fit("fit(X) first")

    @classmethod
    def for_nodes(cls) -> "DPGNotFittedError":
        return cls.require_fit("fit(X) first")

    @classmethod
    def for_global_explanation(cls) -> "DPGNotFittedError":
        return cls.require_fit("fit(X) or explain_global(X=...)")

    @classmethod
    def for_local_explanation(cls) -> "DPGNotFittedError":
        return cls.require_fit("fit(X) or explain_local(X=...)")

    @classmethod
    def for_local_plot(cls) -> "DPGNotFittedError":
        return cls.require_fit("fit(X) or plot_local_on_dpg(X=...)")

    @classmethod
    def for_faithfulness(cls) -> "DPGNotFittedError":
        return cls.require_fit("fit(X) before evaluate_faithfulness()")


class DPGExplanationError(DPGValidationError):
    """Raised when an explanation cannot be produced or evaluated."""

    @classmethod
    def all_local_explanations_failed(cls) -> "DPGExplanationError":
        return cls(
            "All local explanations failed during faithfulness evaluation; "
            "no faithfulness metrics could be computed."
        )

    @classmethod
    def unsupported_weight_keys(
        cls, keys: set[str], supported_keys: set[str]
    ) -> "DPGExplanationError":
        return cls(
            f"Unsupported faithfulness weight keys: {sorted(keys)}. "
            f"Supported keys are: {sorted(supported_keys)}"
        )

    @classmethod
    def missing_weight_keys(
        cls, keys: set[str], supported_keys: set[str]
    ) -> "DPGExplanationError":
        return cls(
            f"Missing faithfulness weight keys: {sorted(keys)}. "
            f"Supported keys are: {sorted(supported_keys)}"
        )

    @classmethod
    def weights_do_not_sum_to_one(cls) -> "DPGExplanationError":
        return cls("Faithfulness weights must sum to 1.0.")


class DPGDatasetError(DPGValidationError):
    """Raised when a dataset cannot be loaded or validated."""

    @classmethod
    def missing_target_column(cls, column: str) -> "DPGDatasetError":
        return cls(f"Target column '{column}' was not found in the dataset.")

    @classmethod
    def load_failed(cls, source: str, cause: BaseException) -> "DPGDatasetError":
        return cls(
            f"Failed to load dataset '{source}': "
            f"{type(cause).__name__}: {cause!s}"
        )


class DPGVisualizationError(DPGError, RuntimeError):
    """Raised when a visualization dependency or rendering step fails."""

    @classmethod
    def graphviz_not_found(cls) -> "DPGVisualizationError":
        return cls(
            "Graphviz executable 'dot' was not found in PATH.\n"
            "Install Graphviz and ensure 'dot' is available from your terminal.\n"
            "Install examples:\n"
            "- macOS (Homebrew): brew install graphviz\n"
            "- Ubuntu/Debian: sudo apt-get install graphviz\n"
            "- Windows (winget): winget install Graphviz.Graphviz"
        )


class DPGMetricError(DPGValidationError):
    """Raised when graph metrics cannot be calculated from the input graph."""

    @classmethod
    def non_positive_edge_weight(cls, metric: str) -> "DPGMetricError":
        return cls(f"Total edge weight must be positive for {metric}.")

    @classmethod
    def non_positive_lrc_weight(cls) -> "DPGMetricError":
        return cls.non_positive_edge_weight("LRC")

    @classmethod
    def non_positive_closeness_weight(cls) -> "DPGMetricError":
        return cls.non_positive_edge_weight("closeness centrality")

    @classmethod
    def non_positive_harmonic_weight(cls) -> "DPGMetricError":
        return cls.non_positive_edge_weight("harmonic centrality")
