from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorDetail(BaseModel):
    """Base model for error details."""

    code: str = Field(..., description="Error code for the specific error")
    message: str = Field(..., description="Human readable error message")
    level: ErrorLevel = Field(default=ErrorLevel.ERROR, description="Severity level of the error")
    context: dict[str, Any] | None = Field(default=None, description="Additional context about the error")


class GitHubURLError(ErrorDetail):
    """Model for GitHub URL related errors."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "INVALID_GITHUB_URL",
                "message": "The provided URL is not a valid GitHub repository URL",
                "level": ErrorLevel.ERROR,
                "context": {"url": "https://invalid-url.com"},
            }
        }
    )


class TopicAnalysisError(ErrorDetail):
    """Model for topic analysis related errors."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "TOPIC_GENERATION_FAILED",
                "message": "Failed to generate topics from the content",
                "level": ErrorLevel.ERROR,
                "context": {"model": "deberta-v3-base", "error": "Model loading failed"},
            }
        }
    )


class APIResponse(BaseModel):
    """Model for API responses."""

    success: bool = Field(default=True, description="Indicates if the operation was successful")
    data: dict[str, Any] | None = Field(default=None, description="Response data when operation is successful")
    errors: list[ErrorDetail] | None = Field(default=None, description="List of errors if any occurred")

    def model_post_init(self, __context):
        """Post initialization hook to update success status based on errors."""
        if self.errors:
            self.success = False


class ErrorHandler:
    """Handler for managing and creating error responses."""

    @staticmethod
    def handle_github_url_error(url: str, error_message: str) -> APIResponse:
        """Handle GitHub URL related errors.

        Args:
            url: The problematic URL
            error_message: Description of the error

        Returns:
            APIResponse with error details

        """
        error = GitHubURLError(
            code="INVALID_GITHUB_URL", message=f"Invalid GitHub URL: {error_message}", context={"url": url}
        )
        return APIResponse(success=False, errors=[error])

    @staticmethod
    def handle_topic_analysis_error(error_message: str, context: dict[str, Any] | None = None) -> APIResponse:
        """Handle topic analysis related errors.

        Args:
            error_message: Description of the error
            context: Additional context information

        Returns:
            APIResponse with error details

        """
        error = TopicAnalysisError(
            code="TOPIC_GENERATION_FAILED", message=f"Topic generation failed: {error_message}", context=context or {}
        )
        return APIResponse(success=False, errors=[error])

    @staticmethod
    def handle_file_fetch_error(file_path: str, error_message: str) -> APIResponse:
        """Handle file fetching related errors.

        Args:
            file_path: Path of the file that failed to fetch
            error_message: Description of the error

        Returns:
            APIResponse with error details

        """
        error = ErrorDetail(
            code="FILE_FETCH_FAILED", message=f"Failed to fetch file: {error_message}", context={"file_path": file_path}
        )
        return APIResponse(success=False, errors=[error])

    @staticmethod
    def success_response(data: dict[str, Any]) -> APIResponse:
        """Create a success response.

        Args:
            data: The response data to be returned

        Returns:
            APIResponse with success status and data

        """
        return APIResponse(success=True, data=data)
