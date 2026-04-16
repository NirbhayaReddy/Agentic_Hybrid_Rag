from fastapi import APIRouter
from sqlalchemy import text

from ..dependencies import DatabaseDep, OllamaDep, OpenSearchDep, SettingsDep
from ..schemas.api.health import HealthResponse, ServiceStatus

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    settings: SettingsDep,
    database: DatabaseDep,
    opensearch_client: OpenSearchDep,
    llm_client: OllamaDep,
) -> HealthResponse:
    """Comprehensive health check endpoint for monitoring and load balancer probes."""
    services = {}
    overall_status = "ok"

    # Database check
    try:
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
        services["database"] = ServiceStatus(status="healthy", message="Connected successfully")
    except Exception as e:
        services["database"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    # OpenSearch check
    try:
        if not opensearch_client.health_check():
            services["opensearch"] = ServiceStatus(status="unhealthy", message="Not responding")
            overall_status = "degraded"
        else:
            stats = opensearch_client.get_index_stats()
            services["opensearch"] = ServiceStatus(
                status="healthy",
                message=f"Index '{stats.get('index_name', 'unknown')}' with {stats.get('document_count', 0)} documents",
            )
    except Exception as e:
        services["opensearch"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    # LLM provider check
    try:
        llm_health = await llm_client.health_check()
        services["llm"] = ServiceStatus(status=llm_health["status"], message=llm_health["message"])
        if llm_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["llm"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
