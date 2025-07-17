"""
Enhanced database configuration with all models for Phase 2
"""

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import redis.asyncio as redis
from config.settings import settings
from models.workflow import Workflow


# MongoDB
class Database:
    client: AsyncIOMotorClient = None  # type: ignore


database = Database()


async def connect_to_mongo():
    """Create database connection"""
    database.client = AsyncIOMotorClient(settings.mongodb_url)

    # Initialize Beanie with all document models
    await init_beanie(database=database.client.ops_iq, document_models=[Workflow])

    # Create indexes for better performance
    await create_indexes()

    print("✅ Connected to MongoDB with all models initialized")


async def create_indexes():
    """Create database indexes for better performance"""
    try:
        # Workflow indexes
        await Workflow.get_motor_collection().create_index(
            [("workflow_id", 1)], unique=True
        )

        await Workflow.get_motor_collection().create_index(
            [("user_id", 1), ("created_at", -1)]
        )

        await Workflow.get_motor_collection().create_index(
            [("status", 1), ("created_at", -1)]
        )

        await Workflow.get_motor_collection().create_index([("session_id", 1)])

        # Text search index for commands
        await Workflow.get_motor_collection().create_index(
            [("command.original_text", "text"), ("text_input", "text")]
        )

        print("✅ Database indexes created")

    except Exception as e:
        print(f"⚠️ Index creation warning: {e}")


async def close_mongo_connection():
    """Close database connection"""
    if database.client:
        database.client.close()
        print("✅ MongoDB connection closed")


# Redis
class RedisConnection:
    pool = None


redis_conn = RedisConnection()


async def connect_to_redis():
    """Create Redis connection"""
    redis_conn.pool = redis.ConnectionPool.from_url(settings.redis_url)
    print("✅ Connected to Redis")


async def close_redis_connection():
    """Close Redis connection"""
    if redis_conn.pool:
        await redis_conn.pool.disconnect()
        print("✅ Redis connection closed")


# Database utility functions
async def get_workflow_by_id(workflow_id: str) -> Workflow:
    """Get workflow by ID"""
    return await Workflow.find_one({"workflow_id": workflow_id})


async def get_user_workflows(user_id: str, limit: int = 20) -> list:
    """Get workflows for a specific user"""
    return (
        await Workflow.find({"user_id": user_id})
        .sort(-Workflow.created_at)
        .limit(limit)
        .to_list()
    )


async def get_workflows_by_status(status: str, limit: int = 50) -> list:
    """Get workflows by status"""
    return (
        await Workflow.find({"status": status})
        .sort(-Workflow.created_at)
        .limit(limit)
        .to_list()
    )


async def search_workflows(query: str, user_id: str = None, limit: int = 20) -> list:
    """Search workflows by text content"""
    search_filter = {"$text": {"$search": query}}
    if user_id:
        search_filter["user_id"] = user_id

    return await Workflow.find(search_filter).limit(limit).to_list()


async def get_workflow_metrics():
    """Get workflow statistics"""
    pipeline = [
        {
            "$group": {
                "_id": "$status",
                "count": {"$sum": 1},
                "avg_processing_time": {
                    "$avg": {"$sum": "$agent_steps.processing_time"}
                },
            }
        }
    ]

    return await Workflow.aggregate(pipeline).to_list()
