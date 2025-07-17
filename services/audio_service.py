"""
Audio Service - Audio file management with GridFS storage
Handles audio upload, storage, retrieval, and metadata management
"""

import io
import hashlib

# import mimetypes
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

# import gridfs
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
import librosa

# from pymongo.errors import GridFSError
# import numpy as np
# from config.settings import settings
from config.logging import get_logger


@dataclass
class AudioMetadata:
    """Audio file metadata"""

    file_id: str
    filename: str
    content_type: str
    size: int
    duration: float
    sample_rate: int
    channels: int
    format: str
    uploaded_at: datetime
    md5_hash: str
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None


class AudioService:
    """Service for audio file management using GridFS"""

    def __init__(self):
        self.logger = get_logger("audio_service")
        self.db_client: Optional[AsyncIOMotorClient] = None
        self.fs: Optional[AsyncIOMotorGridFSBucket] = None

        # Supported audio formats
        self.supported_formats = {
            "audio/wav": ".wav",
            "audio/mp3": ".mp3",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
            "audio/aac": ".aac",
            "audio/ogg": ".ogg",
            "audio/flac": ".flac",
            "audio/webm": ".webm",
        }

        # Audio constraints
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_duration = 300  # 5 minutes
        self.min_duration = 0.1  # 100ms

    async def initialize(self, db_client: AsyncIOMotorClient):
        """Initialize the audio service with database connection"""
        try:
            self.db_client = db_client
            self.fs = AsyncIOMotorGridFSBucket(
                self.db_client.ops_iq, bucket_name="audio_files"
            )

            # Create indexes for better performance
            await self._create_indexes()

            self.logger.info("Audio service initialized with GridFS")

        except Exception as e:
            self.logger.error(f"Audio service initialization failed: {e}")
            raise

    async def _create_indexes(self):
        """Create database indexes for audio metadata"""
        try:
            collection = self.db_client.ops_iq.audio_files.files

            # Create indexes for common queries
            await collection.create_index("metadata.user_id")
            await collection.create_index("metadata.workflow_id")
            await collection.create_index("metadata.md5_hash")
            await collection.create_index("uploadDate")

            self.logger.info("Audio metadata indexes created")

        except Exception as e:
            self.logger.warning(f"Index creation failed: {e}")

    def _validate_audio_file(
        self, file_data: bytes, filename: str, content_type: str
    ) -> bool:
        """Validate audio file before processing"""
        try:
            # Check file size
            if len(file_data) > self.max_file_size:
                raise ValueError(
                    f"File too large: {len(file_data)} bytes (max {self.max_file_size})"
                )

            if len(file_data) < 100:
                raise ValueError("File too small (minimum 100 bytes)")

            # Check content type
            if content_type not in self.supported_formats:
                raise ValueError(f"Unsupported format: {content_type}")

            # Basic file header validation
            if content_type == "audio/wav" and not file_data.startswith(b"RIFF"):
                raise ValueError("Invalid WAV file header")
            elif content_type in ["audio/mp3", "audio/mpeg"] and not (
                file_data.startswith(b"ID3") or file_data.startswith(b"\xff\xfb")
            ):
                raise ValueError("Invalid MP3 file header")

            return True

        except Exception as e:
            self.logger.error(f"Audio validation failed: {e}")
            return False

    def _extract_audio_metadata(self, file_data: bytes) -> Dict[str, Any]:
        """Extract audio metadata using librosa"""
        try:
            # Create temporary file-like object
            audio_buffer = io.BytesIO(file_data)

            # Load audio with librosa
            y, sr = librosa.load(audio_buffer, sr=None)

            # Calculate metadata
            duration = len(y) / sr
            channels = 1 if y.ndim == 1 else y.shape[0]

            # Validate duration
            if duration > self.max_duration:
                raise ValueError(
                    f"Audio too long: {duration:.1f}s (max {self.max_duration}s)"
                )
            elif duration < self.min_duration:
                raise ValueError(
                    f"Audio too short: {duration:.1f}s (min {self.min_duration}s)"
                )

            return {
                "duration": float(duration),
                "sample_rate": int(sr),
                "channels": int(channels),
                "samples": len(y),
            }

        except Exception as e:
            self.logger.error(f"Audio metadata extraction failed: {e}")
            # Return default values if extraction fails
            return {"duration": 0.0, "sample_rate": 16000, "channels": 1, "samples": 0}

    def _calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate MD5 hash of file data"""
        return hashlib.md5(file_data).hexdigest()

    async def upload_audio_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AudioMetadata:
        """
        Upload audio file to GridFS with metadata

        Args:
            file_data: Raw audio file bytes
            filename: Original filename
            content_type: MIME type
            user_id: User who uploaded the file
            workflow_id: Associated workflow ID
            metadata: Additional metadata

        Returns:
            AudioMetadata object with file information
        """
        try:
            # Validate file
            if not self._validate_audio_file(file_data, filename, content_type):
                raise ValueError("Audio file validation failed")

            # Extract audio metadata
            audio_meta = self._extract_audio_metadata(file_data)

            # Calculate file hash
            file_hash = self._calculate_file_hash(file_data)

            # Check for duplicate files
            existing_file = await self._find_file_by_hash(file_hash)
            if existing_file:
                self.logger.info(f"Duplicate file detected: {file_hash}")
                return existing_file

            # Prepare metadata
            file_metadata = {
                "original_filename": filename,
                "content_type": content_type,
                "user_id": user_id,
                "workflow_id": workflow_id,
                "md5_hash": file_hash,
                "audio_duration": audio_meta["duration"],
                "audio_sample_rate": audio_meta["sample_rate"],
                "audio_channels": audio_meta["channels"],
                "audio_samples": audio_meta["samples"],
                "uploaded_at": datetime.utcnow(),
                **(metadata or {}),
            }

            # Upload to GridFS
            file_stream = io.BytesIO(file_data)
            file_id = await self.fs.upload_from_stream(
                filename, file_stream, metadata=file_metadata
            )

            self.logger.info(
                f"Audio file uploaded: {filename} ({len(file_data)} bytes, "
                f"{audio_meta['duration']:.1f}s, ID: {file_id})"
            )

            # Return metadata
            return AudioMetadata(
                file_id=str(file_id),
                filename=filename,
                content_type=content_type,
                size=len(file_data),
                duration=audio_meta["duration"],
                sample_rate=audio_meta["sample_rate"],
                channels=audio_meta["channels"],
                format=self.supported_formats.get(content_type, "unknown"),
                uploaded_at=file_metadata["uploaded_at"],
                md5_hash=file_hash,
                user_id=user_id,
                workflow_id=workflow_id,
            )

        except Exception as e:
            self.logger.error(f"Audio upload failed: {e}")
            raise

    async def _find_file_by_hash(self, file_hash: str) -> Optional[AudioMetadata]:
        """Find existing file by MD5 hash"""
        try:
            cursor = self.fs.find({"metadata.md5_hash": file_hash})
            async for file_doc in cursor:
                metadata = file_doc.metadata
                return AudioMetadata(
                    file_id=str(file_doc._id),
                    filename=file_doc.filename,
                    content_type=metadata.get("content_type", "unknown"),
                    size=file_doc.length,
                    duration=metadata.get("audio_duration", 0.0),
                    sample_rate=metadata.get("audio_sample_rate", 16000),
                    channels=metadata.get("audio_channels", 1),
                    format=metadata.get("format", "unknown"),
                    uploaded_at=metadata.get("uploaded_at", datetime.utcnow()),
                    md5_hash=file_hash,
                    user_id=metadata.get("user_id"),
                    workflow_id=metadata.get("workflow_id"),
                )
            return None

        except Exception as e:
            self.logger.error(f"Hash lookup failed: {e}")
            return None

    async def get_audio_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve audio file data by ID"""
        try:
            from bson import ObjectId

            # Download file from GridFS
            file_stream = io.BytesIO()
            await self.fs.download_to_stream(ObjectId(file_id), file_stream)

            file_data = file_stream.getvalue()
            self.logger.info(
                f"Audio file retrieved: {file_id} ({len(file_data)} bytes)"
            )

            return file_data

        except Exception as e:
            self.logger.error(f"Audio retrieval failed: {e}")
            return None

    async def get_audio_metadata(self, file_id: str) -> Optional[AudioMetadata]:
        """Get audio file metadata by ID"""
        try:
            from bson import ObjectId

            # Find file in GridFS
            file_doc = await self.fs.find_one({"_id": ObjectId(file_id)})

            if not file_doc:
                return None

            metadata = file_doc.metadata
            return AudioMetadata(
                file_id=str(file_doc._id),
                filename=file_doc.filename,
                content_type=metadata.get("content_type", "unknown"),
                size=file_doc.length,
                duration=metadata.get("audio_duration", 0.0),
                sample_rate=metadata.get("audio_sample_rate", 16000),
                channels=metadata.get("audio_channels", 1),
                format=self.supported_formats.get(
                    metadata.get("content_type"), "unknown"
                ),
                uploaded_at=metadata.get("uploaded_at", datetime.utcnow()),
                md5_hash=metadata.get("md5_hash", ""),
                user_id=metadata.get("user_id"),
                workflow_id=metadata.get("workflow_id"),
            )

        except Exception as e:
            self.logger.error(f"Metadata retrieval failed: {e}")
            return None

    async def list_audio_files(
        self,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> List[AudioMetadata]:
        """List audio files with optional filtering"""
        try:
            # Build query filter
            query_filter = {}
            if user_id:
                query_filter["metadata.user_id"] = user_id
            if workflow_id:
                query_filter["metadata.workflow_id"] = workflow_id

            # Query files
            cursor = (
                self.fs.find(query_filter)
                .sort("uploadDate", -1)
                .skip(skip)
                .limit(limit)
            )

            files = []
            async for file_doc in cursor:
                metadata = file_doc.metadata
                files.append(
                    AudioMetadata(
                        file_id=str(file_doc._id),
                        filename=file_doc.filename,
                        content_type=metadata.get("content_type", "unknown"),
                        size=file_doc.length,
                        duration=metadata.get("audio_duration", 0.0),
                        sample_rate=metadata.get("audio_sample_rate", 16000),
                        channels=metadata.get("audio_channels", 1),
                        format=self.supported_formats.get(
                            metadata.get("content_type"), "unknown"
                        ),
                        uploaded_at=metadata.get("uploaded_at", datetime.utcnow()),
                        md5_hash=metadata.get("md5_hash", ""),
                        user_id=metadata.get("user_id"),
                        workflow_id=metadata.get("workflow_id"),
                    )
                )

            return files

        except Exception as e:
            self.logger.error(f"File listing failed: {e}")
            return []

    async def delete_audio_file(self, file_id: str) -> bool:
        """Delete audio file by ID"""
        try:
            from bson import ObjectId

            await self.fs.delete(ObjectId(file_id))
            self.logger.info(f"Audio file deleted: {file_id}")
            return True

        except Exception as e:
            self.logger.error(f"File deletion failed: {e} -- {file_id}")
            return False

    async def cleanup_old_files(self, days_old: int = 30) -> int:
        """Clean up audio files older than specified days"""
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            # Find old files
            cursor = self.fs.find({"uploadDate": {"$lt": cutoff_date}})

            deleted_count = 0
            async for file_doc in cursor:
                try:
                    await self.fs.delete(file_doc._id)
                    deleted_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete file {file_doc._id}: {e}")

            self.logger.info(f"Cleaned up {deleted_count} old audio files")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 0

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get audio storage statistics"""
        try:
            # Count total files
            total_files = await self.fs.find().count()

            # Calculate total size
            total_size = 0
            total_duration = 0.0

            cursor = self.fs.find()
            async for file_doc in cursor:
                total_size += file_doc.length
                total_duration += file_doc.metadata.get("audio_duration", 0.0)

            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_duration_seconds": total_duration,
                "total_duration_hours": total_duration / 3600,
                "average_file_size": total_size / max(total_files, 1),
                "average_duration": total_duration / max(total_files, 1),
            }

        except Exception as e:
            self.logger.error(f"Stats calculation failed: {e}")
            return {}

    def get_supported_formats(self) -> Dict[str, str]:
        """Get supported audio formats"""
        return self.supported_formats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Health check for audio service"""
        try:
            if not self.fs:
                return {"status": "unhealthy", "reason": "GridFS not initialized"}

            # Test basic GridFS operation
            test_data = b"test audio data"
            test_stream = io.BytesIO(test_data)

            # Upload test file
            file_id = await self.fs.upload_from_stream(
                "health_check.bin",
                test_stream,
                metadata={"test": True, "created_at": datetime.utcnow()},
            )

            # Download and verify
            download_stream = io.BytesIO()
            await self.fs.download_to_stream(file_id, download_stream)
            downloaded_data = download_stream.getvalue()

            # Cleanup test file
            await self.fs.delete(file_id)

            # Verify data integrity
            if downloaded_data == test_data:
                return {
                    "status": "healthy",
                    "gridfs_operational": True,
                    "supported_formats": len(self.supported_formats),
                }
            else:
                return {"status": "degraded", "reason": "Data integrity check failed"}

        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}


# Global audio service instance
audio_service = AudioService()


# Convenience functions
async def initialize_audio_service(db_client: AsyncIOMotorClient):
    """Initialize the global audio service"""
    await audio_service.initialize(db_client)


async def upload_audio(
    file_data: bytes, filename: str, content_type: str, **kwargs
) -> AudioMetadata:
    """Upload audio file using the global service"""
    return await audio_service.upload_audio_file(
        file_data, filename, content_type, **kwargs
    )


async def get_audio(file_id: str) -> Optional[bytes]:
    """Get audio file using the global service"""
    return await audio_service.get_audio_file(file_id)


async def get_audio_info(file_id: str) -> Optional[AudioMetadata]:
    """Get audio metadata using the global service"""
    return await audio_service.get_audio_metadata(file_id)
