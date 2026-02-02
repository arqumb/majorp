"""
Scraper Repository Module

This module provides persistent storage for reusable XPath Action Sequences
in a web scraping system. It supports indexing by domain and extraction task
for efficient retrieval and reuse of proven extraction patterns.
"""

import sqlite3
import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse
import hashlib


@dataclass
class ActionSequence:
    """Represents a reusable XPath Action Sequence."""
    sequence_id: str
    domain: str
    extraction_task: str
    xpath_actions: List[Dict[str, Any]]
    success_rate: float
    robustness_score: float
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


class ScraperRepository:
    """
    Repository for storing and retrieving XPath Action Sequences.
    
    Provides persistent storage with indexing by domain and extraction task
    to enable efficient reuse of proven extraction patterns across similar
    websites and scraping scenarios.
    """
    
    def __init__(self, db_path: str = "scraper_sequences.db", use_sqlite: bool = True):
        """
        Initialize the repository with SQLite or JSON storage.
        
        Args:
            db_path: Path to the database file
            use_sqlite: If True, use SQLite; if False, use JSON file storage
        """
        self.db_path = db_path
        self.use_sqlite = use_sqlite
        self.logger = logging.getLogger(__name__)
        
        if self.use_sqlite:
            self._init_sqlite_db()
        else:
            self.json_path = db_path.replace('.db', '.json')
            self._init_json_storage()
    
    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create main sequences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS action_sequences (
                    sequence_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    extraction_task TEXT NOT NULL,
                    xpath_actions TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    robustness_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_domain 
                ON action_sequences(domain)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_extraction_task 
                ON action_sequences(extraction_task)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_domain_task 
                ON action_sequences(domain, extraction_task)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_robustness_score 
                ON action_sequences(robustness_score DESC)
            ''')
            
            conn.commit()
            self.logger.info(f"SQLite database initialized at {self.db_path}")
    
    def _init_json_storage(self) -> None:
        """Initialize JSON file storage."""
        if not os.path.exists(self.json_path):
            initial_data = {
                "sequences": {},
                "domain_index": {},
                "task_index": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            with open(self.json_path, 'w') as f:
                json.dump(initial_data, f, indent=2)
            self.logger.info(f"JSON storage initialized at {self.json_path}")
    
    def save_sequence(self, sequence: ActionSequence) -> bool:
        """
        Save an Action Sequence to the repository.
        
        Args:
            sequence: ActionSequence object to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if self.use_sqlite:
                return self._save_sequence_sqlite(sequence)
            else:
                return self._save_sequence_json(sequence)
        except Exception as e:
            self.logger.error(f"Failed to save sequence {sequence.sequence_id}: {e}")
            return False
    
    def _save_sequence_sqlite(self, sequence: ActionSequence) -> bool:
        """Save sequence to SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if sequence already exists
            cursor.execute(
                "SELECT sequence_id FROM action_sequences WHERE sequence_id = ?",
                (sequence.sequence_id,)
            )
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing sequence
                cursor.execute('''
                    UPDATE action_sequences 
                    SET domain = ?, extraction_task = ?, xpath_actions = ?,
                        success_rate = ?, robustness_score = ?, updated_at = ?, metadata = ?
                    WHERE sequence_id = ?
                ''', (
                    sequence.domain,
                    sequence.extraction_task,
                    json.dumps(sequence.xpath_actions),
                    sequence.success_rate,
                    sequence.robustness_score,
                    datetime.now().isoformat(),
                    json.dumps(sequence.metadata),
                    sequence.sequence_id
                ))
                self.logger.info(f"Updated sequence {sequence.sequence_id}")
            else:
                # Insert new sequence
                cursor.execute('''
                    INSERT INTO action_sequences 
                    (sequence_id, domain, extraction_task, xpath_actions, 
                     success_rate, robustness_score, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sequence.sequence_id,
                    sequence.domain,
                    sequence.extraction_task,
                    json.dumps(sequence.xpath_actions),
                    sequence.success_rate,
                    sequence.robustness_score,
                    sequence.created_at,
                    sequence.updated_at,
                    json.dumps(sequence.metadata)
                ))
                self.logger.info(f"Saved new sequence {sequence.sequence_id}")
            
            conn.commit()
            return True
    
    def _save_sequence_json(self, sequence: ActionSequence) -> bool:
        """Save sequence to JSON file."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Save sequence data
        data["sequences"][sequence.sequence_id] = asdict(sequence)
        
        # Update domain index
        if sequence.domain not in data["domain_index"]:
            data["domain_index"][sequence.domain] = []
        if sequence.sequence_id not in data["domain_index"][sequence.domain]:
            data["domain_index"][sequence.domain].append(sequence.sequence_id)
        
        # Update task index
        if sequence.extraction_task not in data["task_index"]:
            data["task_index"][sequence.extraction_task] = []
        if sequence.sequence_id not in data["task_index"][sequence.extraction_task]:
            data["task_index"][sequence.extraction_task].append(sequence.sequence_id)
        
        # Update metadata
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved sequence {sequence.sequence_id} to JSON storage")
        return True
    
    def fetch_sequence(self, sequence_id: str) -> Optional[ActionSequence]:
        """
        Fetch a specific Action Sequence by ID.
        
        Args:
            sequence_id: Unique identifier for the sequence
            
        Returns:
            ActionSequence object if found, None otherwise
        """
        try:
            if self.use_sqlite:
                return self._fetch_sequence_sqlite(sequence_id)
            else:
                return self._fetch_sequence_json(sequence_id)
        except Exception as e:
            self.logger.error(f"Failed to fetch sequence {sequence_id}: {e}")
            return None
    
    def _fetch_sequence_sqlite(self, sequence_id: str) -> Optional[ActionSequence]:
        """Fetch sequence from SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT sequence_id, domain, extraction_task, xpath_actions,
                       success_rate, robustness_score, created_at, updated_at, metadata
                FROM action_sequences 
                WHERE sequence_id = ?
            ''', (sequence_id,))
            
            row = cursor.fetchone()
            if row:
                return ActionSequence(
                    sequence_id=row[0],
                    domain=row[1],
                    extraction_task=row[2],
                    xpath_actions=json.loads(row[3]),
                    success_rate=row[4],
                    robustness_score=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                )
            return None
    
    def _fetch_sequence_json(self, sequence_id: str) -> Optional[ActionSequence]:
        """Fetch sequence from JSON file."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        if sequence_id in data["sequences"]:
            seq_data = data["sequences"][sequence_id]
            return ActionSequence(**seq_data)
        return None
    
    def fetch_by_domain(self, domain: str, limit: int = 10) -> List[ActionSequence]:
        """
        Fetch Action Sequences for a specific domain.
        
        Args:
            domain: Domain name to search for
            limit: Maximum number of sequences to return
            
        Returns:
            List of ActionSequence objects for the domain
        """
        try:
            if self.use_sqlite:
                return self._fetch_by_domain_sqlite(domain, limit)
            else:
                return self._fetch_by_domain_json(domain, limit)
        except Exception as e:
            self.logger.error(f"Failed to fetch sequences for domain {domain}: {e}")
            return []
    
    def _fetch_by_domain_sqlite(self, domain: str, limit: int) -> List[ActionSequence]:
        """Fetch sequences by domain from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT sequence_id, domain, extraction_task, xpath_actions,
                       success_rate, robustness_score, created_at, updated_at, metadata
                FROM action_sequences 
                WHERE domain = ?
                ORDER BY robustness_score DESC
                LIMIT ?
            ''', (domain, limit))
            
            sequences = []
            for row in cursor.fetchall():
                sequences.append(ActionSequence(
                    sequence_id=row[0],
                    domain=row[1],
                    extraction_task=row[2],
                    xpath_actions=json.loads(row[3]),
                    success_rate=row[4],
                    robustness_score=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                ))
            return sequences
    
    def _fetch_by_domain_json(self, domain: str, limit: int) -> List[ActionSequence]:
        """Fetch sequences by domain from JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        sequences = []
        if domain in data["domain_index"]:
            sequence_ids = data["domain_index"][domain][:limit]
            for seq_id in sequence_ids:
                if seq_id in data["sequences"]:
                    sequences.append(ActionSequence(**data["sequences"][seq_id]))
        
        # Sort by robustness score
        sequences.sort(key=lambda x: x.robustness_score, reverse=True)
        return sequences[:limit]
    
    def fetch_by_task(self, extraction_task: str, limit: int = 10) -> List[ActionSequence]:
        """
        Fetch Action Sequences for a specific extraction task.
        
        Args:
            extraction_task: Task description to search for
            limit: Maximum number of sequences to return
            
        Returns:
            List of ActionSequence objects for the task
        """
        try:
            if self.use_sqlite:
                return self._fetch_by_task_sqlite(extraction_task, limit)
            else:
                return self._fetch_by_task_json(extraction_task, limit)
        except Exception as e:
            self.logger.error(f"Failed to fetch sequences for task {extraction_task}: {e}")
            return []
    
    def _fetch_by_task_sqlite(self, extraction_task: str, limit: int) -> List[ActionSequence]:
        """Fetch sequences by task from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT sequence_id, domain, extraction_task, xpath_actions,
                       success_rate, robustness_score, created_at, updated_at, metadata
                FROM action_sequences 
                WHERE extraction_task = ?
                ORDER BY robustness_score DESC
                LIMIT ?
            ''', (extraction_task, limit))
            
            sequences = []
            for row in cursor.fetchall():
                sequences.append(ActionSequence(
                    sequence_id=row[0],
                    domain=row[1],
                    extraction_task=row[2],
                    xpath_actions=json.loads(row[3]),
                    success_rate=row[4],
                    robustness_score=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                ))
            return sequences
    
    def _fetch_by_task_json(self, extraction_task: str, limit: int) -> List[ActionSequence]:
        """Fetch sequences by task from JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        sequences = []
        if extraction_task in data["task_index"]:
            sequence_ids = data["task_index"][extraction_task][:limit]
            for seq_id in sequence_ids:
                if seq_id in data["sequences"]:
                    sequences.append(ActionSequence(**data["sequences"][seq_id]))
        
        # Sort by robustness score
        sequences.sort(key=lambda x: x.robustness_score, reverse=True)
        return sequences[:limit]
    
    def fetch_by_domain_and_task(self, domain: str, extraction_task: str, 
                                limit: int = 5) -> List[ActionSequence]:
        """
        Fetch Action Sequences matching both domain and extraction task.
        
        Args:
            domain: Domain name to match
            extraction_task: Task description to match
            limit: Maximum number of sequences to return
            
        Returns:
            List of matching ActionSequence objects
        """
        try:
            if self.use_sqlite:
                return self._fetch_by_domain_and_task_sqlite(domain, extraction_task, limit)
            else:
                return self._fetch_by_domain_and_task_json(domain, extraction_task, limit)
        except Exception as e:
            self.logger.error(f"Failed to fetch sequences for {domain}/{extraction_task}: {e}")
            return []
    
    def _fetch_by_domain_and_task_sqlite(self, domain: str, extraction_task: str, 
                                       limit: int) -> List[ActionSequence]:
        """Fetch sequences by domain and task from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT sequence_id, domain, extraction_task, xpath_actions,
                       success_rate, robustness_score, created_at, updated_at, metadata
                FROM action_sequences 
                WHERE domain = ? AND extraction_task = ?
                ORDER BY robustness_score DESC
                LIMIT ?
            ''', (domain, extraction_task, limit))
            
            sequences = []
            for row in cursor.fetchall():
                sequences.append(ActionSequence(
                    sequence_id=row[0],
                    domain=row[1],
                    extraction_task=row[2],
                    xpath_actions=json.loads(row[3]),
                    success_rate=row[4],
                    robustness_score=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                ))
            return sequences
    
    def _fetch_by_domain_and_task_json(self, domain: str, extraction_task: str, 
                                     limit: int) -> List[ActionSequence]:
        """Fetch sequences by domain and task from JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        sequences = []
        
        # Find intersection of domain and task indexes
        domain_sequences = set(data["domain_index"].get(domain, []))
        task_sequences = set(data["task_index"].get(extraction_task, []))
        matching_ids = domain_sequences.intersection(task_sequences)
        
        for seq_id in matching_ids:
            if seq_id in data["sequences"]:
                sequences.append(ActionSequence(**data["sequences"][seq_id]))
        
        # Sort by robustness score and limit
        sequences.sort(key=lambda x: x.robustness_score, reverse=True)
        return sequences[:limit]
    
    def list_all_sequences(self, limit: int = 50, order_by: str = "robustness_score") -> List[ActionSequence]:
        """
        List all stored Action Sequences.
        
        Args:
            limit: Maximum number of sequences to return
            order_by: Field to order by ("robustness_score", "created_at", "updated_at")
            
        Returns:
            List of all ActionSequence objects
        """
        try:
            if self.use_sqlite:
                return self._list_all_sequences_sqlite(limit, order_by)
            else:
                return self._list_all_sequences_json(limit, order_by)
        except Exception as e:
            self.logger.error(f"Failed to list sequences: {e}")
            return []
    
    def _list_all_sequences_sqlite(self, limit: int, order_by: str) -> List[ActionSequence]:
        """List all sequences from SQLite."""
        valid_order_fields = ["robustness_score", "created_at", "updated_at", "success_rate"]
        if order_by not in valid_order_fields:
            order_by = "robustness_score"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT sequence_id, domain, extraction_task, xpath_actions,
                       success_rate, robustness_score, created_at, updated_at, metadata
                FROM action_sequences 
                ORDER BY {order_by} DESC
                LIMIT ?
            ''', (limit,))
            
            sequences = []
            for row in cursor.fetchall():
                sequences.append(ActionSequence(
                    sequence_id=row[0],
                    domain=row[1],
                    extraction_task=row[2],
                    xpath_actions=json.loads(row[3]),
                    success_rate=row[4],
                    robustness_score=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                ))
            return sequences
    
    def _list_all_sequences_json(self, limit: int, order_by: str) -> List[ActionSequence]:
        """List all sequences from JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        sequences = []
        for seq_data in data["sequences"].values():
            sequences.append(ActionSequence(**seq_data))
        
        # Sort by specified field
        if order_by == "robustness_score":
            sequences.sort(key=lambda x: x.robustness_score, reverse=True)
        elif order_by == "created_at":
            sequences.sort(key=lambda x: x.created_at, reverse=True)
        elif order_by == "updated_at":
            sequences.sort(key=lambda x: x.updated_at, reverse=True)
        elif order_by == "success_rate":
            sequences.sort(key=lambda x: x.success_rate, reverse=True)
        
        return sequences[:limit]
    
    def delete_sequence(self, sequence_id: str) -> bool:
        """
        Delete an Action Sequence from the repository.
        
        Args:
            sequence_id: ID of the sequence to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if self.use_sqlite:
                return self._delete_sequence_sqlite(sequence_id)
            else:
                return self._delete_sequence_json(sequence_id)
        except Exception as e:
            self.logger.error(f"Failed to delete sequence {sequence_id}: {e}")
            return False
    
    def _delete_sequence_sqlite(self, sequence_id: str) -> bool:
        """Delete sequence from SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM action_sequences WHERE sequence_id = ?", (sequence_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            
            if deleted:
                self.logger.info(f"Deleted sequence {sequence_id}")
            return deleted
    
    def _delete_sequence_json(self, sequence_id: str) -> bool:
        """Delete sequence from JSON file."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        if sequence_id not in data["sequences"]:
            return False
        
        # Get sequence info before deletion
        sequence = data["sequences"][sequence_id]
        domain = sequence["domain"]
        task = sequence["extraction_task"]
        
        # Remove from main sequences
        del data["sequences"][sequence_id]
        
        # Remove from indexes
        if domain in data["domain_index"]:
            data["domain_index"][domain] = [
                sid for sid in data["domain_index"][domain] if sid != sequence_id
            ]
        
        if task in data["task_index"]:
            data["task_index"][task] = [
                sid for sid in data["task_index"][task] if sid != sequence_id
            ]
        
        # Update metadata
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Deleted sequence {sequence_id}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with repository statistics
        """
        try:
            if self.use_sqlite:
                return self._get_statistics_sqlite()
            else:
                return self._get_statistics_json()
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def _get_statistics_sqlite(self) -> Dict[str, Any]:
        """Get statistics from SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total sequences
            cursor.execute("SELECT COUNT(*) FROM action_sequences")
            total_sequences = cursor.fetchone()[0]
            
            # Unique domains
            cursor.execute("SELECT COUNT(DISTINCT domain) FROM action_sequences")
            unique_domains = cursor.fetchone()[0]
            
            # Unique tasks
            cursor.execute("SELECT COUNT(DISTINCT extraction_task) FROM action_sequences")
            unique_tasks = cursor.fetchone()[0]
            
            # Average robustness score
            cursor.execute("SELECT AVG(robustness_score) FROM action_sequences")
            avg_robustness = cursor.fetchone()[0] or 0.0
            
            # Top domains by sequence count
            cursor.execute('''
                SELECT domain, COUNT(*) as count 
                FROM action_sequences 
                GROUP BY domain 
                ORDER BY count DESC 
                LIMIT 5
            ''')
            top_domains = [{"domain": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            return {
                "total_sequences": total_sequences,
                "unique_domains": unique_domains,
                "unique_tasks": unique_tasks,
                "average_robustness_score": round(avg_robustness, 3),
                "top_domains": top_domains,
                "storage_type": "SQLite"
            }
    
    def _get_statistics_json(self) -> Dict[str, Any]:
        """Get statistics from JSON file."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        sequences = data["sequences"]
        total_sequences = len(sequences)
        unique_domains = len(data["domain_index"])
        unique_tasks = len(data["task_index"])
        
        # Calculate average robustness score
        if sequences:
            robustness_scores = [seq["robustness_score"] for seq in sequences.values()]
            avg_robustness = sum(robustness_scores) / len(robustness_scores)
        else:
            avg_robustness = 0.0
        
        # Top domains by sequence count
        domain_counts = [(domain, len(seq_ids)) for domain, seq_ids in data["domain_index"].items()]
        domain_counts.sort(key=lambda x: x[1], reverse=True)
        top_domains = [{"domain": domain, "count": count} for domain, count in domain_counts[:5]]
        
        return {
            "total_sequences": total_sequences,
            "unique_domains": unique_domains,
            "unique_tasks": unique_tasks,
            "average_robustness_score": round(avg_robustness, 3),
            "top_domains": top_domains,
            "storage_type": "JSON"
        }


def create_sequence_id(domain: str, extraction_task: str) -> str:
    """
    Create a unique sequence ID based on domain and task.
    
    Args:
        domain: Domain name
        extraction_task: Task description
        
    Returns:
        Unique sequence identifier
    """
    # Create hash from domain and task for uniqueness
    content = f"{domain}:{extraction_task}:{datetime.now().isoformat()}"
    hash_obj = hashlib.md5(content.encode())
    return f"seq_{hash_obj.hexdigest()[:12]}"


def extract_domain_from_url(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return "unknown_domain"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with SQLite storage
    print("=== Testing SQLite Storage ===")
    sqlite_repo = ScraperRepository("test_scraper.db", use_sqlite=True)
    
    # Create sample sequences
    sample_sequences = [
        ActionSequence(
            sequence_id=create_sequence_id("example.com", "product_titles"),
            domain="example.com",
            extraction_task="product_titles",
            xpath_actions=[
                {"action_type": "xpath_extract", "xpath": "//h2[@class='title']/text()", "success": True}
            ],
            success_rate=0.85,
            robustness_score=0.78,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata={"source": "phase2_synthesis", "pages_tested": 5}
        ),
        ActionSequence(
            sequence_id=create_sequence_id("shop.com", "product_prices"),
            domain="shop.com",
            extraction_task="product_prices",
            xpath_actions=[
                {"action_type": "xpath_extract", "xpath": "//span[@class='price']/text()", "success": True}
            ],
            success_rate=0.92,
            robustness_score=0.88,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata={"source": "phase2_synthesis", "pages_tested": 8}
        )
    ]
    
    # Test save operations
    for seq in sample_sequences:
        success = sqlite_repo.save_sequence(seq)
        print(f"Saved sequence {seq.sequence_id}: {success}")
    
    # Test fetch operations
    print("\n--- Fetch by Domain ---")
    domain_sequences = sqlite_repo.fetch_by_domain("example.com")
    for seq in domain_sequences:
        print(f"Found: {seq.sequence_id} - {seq.extraction_task}")
    
    print("\n--- Fetch by Task ---")
    task_sequences = sqlite_repo.fetch_by_task("product_prices")
    for seq in task_sequences:
        print(f"Found: {seq.sequence_id} - {seq.domain}")
    
    print("\n--- List All Sequences ---")
    all_sequences = sqlite_repo.list_all_sequences(limit=10)
    for seq in all_sequences:
        print(f"{seq.sequence_id}: {seq.domain}/{seq.extraction_task} (score: {seq.robustness_score})")
    
    print("\n--- Repository Statistics ---")
    stats = sqlite_repo.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test with JSON storage
    print("\n\n=== Testing JSON Storage ===")
    json_repo = ScraperRepository("test_scraper.json", use_sqlite=False)
    
    # Save same sequences to JSON
    for seq in sample_sequences:
        success = json_repo.save_sequence(seq)
        print(f"Saved sequence {seq.sequence_id}: {success}")
    
    # Test JSON statistics
    print("\n--- JSON Repository Statistics ---")
    json_stats = json_repo.get_statistics()
    for key, value in json_stats.items():
        print(f"{key}: {value}")
    
    # Cleanup test files
    import os
    try:
        os.remove("test_scraper.db")
        os.remove("test_scraper.json")
        print("\nTest files cleaned up")
    except:
        pass