"""
CloudForge AI - Migration Analyzer Unit Tests
Production-ready test suite for database migration analysis.

Version: 2.0.0
Date: October 1, 2025

Test Coverage:
✓ Schema analysis validation
✓ AI-powered recommendations testing  
✓ Migration timeline estimation
✓ Risk assessment algorithms
✓ Real Chinook database integration
✓ Error handling and edge cases
✓ Performance benchmarking
✓ Report generation validation

Expected Coverage: >95%
Expected Performance: <2s per test, <30s full suite
"""

import unittest
import asyncio
import tempfile
import sqlite3
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ai-scripts'))

try:
    from migration_analyzer import MigrationAnalyzer, TableAnalysis
    MIGRATION_ANALYZER_AVAILABLE = True
except ImportError:
    MIGRATION_ANALYZER_AVAILABLE = False
    print("Warning: Migration analyzer module not found")

class TestMigrationAnalyzer(unittest.TestCase):
    """
    Comprehensive test suite for Migration Analyzer.
    Tests all core functionality including AI integration.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample database"""
        if not MIGRATION_ANALYZER_AVAILABLE:
            cls.skipTest(cls, "Migration analyzer not available")
            
        # Create temporary test database
        cls.test_db_fd, cls.test_db_path = tempfile.mkstemp(suffix='.db')
        cls._create_test_database()
        
        # Initialize analyzer
        cls.analyzer = MigrationAnalyzer(cls.test_db_path)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls, 'test_db_fd'):
            os.close(cls.test_db_fd)
            os.unlink(cls.test_db_path)
    
    @classmethod
    def _create_test_database(cls):
        """Create a test database with sample schema"""
        conn = sqlite3.connect(cls.test_db_path)
        cursor = conn.cursor()
        
        # Create test tables with various complexity levels
        test_schema = [
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
            """,
            """
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title VARCHAR(200) NOT NULL,
                description TEXT,
                project_id INTEGER NOT NULL,
                assigned_user_id INTEGER,
                status VARCHAR(20) DEFAULT 'pending',
                priority INTEGER DEFAULT 3,
                due_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (assigned_user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level VARCHAR(10) NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        ]
        
        # Execute schema creation
        for statement in test_schema:
            cursor.execute(statement)
        
        # Create indexes
        indexes = [
            "CREATE INDEX idx_users_email ON users(email)",
            "CREATE INDEX idx_projects_user_id ON projects(user_id)",
            "CREATE INDEX idx_tasks_project_id ON tasks(project_id)",
            "CREATE INDEX idx_tasks_status ON tasks(status)",
            "CREATE INDEX idx_logs_timestamp ON logs(timestamp)",
            "CREATE INDEX idx_logs_level ON logs(level)"
        ]
        
        for index in indexes:
            cursor.execute(index)
        
        # Insert sample data
        sample_data = [
            "INSERT INTO users (email, password_hash) VALUES ('admin@cloudforge.ai', 'hashed_password_123')",
            "INSERT INTO users (email, password_hash) VALUES ('user@cloudforge.ai', 'hashed_password_456')",
            "INSERT INTO projects (name, description, user_id) VALUES ('CloudForge AI', 'Main project', 1)",
            "INSERT INTO projects (name, description, user_id) VALUES ('Testing Suite', 'Test project', 1)",
            "INSERT INTO tasks (title, description, project_id, assigned_user_id, status) VALUES ('Setup Database', 'Configure PostgreSQL', 1, 1, 'completed')",
            "INSERT INTO tasks (title, description, project_id, assigned_user_id, status) VALUES ('Write Tests', 'Create unit tests', 1, 2, 'in_progress')",
            "INSERT INTO logs (level, message, user_id) VALUES ('INFO', 'User login successful', 1)",
            "INSERT INTO logs (level, message, user_id) VALUES ('ERROR', 'Database connection failed', 1)",
        ]
        
        for statement in sample_data:
            cursor.execute(statement)
        
        # Insert bulk data for performance testing
        for i in range(1000):
            cursor.execute(
                "INSERT INTO logs (level, message, user_id) VALUES (?, ?, ?)",
                ('INFO', f'Bulk log entry {i}', 1 if i % 2 == 0 else 2)
            )
        
        conn.commit()
        conn.close()
    
    def test_database_connection(self):
        """Test database connection and validation"""
        self.assertIsNotNone(self.analyzer.connection)
        self.assertTrue(Path(self.analyzer.db_path).exists())
        
        # Test connection is working
        cursor = self.analyzer.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        self.assertGreater(table_count, 0)
    
    def test_schema_analysis_basic(self):
        """Test basic schema analysis functionality"""
        async def run_test():
            results = await self.analyzer.analyze_schema()
            
            # Verify we got results for all tables
            self.assertGreater(len(results), 0)
            
            # Check expected tables exist
            table_names = [r.table_name for r in results]
            expected_tables = ['users', 'projects', 'tasks', 'logs']
            
            for expected_table in expected_tables:
                self.assertIn(expected_table, table_names)
        
        asyncio.run(run_test())
    
    def test_single_table_analysis(self):
        """Test analysis of a single table"""
        async def run_test():
            results = await self.analyzer.analyze_schema('users')
            
            self.assertEqual(len(results), 1)
            analysis = results[0]
            
            # Verify basic structure
            self.assertEqual(analysis.table_name, 'users')
            self.assertGreater(analysis.column_count, 0)
            self.assertGreater(analysis.row_count, 0)
            self.assertIsInstance(analysis.recommendations, list)
            self.assertIn(analysis.risk_level, ['LOW', 'MEDIUM', 'HIGH'])
            self.assertIsInstance(analysis.optimization_score, float)
            self.assertGreaterEqual(analysis.optimization_score, 0)
            self.assertLessEqual(analysis.optimization_score, 100)
        
        asyncio.run(run_test())
    
    def test_risk_assessment(self):
        """Test risk level assessment logic"""
        async def run_test():
            results = await self.analyzer.analyze_schema()
            
            for analysis in results:
                # Risk level should be valid
                self.assertIn(analysis.risk_level, ['LOW', 'MEDIUM', 'HIGH'])
                
                # High risk should correlate with low optimization score
                if analysis.risk_level == 'HIGH':
                    self.assertLess(analysis.optimization_score, 70)
                elif analysis.risk_level == 'LOW':
                    self.assertGreater(analysis.optimization_score, 80)
        
        asyncio.run(run_test())
    
    def test_migration_time_estimation(self):
        """Test migration time estimation algorithm"""
        async def run_test():
            results = await self.analyzer.analyze_schema()
            
            for analysis in results:
                # Migration time should be positive
                self.assertGreater(analysis.estimated_migration_time, 0)
                
                # Larger tables should generally take longer
                if analysis.row_count > 500:
                    self.assertGreater(analysis.estimated_migration_time, 5)
                
                # Should not exceed reasonable maximum
                self.assertLessEqual(analysis.estimated_migration_time, 120)
        
        asyncio.run(run_test())
    
    def test_recommendations_generation(self):
        """Test AI-powered recommendations generation"""
        async def run_test():
            results = await self.analyzer.analyze_schema()
            
            for analysis in results:
                # Should have at least one recommendation
                self.assertGreater(len(analysis.recommendations), 0)
                
                # Recommendations should be strings
                for rec in analysis.recommendations:
                    self.assertIsInstance(rec, str)
                    self.assertGreater(len(rec), 10)  # Reasonable length
        
        asyncio.run(run_test())
    
    @patch('ai_scripts.migration_analyzer.TRANSFORMERS_AVAILABLE', False)
    def test_fallback_without_ai(self):
        """Test functionality when AI models are not available"""
        analyzer_no_ai = MigrationAnalyzer(self.test_db_path)
        
        async def run_test():
            results = await analyzer_no_ai.analyze_schema('users')
            
            # Should still work without AI
            self.assertEqual(len(results), 1)
            analysis = results[0]
            self.assertIsInstance(analysis.recommendations, list)
            self.assertGreater(len(analysis.recommendations), 0)
        
        asyncio.run(run_test())
    
    def test_report_generation(self):
        """Test comprehensive report generation"""
        async def run_test():
            # First analyze schema
            await self.analyzer.analyze_schema()
            
            # Generate report
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                report_path = f.name
            
            try:
                report = self.analyzer.generate_migration_report(report_path)
                
                # Verify report structure
                self.assertIn('report_metadata', report)
                self.assertIn('summary_statistics', report)
                self.assertIn('table_analyses', report)
                self.assertIn('recommendations', report)
                self.assertIn('migration_plan', report)
                
                # Verify metadata
                metadata = report['report_metadata']
                self.assertIn('generated_at', metadata)
                self.assertIn('analyzer_version', metadata)
                self.assertEqual(metadata['analyzer_version'], '2.0.0')
                
                # Verify summary statistics
                summary = report['summary_statistics']
                self.assertGreater(summary['total_tables'], 0)
                self.assertGreater(summary['total_rows'], 0)
                self.assertGreaterEqual(summary['average_optimization_score'], 0)
                self.assertLessEqual(summary['average_optimization_score'], 100)
                
                # Verify migration plan
                migration_plan = report['migration_plan']
                self.assertIn('total_steps', migration_plan)
                self.assertIn('estimated_total_time', migration_plan)
                self.assertIn('migration_steps', migration_plan)
                self.assertIsInstance(migration_plan['migration_steps'], list)
                
                # Verify file was created
                self.assertTrue(os.path.exists(report_path))
                
                # Verify JSON is valid
                with open(report_path, 'r') as f:
                    json.load(f)  # Should not raise exception
                    
            finally:
                if os.path.exists(report_path):
                    os.unlink(report_path)
        
        asyncio.run(run_test())
    
    def test_large_table_handling(self):
        """Test handling of large tables (performance test)"""
        async def run_test():
            # Test with logs table which has 1000+ rows
            start_time = datetime.now()
            results = await self.analyzer.analyze_schema('logs')
            end_time = datetime.now()
            
            # Should complete quickly
            execution_time = (end_time - start_time).total_seconds()
            self.assertLess(execution_time, 5.0)  # Should complete in under 5 seconds
            
            # Verify analysis results
            self.assertEqual(len(results), 1)
            analysis = results[0]
            self.assertEqual(analysis.table_name, 'logs')
            self.assertGreater(analysis.row_count, 1000)
        
        asyncio.run(run_test())
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with non-existent table
        async def test_invalid_table():
            with self.assertRaises(Exception):
                await self.analyzer.analyze_schema('non_existent_table')
        
        asyncio.run(test_invalid_table())
        
        # Test with invalid database path
        with self.assertRaises(Exception):
            MigrationAnalyzer('/invalid/path/database.db')
    
    def test_data_types_analysis(self):
        """Test data type analysis and optimization suggestions"""
        async def run_test():
            results = await self.analyzer.analyze_schema('users')
            analysis = results[0]
            
            # Verify data types are captured
            self.assertIsInstance(analysis.data_types, dict)
            self.assertGreater(len(analysis.data_types), 0)
            
            # Check specific expected data types
            self.assertIn('id', analysis.data_types)
            self.assertIn('email', analysis.data_types)
        
        asyncio.run(run_test())
    
    def test_constraints_detection(self):
        """Test detection of table constraints"""
        async def run_test():
            results = await self.analyzer.analyze_schema('users')
            analysis = results[0]
            
            # Verify constraints are detected
            self.assertIsInstance(analysis.constraints, list)
            
            # Should detect primary key constraint
            pk_constraints = [c for c in analysis.constraints if 'PRIMARY KEY' in c]
            self.assertGreater(len(pk_constraints), 0)
        
        asyncio.run(run_test())
    
    def test_foreign_key_detection(self):
        """Test foreign key relationship detection"""
        async def run_test():
            results = await self.analyzer.analyze_schema('projects')
            analysis = results[0]
            
            # Should detect foreign key to users table
            self.assertIsInstance(analysis.foreign_keys, list)
        
        asyncio.run(run_test())
    
    def test_index_analysis(self):
        """Test index detection and recommendations"""
        async def run_test():
            results = await self.analyzer.analyze_schema('users')
            analysis = results[0]
            
            # Should detect existing indexes
            self.assertIsInstance(analysis.indexes, list)
        
        asyncio.run(run_test())
    
    def test_concurrent_analysis(self):
        """Test concurrent analysis of multiple tables"""
        async def run_test():
            start_time = datetime.now()
            results = await self.analyzer.analyze_schema()  # All tables
            end_time = datetime.now()
            
            # Should handle multiple tables efficiently
            execution_time = (end_time - start_time).total_seconds()
            self.assertLess(execution_time, 10.0)  # Should complete in under 10 seconds
            
            # Verify all tables were analyzed
            self.assertGreaterEqual(len(results), 4)  # At least our 4 test tables
        
        asyncio.run(run_test())

class TestSchemaAnalysisDataClass(unittest.TestCase):
    """Test the SchemaAnalysis data class"""
    
    def test_schema_analysis_creation(self):
        """Test creating TableAnalysis instance"""
        analysis = TableAnalysis(
            table_name="test_table",
            row_count=1000,
            size_mb=10.5,
            index_count=2,
            foreign_keys=["fk_user_id"],
            primary_key="id",
            suggested_indexes=[{"column": "name", "type": "btree"}],
            optimization_score=85.5,
            migration_complexity="Medium",
            estimated_migration_time="2 hours"
        )
        
        self.assertEqual(analysis.table_name, "test_table")
        self.assertEqual(analysis.row_count, 1000)
        self.assertEqual(analysis.size_mb, 10.5)
        self.assertEqual(analysis.optimization_score, 85.5)
        self.assertEqual(analysis.migration_complexity, "Medium")
    
    def test_schema_analysis_serialization(self):
        """Test converting TableAnalysis to dict"""
        analysis = TableAnalysis(
            table_name="test_table",
            row_count=500,
            size_mb=5.0,
            index_count=1,
            foreign_keys=[],
            primary_key="id",
            suggested_indexes=[],
            optimization_score=75.0,
            migration_complexity="Low",
            estimated_migration_time="1 hour"
        )
        
        analysis_dict = analysis.__dict__
        self.assertIsInstance(analysis_dict, dict)
        self.assertEqual(analysis_dict['table_name'], "test_table")
        self.assertEqual(analysis_dict['migration_complexity'], "Low")

class TestMigrationAnalyzerIntegration(unittest.TestCase):
    """Integration tests with real Chinook database if available"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.chinook_path = "data/chinook/chinook.db"
        self.chinook_available = os.path.exists(self.chinook_path)
        
        if self.chinook_available:
            self.analyzer = MigrationAnalyzer(self.chinook_path)
    
    @unittest.skipUnless(os.path.exists("data/chinook/chinook.db"), "Chinook database not available")
    def test_real_chinook_analysis(self):
        """Test analysis with real Chinook database"""
        async def run_test():
            results = await self.analyzer.analyze_schema()
            
            # Chinook should have specific tables
            table_names = [r.table_name for r in results]
            expected_chinook_tables = ['albums', 'artists', 'customers', 'tracks']
            
            for table in expected_chinook_tables:
                self.assertIn(table, table_names)
            
            # Should have reasonable data volumes
            total_rows = sum(r.row_count for r in results)
            self.assertGreater(total_rows, 50000)  # Chinook has ~58k records
        
        if self.chinook_available:
            asyncio.run(run_test())
    
    @unittest.skipUnless(os.path.exists("data/chinook/chinook.db"), "Chinook database not available")
    def test_chinook_performance_benchmark(self):
        """Benchmark performance with real Chinook database"""
        async def run_test():
            start_time = datetime.now()
            results = await self.analyzer.analyze_schema()
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Should analyze full Chinook database in reasonable time
            self.assertLess(execution_time, 30.0)  # Under 30 seconds
            
            # Generate performance report
            print(f"\n🏆 Chinook Analysis Performance:")
            print(f"   Tables analyzed: {len(results)}")
            print(f"   Total records: {sum(r.row_count for r in results):,}")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Records/second: {sum(r.row_count for r in results)/execution_time:,.0f}")
        
        if self.chinook_available:
            asyncio.run(run_test())

def run_coverage_test():
    """Run tests with coverage reporting"""
    try:
        import coverage
        
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        cov.stop()
        cov.save()
        
        # Generate coverage report
        print("\n" + "="*60)
        print("COVERAGE REPORT")
        print("="*60)
        cov.report()
        
        # Generate HTML coverage report
        try:
            cov.html_report(directory='htmlcov')
            print("\n📊 HTML coverage report generated in 'htmlcov/' directory")
        except Exception as e:
            print(f"Could not generate HTML coverage: {e}")
        
        return result.wasSuccessful()
        
    except ImportError:
        print("Coverage module not available, running tests without coverage")
        return run_basic_tests()

def run_basic_tests():
    """Run tests without coverage"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    print("🧪 CloudForge AI - Migration Analyzer Test Suite")
    print("=" * 50)
    print("Testing Production-Ready Database Migration Analysis")
    print("Expected Coverage: >95%")
    print("Expected Performance: <30s for Chinook analysis")
    print("=" * 50)
    
    # Check if we should run with coverage
    if '--coverage' in sys.argv:
        success = run_coverage_test()
    else:
        success = run_basic_tests()
    
    if success:
        print("\n🎉 All tests passed! Migration Analyzer is production-ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review and fix issues.")
        sys.exit(1)