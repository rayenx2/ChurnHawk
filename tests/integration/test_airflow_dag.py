import pytest

try:
    from airflow.models import DagBag

    airflow_installed = True
except ImportError:
    airflow_installed = False
    DagBag = None


@pytest.mark.skipif(not airflow_installed, reason="Airflow not installed in CI environment")
class TestChurnPipelineDAG:

    @pytest.fixture
    def dagbag(self):
        """Load DAG bag"""
        return DagBag(dag_folder="airflow/dags/", include_examples=False)

    def test_dag_loaded(self, dagbag):
        """Test that DAG is loaded without errors"""
        assert dagbag.import_errors == {}
        assert "churn_training_pipeline" in dagbag.dags

    def test_dag_structure(self, dagbag):
        """Test DAG has expected structure"""
        dag = dagbag.get_dag("churn_training_pipeline")

        assert dag is not None
        assert len(dag.tasks) > 0

        task_ids = [task.task_id for task in dag.tasks]
        assert "validate_data" in task_ids
        assert "train_model" in task_ids
        assert "evaluate_model" in task_ids

    def test_dag_dependencies(self, dagbag):
        """Test task dependencies are correct"""
        dag = dagbag.get_dag("churn_training_pipeline")

        validate_task = dag.get_task("validate_data")
        train_task = dag.get_task("train_model")

        assert train_task in validate_task.downstream_list

    def test_dag_schedule(self, dagbag):
        """Test DAG schedule is set correctly"""
        dag = dagbag.get_dag("churn_training_pipeline")

        assert dag.schedule_interval is not None
        assert dag.schedule_interval == "@weekly"

    def test_dag_tags(self, dagbag):
        """Test DAG has appropriate tags"""
        dag = dagbag.get_dag("churn_training_pipeline")

        assert "ml" in dag.tags
        assert "churn" in dag.tags
