# Specialization Benchmarks

## Motivation

Specialization in LLM-based agents concerns the activation of domain-relevant reasoning behaviors through textual conditioning rather than external knowledge acquisition. This experiment measures whether modifying an agent's analytical context through role descriptions, explicit planning opportunities, or expert methodological guidance can activate or better structure the domain-specific reasoning already encoded in the model.

We use concrete machine learning tasks to evaluate how LLM behavior changes under different role conditions (e.g., assigning the agent a *data scientist* role). These tasks are drawn from public datasets that the model has likely encountered during pretraining, ensuring that performance differences reflect changes in reasoning behavior rather than access to external knowledge.

## Dataset Source

The benchmark uses five public machine learning datasets covering regression, binary, and multiclass classification tasks, collected from the CatDB repository:

- **Utility** - Regression task (target: CSRI) - CatDB repository
- **Wifi** - Binary classification (target: TechCenter) - CatDB repository
- **EU-IT** - Multiclass classification (target: Position) - CatDB repository
- **Yelp** - Multiclass classification (target: stars) - [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
- **Volkert** - Multiclass classification (target: class) - [OpenML](https://www.openml.org/search?type=data&status=active&id=41166&sort=runs)

### Dataset Setup

**Dataset Availability**: Some datasets are included in this repository:
- `Utility/Utility.csv` ✓ (included)
- `Wifi/WiFi.csv` ✓ (included, note: filename is `WiFi.csv` with capital letters)
- `EU-IT/EU-IT_cleaned.csv` ✓ (included)
- `Yelp/Yelp_Merged.csv` ✗ (not included - must be downloaded)
- `Volkert/volkert.csv` ✗ (not included - must be downloaded)

**Manual Download Required**: The following datasets must be downloaded manually:

1. **Utility, Wifi, EU-IT**: These datasets from the CatDB repository are included in the repository. If you need to re-download them, obtain them from the CatDB source (specific download links may vary).

2. **Yelp**: Download from the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/). The benchmark uses a merged CSV file (`Yelp_Merged.csv`) created from multiple Yelp dataset files:
   - `Business.csv`
   - `Checkins.csv`
   - `Reviews.csv`
   - `Users.csv`
   
   Place the merged file as `Yelp_Merged.csv` in the `Yelp/` directory.

3. **Volkert**: Download from [OpenML](https://www.openml.org/search?type=data&status=active&id=41166&sort=runs) (dataset ID: 41166). Save it as `volkert.csv` in the `Volkert/` directory.

**Important**: The generated code expects CSV files to be in the **current working directory** when executed. When running the generated scripts:
- Navigate to the dataset's directory (e.g., `cd Utility/`) before executing, OR
- Modify the file paths in the generated scripts to point to the correct dataset location

## Benchmark Structure

Each dataset is evaluated under three conditioning strategies with identical tasks, data, and models:

1. **Role-based prompting** - Assigns a professional identity (e.g., data scientist) alongside the task description
2. **Planning-based conditioning** - Adds an explicit intermediate step where the LLM generates a high-level solution plan before producing executable code
3. **Expert-guided conditioning** - Injects methodological instructions reflecting standard data-science workflows

These variants isolate the effects of identity framing, added reasoning structure, and explicit procedural guidance on agent behavior.

### Implementation Details

**Agent Roles**: The benchmark uses five agent roles defined in `agents.yaml`:
- **Data Scientist**: Focuses on cleaning, preprocessing, and building reproducible ML pipelines
- **Researcher**: Emphasizes pattern discovery, experimentation, and result interpretation
- **Engineer**: Prioritizes reliable, efficient implementations with performance optimization
- **Data Analyst**: Focuses on trend analysis, relationship identification, and clear interpretation
- **No Role**: Baseline condition without role assignment

**Conditioning Strategies**:
- **Role-based**: Implemented via CrewAI agent configurations with role-specific goals and backstories (`crew.py`)
- **Planning-based**: Enabled by setting `planning=True` in the Crew configuration (`crewplanning.py`), which adds explicit step-by-step planning before code generation
- **Expert-guided**: Implemented through detailed task descriptions in `task.yaml` that include explicit methodological workflows (e.g., data profiling, cleaning, feature engineering, modeling, evaluation)

**Task Structure**: Each dataset requires generating executable Python code that:
- Loads the dataset from CSV
- Performs appropriate preprocessing (encoding, scaling, imputation)
- Trains a model (regression or classification depending on task)
- Evaluates using standard metrics (MAE for regression; Accuracy, Precision, Recall, F1-score for classification)
- Outputs results including predicted vs actual values

**Code Generation**: Generated code files follow naming conventions:
- `{Dataset}_{role}.py` - Role-based conditioning
- `{Dataset}_{role}planning.py` - Planning-based conditioning  
- `{Dataset}_{role}exp.py` - Expert-guided conditioning

## Our Contribution

Our contribution is a controlled specialization evaluation framework that isolates the effect of textual conditioning on agent behavior. By holding the model, data, and task specifications fixed and varying only the conditioning strategy, the benchmark enables direct comparison of role-based prompting, planning opportunities, and expert guidance. This design supports principled analysis of when domain-relevant reasoning can be activated through lightweight prompt interventions alone and when stronger procedural structure is required.
