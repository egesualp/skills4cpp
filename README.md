# skills4cpp

Leveraging Skills and LLMs for Career Path Prediction

A comprehensive machine learning project for career path prediction using multiple datasets and advanced data processing techniques.

## 🚀 Features

- **Multi-dataset support**: Decorte, Karrierewege, and TaskA datasets
- **Advanced data processing**: Title extraction, document segmentation, and career path analysis
- **Comprehensive testing**: 18 unit tests with 100% coverage of core functionality
- **Production-ready**: Robust error handling and edge case management
- **Multi-language support**: English, German, and Spanish datasets

## 📊 Available Datasets

### 1. **Decorte Dataset** (130 MB)
- **Source**: `jensjorisdecorte/anonymous-working-histories`
- **Content**: Anonymous working histories with ESCO mappings
- **Features**: Career progression data with job transitions

### 2. **Karrierewege Dataset** (4.9 GB)
- **Source**: `ElenaSenger/Karrierewege`
- **Content**: Career pathway data with job transitions
- **Features**: Multi-language support (EN, DE, ES)

### 3. **TaskA Dataset** (13 MB)
- **Source**: Zenodo (https://zenodo.org/records/14693201/files/TaskA.zip)
- **Content**: Job title pairs for similarity tasks
- **Features**: ESCO occupation mappings and job title relationships

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Internet connection (for dataset downloads)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd skills4cpp

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Install additional dependencies for data processing
pip install datasets pandas tqdm
```

## 📖 Usage

### Basic Data Loading
```python
from src.data import Data

# Load Decorte dataset
data = Data('decorte')
train, val, test = data.get_data('embedding_finetuning')

# Load Karrierewege dataset with custom settings
data = Data('karrierewege', ONLY_TITLES=True)
train, val, test = data.get_data('transformation_finetuning')
```

### Supported Dataset Types
- `decorte` - Decorte working histories
- `decorte_esco` - Decorte with ESCO mappings
- `karrierewege` - Karrierewege career paths
- `karrierewege_occ` - Karrierewege occupations
- `karrierewege_100k` - Karrierewege 100k subset
- `karrierewege_cp` - Karrierewege career paths

### Data Processing Stages
- `embedding_finetuning` - Full document pairs
- `transformation_finetuning` - Processed with minus_last
- `evaluation` - Evaluation-ready format

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage
- **18 comprehensive test cases**
- **100% coverage** of core functionality
- **Edge cases and error handling**
- **Static method validation**

## 📁 Project Structure

```
├── src/                      # Source code
│   ├── data.py              # Data loading and processing
│   └── utils.py             # Utility functions
├── tests/                   # Test suite
│   ├── test_data.py         # Data class tests (18 tests)
│   ├── test_api.py          # API tests
│   └── test_model.py        # Model tests
├── data/                    # Data directory
├── models/                  # Trained models
├── notebooks/               # Jupyter notebooks
├── reports/                 # Reports and figures
├── requirements.txt         # Main dependencies
├── requirements_dev.txt     # Development dependencies
└── pyproject.toml          # Project configuration
```

## 🔧 Data Processing Features

### Document Processing
- **Title extraction**: Regex-based job title extraction
- **Document segmentation**: SEP_TOKEN-based splitting
- **Career path analysis**: Multi-step career progression
- **Label extraction**: Unique label identification

### Static Methods
- `__minus_last()`: Remove last segment from documents
- `_extract_titles()`: Extract job titles using regex patterns
- **Error handling**: Robust handling of edge cases

## 📈 Dataset Statistics

| Dataset | Size | Samples | Languages |
|---------|------|---------|-----------|
| Decorte | 130 MB | ~2,164 | English |
| Karrierewege | 4.9 GB | ~455,129 | EN, DE, ES |
| TaskA | 13 MB | 28,880 | EN, DE, ES |

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_dev.txt
   ```

2. **Run tests**:
   ```bash
   pytest tests/test_data.py -v
   ```

3. **Load data**:
   ```python
   from src.data import Data
   data = Data('decorte')
   train, val, test = data.get_data('embedding_finetuning')
   ```

## 📝 Development

### Adding New Datasets
1. Add dataset loading function to `utils.py`
2. Update `Data.__load_data()` method
3. Add corresponding tests
4. Update documentation

### Running Tests
```bash
# Run specific test file
pytest tests/test_data.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run all tests
pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
