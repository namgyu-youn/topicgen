# GitHub Topic Generator

It automatically generates relevant GitHub topics from README.md link. It uses zero-shot to analyze repository content and suggest appropriate topics based on the selected categories.

**⬇️ Sample Image ⬇️**

<img width="900" alt="image" src="https://github.com/Namgyu-Youn/github-tag-generator/blob/main/src/image.png">

![image|width="900"](https://github.com/user-attachments/assets/bbc5b0dd-137c-4d52-876c-3d2e9c3d0659)


```
flowchart TD
    subgraph Gradio Interface
        A[process_url]
    end

    subgraph Fetcher
        B[fetch_readme]
    end

    subgraph Analyzer
        C[generate_topics]
    end

    subgraph Topic Hierarchy
        D[topic_tags]
    end

    A -->|GitHub URL| B
    B -->|README Content| C
    D -->|Category Topics| C
    C -->|Generated Topics| A

    classDef interface fill:#f9f,stroke:#333,stroke-width:2px;
    classDef core fill:#bbf,stroke:#333,stroke-width:2px;
    classDef data fill:#bfb,stroke:#333,stroke-width:2px;

    class A interface;
    class B,C core;
    class D data;
```
## Features
- Analyzes GitHub repository README.md files
- Generates relevant topics based on content analysis
- Supports multiple categories including Data & AI, Scientific Research
- Provides topic recommendations based on category selection
- User-friendly Gradio interface

## Prerequisites
- Python 3.10 or higher
- Docker (optional)
- Poetry (optional)
- transformer

## Installation

### Option 1: Standard Python Setup

1. Clone the repository
```bash
git clone https://github.com/Namgyu-Youn/github-topic-generator.git
cd github-topic-generator
```

2. Create and activate virtual environment
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Option 2: Using Poetry

1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone and install dependencies
```bash
git clone https://github.com/Namgyu-Youn/github-topic-generator.git
cd github-topic-generator
poetry install
```

### Option 3: Using Docker

1. Clone the repository
```bash
git clone https://github.com/yourusername/github-topic-generator.git
cd github-topic-generator
```

2. Build and run with Docker Compose
```bash
docker-compose up --build
```

## Usage

### Running the Application

1. Start the Gradio interface:
```bash
# If using standard Python setup
python gradio_app.py

# If using Poetry
poetry run python gradio_app.py

# If using Docker
# The application will start automatically after docker-compose up
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

### Using the Interface

1. Enter a GitHub README.md URL
2. Select the main category that best matches your repository
3. Choose a sub-category for more specific topic generation
4. Click "Generate Topics" to get your results

## Development

### Project Structure
```
github-topic-generator/
├── gradio_app.py          # Gradio interface
├── topic_gen/
│   ├── analyzer.py        # Topic analysis logic
│   ├── fetcher.py        # GitHub README fetcher
│   └── topic_hierarchy.py # Topic categories
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

### Running Tests
```bash
# Using Poetry
poetry run pytest

# Using standard Python setup
pytest
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. 💡
