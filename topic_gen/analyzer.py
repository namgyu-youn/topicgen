from transformers import pipeline
import re

class TopicAnalyzer:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification")
        self.candidate_labels = [
            # Programming Languages
            "python", "javascript", "typescript", "java", "cpp", "csharp", "go", "rust", "ruby", "php",
            "swift", "kotlin", "scala", "perl", "haskell", "r", "julia", "dart", "lua", "matlab",

            # Web Development
            "react", "vue", "angular", "svelte", "nextjs", "nuxtjs", "django", "flask", "fastapi",
            "spring-boot", "express", "nodejs", "deno", "graphql", "rest-api", "webassembly",
            "wordpress", "html5", "css3", "sass", "less", "tailwind", "bootstrap", "jquery",

            # Mobile Development
            "android", "ios", "flutter", "react-native", "xamarin", "ionic", "cordova", "kotlin-android",
            "swift-ui", "mobile-app", "pwa",

            # Database & Storage
            "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra", "sqlite", "oracle",
            "mssql", "dynamodb", "firebase", "neo4j", "cockroachdb", "mariadb", "influxdb",

            # Cloud & DevOps
            "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "ansible", "jenkins", "github-actions",
            "gitlab-ci", "circleci", "prometheus", "grafana", "nginx", "apache", "serverless",
            "microservices", "cloud-native", "devops", "devsecops", "monitoring", "logging",

            # AI & Machine Learning
            "machine-learning", "deep-learning", "artificial-intelligence", "neural-networks", "computer-vision",
            "nlp", "reinforcement-learning", "tensorflow", "pytorch", "keras", "scikit-learn", "opencv",
            "transformers", "gans", "chatbot", "mlops", "data-science", "pandas", "numpy", "jupyter",

            # Game Development
            "unity", "unreal-engine", "godot", "gamedev", "game-engine", "3d-graphics", "2d-graphics",
            "vulkan", "opengl", "directx", "physics-engine",

            # Security
            "security", "cryptography", "authentication", "oauth", "jwt", "encryption", "blockchain",
            "smart-contracts", "solidity", "web3", "defi", "penetration-testing", "cybersecurity",

            # Tools & Frameworks
            "git", "webpack", "babel", "vite", "rollup", "eslint", "prettier", "jest", "pytest",
            "selenium", "puppeteer", "cypress", "gradle", "maven", "cmake", "bash", "powershell",

            # Development Concepts
            "api", "sdk", "cli", "gui", "testing", "automation", "ci-cd", "agile", "documentation",
            "package", "library", "framework", "boilerplate", "starter-kit", "template",

            # Emerging Tech
            "iot", "robotics", "ar", "vr", "xr", "quantum-computing", "5g", "edge-computing",
            "cyber-physical", "bioinformatics", "computational-biology",

            # Data & Analytics
            "big-data", "data-analytics", "data-visualization", "business-intelligence", "etl",
            "data-pipeline", "data-warehouse", "data-lake", "spark", "hadoop", "tableau", "power-bi",

            # System & Infrastructure
            "linux", "windows", "macos", "embedded", "rtos", "distributed-systems", "networking",
            "virtualization", "container", "cluster", "fault-tolerance", "high-availability",

            # UX/UI & Design
            "ui", "ux", "design-system", "accessibility", "responsive-design", "mobile-first",
            "web-components", "material-design", "animation", "svg", "canvas", "webgl",

            # Development Practices
            "clean-code", "design-patterns", "architecture", "refactoring", "performance",
            "optimization", "scalability", "maintainability", "code-quality", "best-practices",

            # Domain Specific
            "fintech", "healthtech", "edtech", "e-commerce", "social-media", "streaming",
            "geospatial", "audio", "video", "image-processing", "multimedia", "3d-printing"
        ]

    def extract_keywords(self, text: str) -> list[str]:
        """Extract potential keywords from text."""
        # Remove code blocks and URLs
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'http\S+', '', text)

        # Extract words that might be topics
        words = re.findall(r'[A-Za-z]+(?:-[A-Za-z]+)*', text.lower())
        return list(set(words))

    async def generate_topics(self, text: str) -> list[str]:
        """Generate topics from README content."""
        keywords = self.extract_keywords(text)

        # Get top matching labels using zero-shot classification
        result = self.classifier(text[:512], self.candidate_labels, multi_label=True)

        # Filter labels with high confidence
        topics = [
            label for label, score in zip(result["labels"], result["scores"])
            if score > 0.6  # Increased confidence threshold
        ]

        # Add exact matches from keywords
        exact_matches = [
            word for word in keywords
            if word in self.candidate_labels
        ]

        # Add technology-specific patterns
        tech_patterns = [
            word for word in keywords
            if any(tech in word for tech in ["js", "py", "api", "sdk", "ml", "ai", "db"])
        ]

        return list(set(topics + exact_matches + tech_patterns))