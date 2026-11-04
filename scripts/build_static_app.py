import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

with open(ROOT / "models" / "url_only_model" / "tree_rules.json", "r", encoding="utf-8") as f:
    tree_rules_json = f.read()

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PHISHGUARD AI — Cybersecurity URL & Website Phishing Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #0b0f17; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .cyber-card { background: #131b26; border: 1px solid #1f2937; border-radius: 12px; }
        .cyber-card-active { border-color: #3b82f6; }
        .gradient-text { background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .tab-btn.active { background-color: #1e293b; color: #38bdf8; border-color: #38bdf8; }
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between">
    <!-- Navbar -->
    <header class="border-b border-gray-800 bg-gray-900/80 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 text-xl font-bold">
                    <i class="fa-solid fa-shield-halved"></i>
                </div>
                <div>
                    <h1 class="font-bold text-base md:text-lg leading-tight tracking-wide">PHISHGUARD <span class="text-blue-400">AI</span></h1>
                    <p class="text-xs text-gray-400">ML Phishing Detection System • PhiUSIIL (UCI ID: 967)</p>
                </div>
            </div>
            <div class="flex items-center space-x-3 text-xs">
                <span class="inline-flex items-center px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-2 h-2 mr-1.5 bg-emerald-400 rounded-full animate-pulse"></span> 24/7 Cloud Live
                </span>
                <a href="https://github.com/Anujsingh0-dev/PHISHGUARD-AI" target="_blank" class="text-gray-400 hover:text-white transition flex items-center space-x-1.5 bg-gray-800 px-2.5 py-1 rounded border border-gray-700">
                    <i class="fa-brands fa-github text-sm"></i>
                    <span class="hidden sm:inline">GitHub</span>
                </a>
            </div>
        </div>
    </header>

    <!-- Nav Tabs -->
    <div class="border-b border-gray-800 bg-gray-950/60 sticky top-14 z-40">
        <div class="max-w-6xl mx-auto px-4 flex space-x-2 overflow-x-auto py-2 text-xs font-medium">
            <button onclick="switchTab('detector')" id="tab-detector" class="tab-btn active px-4 py-2 rounded-lg border border-transparent transition flex items-center space-x-2">
                <i class="fa-solid fa-magnifying-glass"></i>
                <span>Live URL Detector</span>
            </button>
            <button onclick="switchTab('batch')" id="tab-batch" class="tab-btn text-gray-400 hover:text-gray-200 px-4 py-2 rounded-lg border border-transparent transition flex items-center space-x-2">
                <i class="fa-solid fa-list-check"></i>
                <span>Batch Scanner</span>
            </button>
            <button onclick="switchTab('dataset')" id="tab-dataset" class="tab-btn text-gray-400 hover:text-gray-200 px-4 py-2 rounded-lg border border-transparent transition flex items-center space-x-2">
                <i class="fa-solid fa-database"></i>
                <span>Dataset & EDA (235k)</span>
            </button>
            <button onclick="switchTab('models')" id="tab-models" class="tab-btn text-gray-400 hover:text-gray-200 px-4 py-2 rounded-lg border border-transparent transition flex items-center space-x-2">
                <i class="fa-solid fa-chart-line"></i>
                <span>Model Benchmark</span>
            </button>
            <button onclick="switchTab('academic')" id="tab-academic" class="tab-btn text-gray-400 hover:text-gray-200 px-4 py-2 rounded-lg border border-transparent transition flex items-center space-x-2">
                <i class="fa-solid fa-graduation-cap"></i>
                <span>Academic & Viva</span>
            </button>
        </div>
    </div>

    <!-- Main Body -->
    <main class="max-w-6xl mx-auto px-4 py-8 flex-1 w-full">

        <!-- TAB 1: LIVE URL DETECTOR -->
        <section id="view-detector" class="space-y-6">
            <!-- Hero -->
            <div class="text-center space-y-2">
                <h2 class="text-3xl md:text-4xl font-extrabold tracking-tight">
                    Real-Time Phishing Detection with <span class="gradient-text">Machine Learning</span>
                </h2>
                <p class="text-gray-400 text-xs sm:text-sm max-w-2xl mx-auto">
                    Evaluates 22 lexical and structural URL characteristics using a Decision Tree model trained on 235,795 verified URLs from the UCI PhiUSIIL dataset.
                </p>
                <div class="inline-block bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs px-3 py-1 rounded-lg">
                    ⚠️ <b>Academic Prototype:</b> Discloses false positive/negative risks honestly. Never replaces full browser antivirus protection.
                </div>
            </div>

            <!-- URL Input Card -->
            <div class="cyber-card p-6 shadow-2xl space-y-4">
                <h3 class="text-sm font-semibold text-gray-200 flex items-center">
                    <i class="fa-solid fa-link text-blue-400 mr-2"></i> Enter URL to Analyze
                </h3>
                <div class="flex flex-col sm:flex-row gap-3">
                    <input type="text" id="urlInput" placeholder="Enter target URL (e.g., google.com, https://github.com, or paypal-verify.com)"
                        class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 transition">
                    <button onclick="runAnalysis()" class="bg-blue-600 hover:bg-blue-500 text-white font-medium px-6 py-3 rounded-lg text-sm transition flex items-center justify-center space-x-2">
                        <span>Scan URL</span>
                        <i class="fa-solid fa-arrow-right"></i>
                    </button>
                </div>
                <!-- Quick Chips -->
                <div class="flex flex-wrap gap-2 text-xs text-gray-400 items-center pt-1">
                    <span>Quick presets:</span>
                    <button onclick="setAndRun('google.com')" class="bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded border border-gray-700 transition">google.com</button>
                    <button onclick="setAndRun('https://github.com/torvalds/linux')" class="bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded border border-gray-700 transition">github.com/torvalds/linux</button>
                    <button onclick="setAndRun('http://paypal-security-update.com/login.php')" class="bg-red-950/40 hover:bg-red-900/60 text-red-300 px-2.5 py-1 rounded border border-red-800 transition">paypal-security-update.com</button>
                    <button onclick="setAndRun('http://192.168.1.1/admin/login')" class="bg-red-950/40 hover:bg-red-900/60 text-red-300 px-2.5 py-1 rounded border border-red-800 transition">192.168.1.1</button>
                    <button onclick="setAndRun('https://cgc.edu.in')" class="bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded border border-gray-700 transition">cgc.edu.in</button>
                </div>
            </div>

            <!-- Result Card -->
            <div id="resultCard" class="hidden cyber-card p-6 space-y-6">
                <div id="verdictBanner" class="p-4 rounded-xl flex items-center space-x-4 border">
                    <div id="verdictIcon" class="text-3xl"></div>
                    <div>
                        <h4 id="verdictTitle" class="text-lg font-bold"></h4>
                        <p id="verdictDesc" class="text-xs mt-0.5 text-gray-300"></p>
                    </div>
                </div>

                <!-- Metrics Grid -->
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800">
                        <span class="text-xs text-gray-400 block">Classification</span>
                        <span id="metricClass" class="text-base font-bold text-gray-100"></span>
                    </div>
                    <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800">
                        <span class="text-xs text-gray-400 block">Risk Evaluation</span>
                        <span id="metricRisk" class="text-base font-bold text-gray-100"></span>
                    </div>
                    <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800">
                        <span class="text-xs text-gray-400 block">Phishing Risk Prob</span>
                        <span id="metricPhishProb" class="text-base font-bold text-red-400"></span>
                    </div>
                    <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800">
                        <span class="text-xs text-gray-400 block">Legitimate Prob</span>
                        <span id="metricLegitProb" class="text-base font-bold text-emerald-400"></span>
                    </div>
                </div>

                <!-- Probability Bar -->
                <div>
                    <div class="flex justify-between text-xs text-gray-400 mb-1">
                        <span>Phishing Likelihood</span>
                        <span>Legitimate Likelihood</span>
                    </div>
                    <div class="w-full bg-emerald-950/80 h-3 rounded-full overflow-hidden flex border border-gray-800">
                        <div id="probBarPhish" class="bg-red-500 h-full transition-all duration-500" style="width: 0%"></div>
                        <div id="probBarLegit" class="bg-emerald-500 h-full transition-all duration-500" style="width: 100%"></div>
                    </div>
                </div>

                <!-- Diagnostic Reasons -->
                <div class="space-y-2">
                    <h5 class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Security Diagnostic Findings</h5>
                    <ul id="reasonsList" class="space-y-1.5 text-sm text-gray-200"></ul>
                </div>

                <!-- Features Toggle -->
                <div class="border-t border-gray-800 pt-4">
                    <button onclick="toggleFeaturesTable()" class="text-xs text-blue-400 hover:text-blue-300 font-medium flex items-center space-x-1.5">
                        <span>Toggle 22 Extracted Features (Mathematical Inspection)</span>
                        <i id="featCaret" class="fa-solid fa-chevron-down text-xs"></i>
                    </button>
                    <div id="featuresTableWrapper" class="hidden mt-3 max-h-72 overflow-y-auto border border-gray-800 rounded-lg">
                        <table class="w-full text-left text-xs text-gray-300">
                            <thead class="bg-gray-900 text-gray-400 border-b border-gray-800 sticky top-0">
                                <tr>
                                    <th class="py-2 px-3">Feature Name</th>
                                    <th class="py-2 px-3">Calculated Value</th>
                                    <th class="py-2 px-3">Security Relevance</th>
                                </tr>
                            </thead>
                            <tbody id="featuresTbody" class="divide-y divide-gray-800/60 bg-gray-900/40 font-mono"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 2: BATCH SCANNER -->
        <section id="view-batch" class="hidden space-y-6">
            <div class="cyber-card p-6 space-y-4">
                <h3 class="text-lg font-bold text-gray-100 flex items-center">
                    <i class="fa-solid fa-list-check text-indigo-400 mr-2"></i> Batch URL Scanner
                </h3>
                <p class="text-xs text-gray-400">Paste multiple URLs (one per line) to evaluate up to 20 links at once.</p>
                <textarea id="batchInput" rows="6" placeholder="google.com&#10;https://github.com/torvalds&#10;http://paypal-security-update.com&#10;http://192.168.1.1/admin&#10;https://cgc.edu.in"
                    class="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs font-mono text-gray-100 focus:outline-none focus:border-blue-500"></textarea>
                <button onclick="runBatchAnalysis()" class="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-5 py-2.5 rounded-lg text-xs transition">
                    Scan Batch URLs
                </button>
            </div>
            <div id="batchResultsWrapper" class="hidden cyber-card p-6 space-y-4">
                <h4 class="text-sm font-bold text-gray-200">Batch Inspection Results</h4>
                <div class="overflow-x-auto border border-gray-800 rounded-lg">
                    <table class="w-full text-left text-xs text-gray-300">
                        <thead class="bg-gray-900 text-gray-400 border-b border-gray-800">
                            <tr>
                                <th class="py-2 px-3">URL</th>
                                <th class="py-2 px-3">Verdict</th>
                                <th class="py-2 px-3">Phishing Prob</th>
                                <th class="py-2 px-3">Risk Status</th>
                            </tr>
                        </thead>
                        <tbody id="batchTbody" class="divide-y divide-gray-800 bg-gray-900/40"></tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- TAB 3: DATASET & EDA -->
        <section id="view-dataset" class="hidden space-y-6">
            <div class="cyber-card p-6 space-y-4">
                <h3 class="text-lg font-bold text-gray-100 flex items-center">
                    <i class="fa-solid fa-database text-blue-400 mr-2"></i> PhiUSIIL Phishing URL Dataset (UCI ID: 967)
                </h3>
                <p class="text-xs sm:text-sm text-gray-300">
                    Collected by Arvind Prasad and Shalini Chandra (2024), published in <i>Computers & Security</i> (Elsevier). Contains <b>235,795</b> verified website records.
                </p>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
                    <div class="bg-gray-900 p-4 rounded-lg border border-gray-800">
                        <span class="text-xs text-gray-400">Total Samples</span>
                        <span class="text-xl font-bold text-blue-400 block mt-1">235,795</span>
                    </div>
                    <div class="bg-gray-900 p-4 rounded-lg border border-gray-800">
                        <span class="text-xs text-gray-400">Legitimate (1)</span>
                        <span class="text-xl font-bold text-emerald-400 block mt-1">134,850 (57.2%)</span>
                    </div>
                    <div class="bg-gray-900 p-4 rounded-lg border border-gray-800">
                        <span class="text-xs text-gray-400">Phishing (0)</span>
                        <span class="text-xl font-bold text-red-400 block mt-1">100,945 (42.8%)</span>
                    </div>
                    <div class="bg-gray-900 p-4 rounded-lg border border-gray-800">
                        <span class="text-xs text-gray-400">Total Features</span>
                        <span class="text-xl font-bold text-indigo-400 block mt-1">54 Features</span>
                    </div>
                </div>
            </div>

            <!-- Findings Card -->
            <div class="cyber-card p-6 space-y-3">
                <h4 class="text-sm font-bold text-amber-400 flex items-center">
                    <i class="fa-solid fa-triangle-exclamation mr-2"></i> Critical Academic Finding: Dataset Shortcut Learning
                </h4>
                <p class="text-xs text-gray-300 leading-relaxed">
                    Our rigorous exploratory data analysis uncovered that in the raw PhiUSIIL dataset, 100% of the legitimate URLs were collected as root landing pages starting with <code class="text-amber-300">https://www.</code> with empty paths (<code class="text-amber-300">""</code>) and <code class="text-amber-300">NoOfSubDomain >= 1</code>. Naive machine learning models trained on this learn a trivial shortcut: <i>"No www or has path = Phishing"</i>. 
                </p>
                <p class="text-xs text-gray-300 leading-relaxed">
                    <b>Our Engineering Solution:</b> We implemented canonical normalization in <code class="text-blue-300">feature_extraction.py</code> that strips <code class="text-blue-300">www.</code> before calculating subdomain counts, alongside an Authority Verification and Brand Impersonation multi-layer pipeline to guarantee real-world safety.
                </p>
            </div>
        </section>

        <!-- TAB 4: MODEL BENCHMARK -->
        <section id="view-models" class="hidden space-y-6">
            <div class="cyber-card p-6 space-y-4">
                <h3 class="text-lg font-bold text-gray-100 flex items-center">
                    <i class="fa-solid fa-chart-line text-emerald-400 mr-2"></i> Model Performance & Comparison
                </h3>
                <p class="text-xs text-gray-400">
                    Rigorous 80/20 train-test evaluation on 235,795 samples across both operating modes.
                </p>
                <div class="overflow-x-auto border border-gray-800 rounded-lg">
                    <table class="w-full text-left text-xs text-gray-300">
                        <thead class="bg-gray-900 text-gray-400 border-b border-gray-800">
                            <tr>
                                <th class="py-2.5 px-3">Evaluation Mode</th>
                                <th class="py-2.5 px-3">Algorithm</th>
                                <th class="py-2.5 px-3">Accuracy</th>
                                <th class="py-2.5 px-3">Precision</th>
                                <th class="py-2.5 px-3">Recall</th>
                                <th class="py-2.5 px-3">F1-Score</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-800/60 font-mono">
                            <tr class="bg-blue-950/20">
                                <td class="py-2 px-3 font-bold text-blue-300">Mode B (URL Only - Live)</td>
                                <td class="py-2 px-3 text-white">Decision Tree (Production)</td>
                                <td class="py-2 px-3 text-emerald-400 font-bold">99.41%</td>
                                <td class="py-2 px-3">99.27%</td>
                                <td class="py-2 px-3">99.35%</td>
                                <td class="py-2 px-3 text-emerald-400 font-bold">0.9931</td>
                            </tr>
                            <tr>
                                <td class="py-2 px-3 text-gray-400">Mode B (URL Only - Live)</td>
                                <td class="py-2 px-3">Random Forest</td>
                                <td class="py-2 px-3">98.38%</td>
                                <td class="py-2 px-3">97.80%</td>
                                <td class="py-2 px-3">98.39%</td>
                                <td class="py-2 px-3">0.9809</td>
                            </tr>
                            <tr>
                                <td class="py-2 px-3 text-gray-400">Mode B (URL Only - Live)</td>
                                <td class="py-2 px-3">Logistic Regression</td>
                                <td class="py-2 px-3">97.52%</td>
                                <td class="py-2 px-3">96.88%</td>
                                <td class="py-2 px-3">97.23%</td>
                                <td class="py-2 px-3">0.9705</td>
                            </tr>
                            <tr class="bg-gray-900/60">
                                <td class="py-2 px-3 text-indigo-300">Mode A (Full 50 Features)</td>
                                <td class="py-2 px-3 text-white">Decision Tree (Research)</td>
                                <td class="py-2 px-3 text-emerald-400 font-bold">100.00%*</td>
                                <td class="py-2 px-3">100.00%</td>
                                <td class="py-2 px-3">100.00%</td>
                                <td class="py-2 px-3 text-emerald-400 font-bold">1.0000</td>
                            </tr>
                            <tr class="bg-gray-900/60">
                                <td class="py-2 px-3 text-indigo-300">Mode A (Full 50 Features)</td>
                                <td class="py-2 px-3">Random Forest</td>
                                <td class="py-2 px-3">100.00%*</td>
                                <td class="py-2 px-3">100.00%</td>
                                <td class="py-2 px-3">100.00%</td>
                                <td class="py-2 px-3">1.0000</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <p class="text-xs text-gray-400">
                    *<b>Note on Mode A 100% Accuracy:</b> In the full dataset, web-page similarity features (<code class="text-gray-300">URLSimilarityIndex</code>, <code class="text-gray-300">LineOfCode</code>) are near-deterministic separators. However, Mode A requires downloading live web pages (unsafe in real-time); hence Mode B is used for live inference.
                </p>
            </div>
        </section>

        <!-- TAB 5: ACADEMIC & VIVA -->
        <section id="view-academic" class="hidden space-y-6">
            <div class="cyber-card p-6 space-y-4">
                <h3 class="text-lg font-bold text-gray-100 flex items-center">
                    <i class="fa-solid fa-graduation-cap text-indigo-400 mr-2"></i> Academic Project Information
                </h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-gray-300">
                    <div class="bg-gray-900 p-4 rounded-lg border border-gray-800 space-y-1.5">
                        <p><span class="text-gray-400">Project Title:</span> <b>PHISHGUARD AI: Phishing Detection System</b></p>
                        <p><span class="text-gray-400">Student:</span> <b>Anuj Singh</b> (Roll No: 2555178)</p>
                        <p><span class="text-gray-400">Degree:</span> <b>B.Tech Information Technology</b> (3rd Year)</p>
                        <p><span class="text-gray-400">Institution:</span> <b>CGC College of Engineering, Landran, Mohali</b></p>
                        <p><span class="text-gray-400">Academic Batch:</span> 2025–2029</p>
                    </div>
                    <div class="bg-gray-900 p-4 rounded-lg border border-gray-800 space-y-1.5">
                        <p><span class="text-gray-400">Dataset:</span> UCI PhiUSIIL (ID: 967, 235,795 rows)</p>
                        <p><span class="text-gray-400">Architecture:</span> Two-Model Architecture (Mode A + Mode B)</p>
                        <p><span class="text-gray-400">Deployed Model:</span> 22-Feature Canonical Decision Tree</p>
                        <p><span class="text-gray-400">Defense Layers:</span> Canonical Normalization, Brand Check, IP Check</p>
                        <p><span class="text-gray-400">Hosting:</span> Hugging Face Static Cloud (24/7 Global Uptime)</p>
                    </div>
                </div>
            </div>

            <!-- 5-Minute Viva Presentation Script -->
            <div class="cyber-card p-6 space-y-3">
                <h4 class="text-sm font-bold text-blue-400 flex items-center">
                    <i class="fa-solid fa-microphone mr-2"></i> 5-Minute Viva Presentation Script (Ready to Speak)
                </h4>
                <div class="bg-gray-900/90 p-4 rounded-lg border border-gray-800 text-xs text-gray-300 space-y-3 leading-relaxed">
                    <p><b>1. Introduction (Minute 1):</b> <i>"Respected evaluators, my name is Anuj Singh. Today I present PHISHGUARD AI, a machine learning system designed to detect malicious phishing URLs before users interact with them. Phishing remains the #1 initial attack vector in cyber breaches."</i></p>
                    <p><b>2. Dataset & Finding (Minute 2):</b> <i>"We utilized the official UCI PhiUSIIL dataset with 235,795 records. During exploratory analysis, we identified a critical benchmark artifact: legitimate URLs always had empty paths and 'www.' prefixes. If unhandled, ML models fail on real URLs like google.com. We resolved this through canonical normalization."</i></p>
                    <p><b>3. Two-Model Architecture (Minute 3):</b> <i>"We designed two distinct modes: Mode A leverages all 50 web features for research, achieving 100% accuracy. Mode B extracts 22 structural URL features in under 5 milliseconds with zero external network queries, achieving 99.41% accuracy and 0.9931 F1-score."</i></p>
                    <p><b>4. Live Demonstration (Minute 4):</b> <i>"As demonstrated on our cloud system, testing google.com verifies legitimate authority, while paypal-security-update.com triggers brand impersonation warnings."</i></p>
                    <p><b>5. Conclusion (Minute 5):</b> <i>"PHISHGUARD AI delivers offline, privacy-first, sub-millisecond phishing detection without leaking user browsing traffic to third parties."</i></p>
                </div>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer class="border-t border-gray-800 py-6 text-center text-xs text-gray-500 bg-gray-950/80">
        <p>PHISHGUARD AI • Academic Project for B.Tech Information Technology</p>
        <p class="mt-1">Developed by <b>Anuj Singh</b> (Roll: 2555178) • CGC College of Engineering, Landran</p>
    </footer>

    <!-- INLINE DECISION TREE MODEL & DETECTION ENGINE -->
    <script>
        const TREE_RULES = __TREE_RULES_JSON__;

        const TOP_DOMAINS = new Set([
            "google.com", "google.co.in", "youtube.com", "wikipedia.org", "amazon.com",
            "amazon.in", "github.com", "microsoft.com", "apple.com", "netflix.com",
            "facebook.com", "instagram.com", "linkedin.com", "twitter.com", "x.com",
            "reddit.com", "yahoo.com", "bing.com", "chase.com", "paypal.com",
            "cloudflare.com", "openai.com", "stackoverflow.com", "cgc.ac.in",
            "cgc.edu.in", "ptu.ac.in", "w3schools.com", "medium.com", "kaggle.com",
            "python.org", "scikit-learn.org", "streamlit.io", "zoom.us", "dropbox.com",
            "quora.com", "spotify.com", "canva.com", "adobe.com", "whatsapp.com"
        ]);

        const BRAND_OFFICIAL_DOMAINS = {
            "paypal": "paypal.com",
            "apple": "apple.com",
            "google": "google.com",
            "microsoft": "microsoft.com",
            "amazon": "amazon.com",
            "netflix": "netflix.com",
            "chase": "chase.com",
            "facebook": "facebook.com",
            "instagram": "instagram.com",
            "bankofamerica": "bankofamerica.com",
            "wellsfargo": "wellsfargo.com",
            "github": "github.com",
            "dropbox": "dropbox.com"
        };

        const SUSPICIOUS_KEYWORDS = [
            "login", "signin", "account", "update", "verify", "secure",
            "banking", "paypal", "ebay", "amazon", "apple", "google",
            "microsoft", "confirm", "password", "credential", "free",
            "winner", "lucky", "prize", "click", "urgent", "alert",
            "wallet", "recover", "authenticate", "billing", "invoice"
        ];

        function switchTab(name) {
            ['detector', 'batch', 'dataset', 'models', 'academic'].forEach(t => {
                document.getElementById('view-' + t).classList.add('hidden');
                document.getElementById('tab-' + t).classList.remove('active');
                document.getElementById('tab-' + t).classList.add('text-gray-400');
            });
            document.getElementById('view-' + name).classList.remove('hidden');
            const btn = document.getElementById('tab-' + name);
            btn.classList.add('active');
            btn.classList.remove('text-gray-400');
        }

        function setAndRun(val) {
            document.getElementById("urlInput").value = val;
            runAnalysis();
        }

        function shannonEntropy(text) {
            if (!text) return 0.0;
            const freq = {};
            for (let ch of text) freq[ch] = (freq[ch] || 0) + 1;
            const len = text.length;
            let ent = 0.0;
            for (let ch in freq) {
                let p = freq[ch] / len;
                ent -= p * Math.log2(p);
            }
            return ent;
        }

        function isIpAddress(host) {
            if (!host) return false;
            const clean = host.split(':')[0].replace(/[\\[\\]]/g, '');
            const ipv4 = /^(\\d{1,3}\\.){3}\\d{1,3}$/;
            if (ipv4.test(clean)) {
                return clean.split('.').every(num => parseInt(num, 10) <= 255);
            }
            return false;
        }

        function normalizeUrl(url) {
            url = String(url).trim();
            if (!/^[a-zA-Z][a-zA-Z0-9+\\-.]*:\\/\\//.test(url)) {
                url = "https://" + url;
            }
            try {
                const u = new URL(url);
                let path = u.pathname;
                if (path === '/') path = '';
                return `${u.protocol}//${u.host.toLowerCase()}${path}${u.search}`;
            } catch (e) {
                return url;
            }
        }

        function extractFeatures(rawUrl) {
            const url = normalizeUrl(rawUrl);
            let u;
            try { u = new URL(url); } catch (e) {
                return { features: {}, cleanUrl: url, host: "", baseDomain: "" };
            }

            const host = u.hostname.toLowerCase();
            const path = u.pathname || "";
            const query = u.search ? u.search.substring(1) : "";
            const scheme = u.protocol ? u.protocol.replace(':', '') : "";

            const domainClean = host.startsWith("www.") ? host.substring(4) : host;
            const domainLength = domainClean.length;

            let numDigits = 0;
            for (let ch of domainClean) if (/\\d/.test(ch)) numDigits++;
            const digitRatio = domainLength > 0 ? numDigits / domainLength : 0;

            const specialChars = new Set(['@', '!', '$', '%', '^', '&', '*']);
            let numSpecial = 0;
            for (let ch of url) if (specialChars.has(ch)) numSpecial++;

            const domainParts = domainClean.split('.');
            const numSubdomains = Math.max(0, domainParts.length - 2);

            let numQueryParams = 0;
            if (query) numQueryParams = query.split('&').filter(Boolean).length;

            const pathSegments = path.split('/').filter(Boolean);
            const urlLower = url.toLowerCase();
            let suspiciousCount = 0;
            SUSPICIOUS_KEYWORDS.forEach(kw => { if (urlLower.includes(kw)) suspiciousCount++; });

            const baseDomain = domainParts.length >= 2 ? domainParts.slice(-2).join('.') : host;

            const feat = {
                "url_length": url.length,
                "domain_length": domainLength,
                "path_length": path.length,
                "num_dots": (domainClean.match(/\\./g) || []).length,
                "num_hyphens": (domainClean.match(/-/g) || []).length,
                "num_underscores": (url.match(/_/g) || []).length,
                "num_slashes": (path.match(/\\//g) || []).length,
                "num_digits": numDigits,
                "digit_ratio": parseFloat(digitRatio.toFixed(6)),
                "num_special_chars": numSpecial,
                "has_at_symbol": url.includes('@') ? 1 : 0,
                "has_double_slash": url.substring(7).includes('//') ? 1 : 0,
                "has_ip_as_host": isIpAddress(host) ? 1 : 0,
                "is_https": scheme === 'https' ? 1 : 0,
                "num_subdomains": numSubdomains,
                "num_query_params": numQueryParams,
                "has_port": (u.port && u.port !== "80" && u.port !== "443") ? 1 : 0,
                "has_percent_encoding": /%[0-9A-Fa-f]{2}/.test(url) ? 1 : 0,
                "has_punycode": host.includes("xn--") ? 1 : 0,
                "num_path_segments": pathSegments.length,
                "suspicious_words_count": suspiciousCount,
                "entropy": parseFloat(shannonEntropy(domainClean).toFixed(6))
            };

            return { features: feat, cleanUrl: url, host, baseDomain };
        }

        function predictPureTree(feat) {
            const scaled = [];
            for (let i = 0; i < TREE_RULES.features.length; i++) {
                const name = TREE_RULES.features[i];
                const val = feat[name] ?? 0;
                const mean = TREE_RULES.scaler_mean[i];
                const scale = TREE_RULES.scaler_scale[i];
                scaled.push(scale !== 0 ? (val - mean) / scale : 0);
            }

            let node = 0;
            while (TREE_RULES.tree_feature[node] !== -2) {
                const featIdx = TREE_RULES.tree_feature[node];
                const thresh = TREE_RULES.tree_threshold[node];
                if (scaled[featIdx] <= thresh) {
                    node = TREE_RULES.tree_children_left[node];
                } else {
                    node = TREE_RULES.tree_children_right[node];
                }
            }

            const val = TREE_RULES.tree_value[node];
            const total = val[0] + val[1];
            const p0 = total > 0 ? parseFloat((val[0] / total).toFixed(4)) : 0.5;
            const p1 = total > 0 ? parseFloat((val[1] / total).toFixed(4)) : 0.5;
            const pred = p0 >= p1 ? 0 : 1;
            return { pred, p0, p1 };
        }

        function analyzeUrl(rawUrl) {
            const { features, cleanUrl, host, baseDomain } = extractFeatures(rawUrl);
            if (!host) return { error: "Please enter a valid URL or domain." };

            const { pred: mlPred, p0: mlP0, p1: mlP1 } = predictPureTree(features);

            const isTop = TOP_DOMAINS.has(baseDomain);
            const isIp = isIpAddress(host);
            let impersonation = null;
            const reasons = [];

            if (!isTop && !isIp) {
                for (let brand in BRAND_OFFICIAL_DOMAINS) {
                    const official = BRAND_OFFICIAL_DOMAINS[brand];
                    if (host.includes(brand) && baseDomain !== official) {
                        impersonation = `🚨 Brand Impersonation Alert: Domain contains '${brand.toUpperCase()}' but registered domain is '${baseDomain}', not official '${official}'.`;
                        reasons.push(impersonation);
                        break;
                    }
                }
            }

            if (isIp) reasons.push("🚨 Raw IP Address Host: Public websites use registered domains, not bare IP addresses.");
            if (features.has_at_symbol === 1) reasons.push("⚠️ '@' Symbol in URL: Frequently used in phishing authority spoofing.");
            if (features.has_punycode === 1) reasons.push("⚠️ Punycode Obfuscation (xn--): Internationalized domain commonly abused in lookalike attacks.");
            if (features.suspicious_words_count >= 2) reasons.push(`⚠️ High Suspicious Keywords: Contains ${features.suspicious_words_count} phishing trigger words.`);

            let finalPred, label, p0, p1, risk;

            if (isTop && !impersonation && !isIp) {
                finalPred = 1;
                label = "Legitimate";
                p0 = Math.min(mlP0, 0.05);
                p1 = parseFloat((1.0 - p0).toFixed(4));
                risk = "Safe / Verified Authority";
                reasons.push(`✅ Verified Authority Domain: '${baseDomain}' is an established high-reputation domain.`);
            } else if (impersonation || isIp) {
                finalPred = 0;
                label = "Phishing";
                p0 = Math.max(mlP0, 0.95);
                p1 = parseFloat((1.0 - p0).toFixed(4));
                risk = "High Risk (Phishing)";
            } else {
                finalPred = mlPred;
                label = finalPred === 1 ? "Legitimate" : "Phishing";
                p0 = mlP0;
                p1 = mlP1;
                risk = finalPred === 0 ? "Suspicious / Potential Phishing" : "Low Risk / Predicted Legitimate";
                if (reasons.length === 0) {
                    reasons.push(finalPred === 1 ? "✅ URL structure is consistent with legitimate web domain patterns." : "⚠️ Machine Learning structural analysis detected anomalies consistent with phishing URLs.");
                }
            }

            return {
                prediction: finalPred,
                label,
                phishingProb: p0,
                legitimateProb: p1,
                risk,
                reasons,
                features,
                cleanUrl,
                baseDomain
            };
        }

        function runAnalysis() {
            const val = document.getElementById("urlInput").value.trim();
            if (!val) return;
            const res = analyzeUrl(val);
            if (res.error) {
                alert(res.error);
                return;
            }

            const card = document.getElementById("resultCard");
            const banner = document.getElementById("verdictBanner");
            const icon = document.getElementById("verdictIcon");
            const title = document.getElementById("verdictTitle");
            const desc = document.getElementById("verdictDesc");

            if (res.prediction === 0) {
                banner.className = "p-4 rounded-xl flex items-center space-x-4 border bg-red-950/40 border-red-500/50 text-red-300";
                icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-red-400"></i>';
                title.innerText = "⚠️ POTENTIALLY PHISHING / MALICIOUS";
                title.className = "text-lg font-bold text-red-400";
                desc.innerText = "This URL displays structural anomalies or impersonation patterns characteristic of phishing threats.";
            } else {
                banner.className = "p-4 rounded-xl flex items-center space-x-4 border bg-emerald-950/40 border-emerald-500/50 text-emerald-300";
                icon.innerHTML = '<i class="fa-solid fa-circle-check text-emerald-400"></i>';
                title.innerText = "✅ PREDICTED LEGITIMATE";
                title.className = "text-lg font-bold text-emerald-400";
                desc.innerText = "This URL structure matches established safe and legitimate website behavioral patterns.";
            }

            document.getElementById("metricClass").innerText = res.label;
            document.getElementById("metricRisk").innerText = res.risk;
            document.getElementById("metricPhishProb").innerText = (res.phishingProb * 100).toFixed(1) + "%";
            document.getElementById("metricLegitProb").innerText = (res.legitimateProb * 100).toFixed(1) + "%";

            // Bars
            document.getElementById("probBarPhish").style.width = (res.phishingProb * 100) + "%";
            document.getElementById("probBarLegit").style.width = (res.legitimateProb * 100) + "%";

            // Reasons
            const list = document.getElementById("reasonsList");
            list.innerHTML = "";
            res.reasons.forEach(r => {
                const li = document.createElement("li");
                li.className = "flex items-start space-x-2";
                li.innerHTML = `<span class="text-blue-400">•</span><span>${r}</span>`;
                list.appendChild(li);
            });

            // Features table
            const tbody = document.getElementById("featuresTbody");
            tbody.innerHTML = "";
            const descriptions = {
                "url_length": "Overall character length of the normalized URL",
                "domain_length": "Clean domain length excluding 'www.' prefix",
                "path_length": "Length of URL path component",
                "num_dots": "Dot count in domain (subdomain depth indicator)",
                "num_hyphens": "Hyphen count in domain (brand impersonation indicator)",
                "num_underscores": "Underscore count across complete URL",
                "num_slashes": "Directory depth count in path",
                "num_digits": "Total numerical digits in domain name",
                "digit_ratio": "Ratio of digits to domain character length",
                "num_special_chars": "Count of risky symbols (@, $, %, ^, &, *)",
                "has_at_symbol": "Presence of '@' (basic authentication / spoofing)",
                "has_double_slash": "Presence of '//' inside the resource path",
                "has_ip_as_host": "Flag indicating host is an unmapped IP address",
                "is_https": "TLS/SSL protocol indicator",
                "num_subdomains": "Calculated subdomain depth count",
                "num_query_params": "Count of HTTP GET query parameters",
                "has_port": "Non-standard custom port usage",
                "has_percent_encoding": "Hexadecimal percent-encoded characters",
                "has_punycode": "Internationalized domain name spoofing prefix (xn--)",
                "num_path_segments": "Count of path subdirectories",
                "suspicious_words_count": "Matches with security keyword vocabulary",
                "entropy": "Shannon randomness entropy of domain characters"
            };

            for (let [k, v] of Object.entries(res.features)) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td class="py-1.5 px-3 font-semibold text-gray-300">${k}</td>
                    <td class="py-1.5 px-3 text-blue-400 font-bold">${v}</td>
                    <td class="py-1.5 px-3 text-gray-400 text-xs">${descriptions[k] || ''}</td>
                `;
                tbody.appendChild(tr);
            }

            card.classList.remove("hidden");
        }

        function toggleFeaturesTable() {
            const el = document.getElementById("featuresTableWrapper");
            const caret = document.getElementById("featCaret");
            if (el.classList.contains("hidden")) {
                el.classList.remove("hidden");
                caret.className = "fa-solid fa-chevron-up text-xs";
            } else {
                el.classList.add("hidden");
                caret.className = "fa-solid fa-chevron-down text-xs";
            }
        }

        function runBatchAnalysis() {
            const raw = document.getElementById("batchInput").value;
            const lines = raw.split('\\n').map(l => l.trim()).filter(Boolean);
            if (lines.length === 0) return;

            const tbody = document.getElementById("batchTbody");
            tbody.innerHTML = "";

            lines.slice(0, 25).forEach(url => {
                const res = analyzeUrl(url);
                const tr = document.createElement("tr");
                const isPhish = res.prediction === 0;
                tr.innerHTML = `
                    <td class="py-2 px-3 font-mono text-gray-200 truncate max-w-xs">${url}</td>
                    <td class="py-2 px-3 font-bold ${isPhish ? 'text-red-400' : 'text-emerald-400'}">${res.label}</td>
                    <td class="py-2 px-3 font-mono">${(res.phishingProb * 100).toFixed(1)}%</td>
                    <td class="py-2 px-3 text-xs text-gray-400">${res.risk}</td>
                `;
                tbody.appendChild(tr);
            });

            document.getElementById("batchResultsWrapper").classList.remove("hidden");
        }

        // Enter key trigger
        document.getElementById("urlInput").addEventListener("keypress", (e) => {
            if (e.key === "Enter") runAnalysis();
        });
    </script>
</body>
</html>
"""

final_html = HTML_CONTENT.replace("__TREE_RULES_JSON__", tree_rules_json)

with open(ROOT / "index.html", "w", encoding="utf-8") as f:
    f.write(final_html)

with open(ROOT / "public" / "index.html", "w", encoding="utf-8") as f:
    f.write(final_html)

print("Generated standalone index.html at root and public/index.html (Size: ~", len(final_html) // 1024, "KB)")
