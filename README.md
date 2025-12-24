# Enhancing automated trading systems knowledge discovery through a novel fusion framework in predicting stocks explainable turning points

A novel machine learning-based stock trading system that combines **Turning Points Labeling (TPL)** with **Multi-stage Self-adaptive Feature Engineering (MSFE)** to generate profitable Buy/Sell trading signals.

## Overview

This repository implements an advanced trading system designed to address the fundamental challenge of "Buy Low, Sell High" in stock market trading. By fusing investor trading principles with adaptive feature engineering, the system provides timely, reliable, and substantially profitable Entry/Exit signals.

**Key Innovation**: Information fusion framework combining TPL and MSFE for enhanced trading knowledge discovery - a unique approach not previously explored in existing literature.

## Key Features

- 📊 **Turning Points Labeling (TPL)**: Novel labeling based on actual investor trading principles
- 🔧 **Multi-stage Self-adaptive Feature Engineering (MSFE)**: Advanced feature discovery pipeline
- 🤖 **Model-Agnostic**: Works with tree-based, linear, instance-based, and neural network models
- 🔍 **Explainable AI Platform**: Regime-aware feature importance and actionable insights
- 📈 **Comprehensive Testing**: Validated on 30 NYSE100 stocks from 2012-2023
- 💰 **Two-Sided Trading**: Greedy Long/Short strategy maximizing both buy and sell opportunities

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Clone Repository
```bash
git clone https://github.com/yourusername/trading-tpl-msfe.git
cd trading-tpl-msfe
```

### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

### Core Dependencies
```txt
python>=3.8
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.40.0
```

### Additional Libraries
```txt
yfinance>=0.1.70        # For downloading stock data
ta>=0.10.0              # Technical analysis indicators
scipy>=1.7.0            # Scientific computing
plotly>=5.0.0           # Interactive visualizations
jupyter>=1.0.0          # Notebook support
tqdm>=4.62.0            # Progress bars
pyyaml>=5.4.0           # Configuration files
```

### Optional Dependencies
```txt
tensorboard>=2.8.0      # For neural network training visualization
optuna>=2.10.0          # Hyperparameter optimization
mlflow>=1.20.0          # Experiment tracking
```

### Full Requirements File

See [requirements.txt](requirements.txt) for the complete list of dependencies with pinned versions.

---

## 🚀 Quick Start
```bash
# Reproduce paper results
python reproduce_paper_results.py --config configs/paper_experiment.yaml

# Run on your own data
python demo.py --stock AAPL --start-date 2020-01-01 --end-date 2023-12-31

# Train custom model
python train.py --config configs/custom_config.yaml
```

---

## 💻 Usage Examples

### Basic Trading System
```python
from trading_system import TPLMSFETrader
import yfinance as yf

# Download data
data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')

# Initialize trader
trader = TPLMSFETrader(
    tpl_params={'window': 20, 'threshold': 0.02},
    msfe_params={'stages': 3, 'adaptive': True}
)

# Train and backtest
trader.fit(data)
results = trader.backtest(data)

print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['drawdown']:.2%}")
print(f"Total Return: {results['return']:.2%}")
```

### Explainable AI Analysis
```python
from explainability import XAIAnalyzer

# Analyze model decisions
analyzer = XAIAnalyzer(trader.model)

# Get regime-specific feature importance
bull_importance = analyzer.get_regime_importance('bull')
bear_importance = analyzer.get_regime_importance('bear')

# Visualize decision process
analyzer.plot_waterfall(trade_date='2023-06-15')
analyzer.plot_feature_importance()
```

---

## 📥 Replication Package

To ensure full reproducibility, we provide:

- ✅ Complete source code
- ✅ Pre-processed datasets with train/val/test splits
- ✅ Trained model checkpoints
- ✅ Experiment configurations
- ✅ Result files and analysis notebooks
- ✅ Step-by-step replication guide

**Download**: [Replication Package (DOI)](#) | [Zenodo Archive](#)

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository**
```bash
   git clone https://github.com/yourusername/trading-tpl-msfe.git
```

2. **Create a feature branch**
```bash
   git checkout -b feature/improvement
```

3. **Make your changes**
   - Write clean, documented code
   - Add tests for new features
   - Update documentation as needed

4. **Commit your changes**
```bash
   git commit -am 'Add new feature: description'
```

5. **Push to your branch**
```bash
   git push origin feature/improvement
```

6. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Describe your changes clearly

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Write clear commit messages
- Add unit tests for new functionality
- Update documentation for API changes
- Ensure all tests pass before submitting PR

### Types of Contributions

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 Code refactoring
- ✅ Test coverage improvements
- 🌐 Translation support

### Code of Conduct

Please be respectful and constructive in all interactions. We're committed to providing a welcoming and inclusive environment for all contributors.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
```
Copyright (c) 2024 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ⚠️ Disclaimer

**Important Notice**: This software is provided for **research and educational purposes only**. 

### Risk Warning

- 📉 **Trading Risk**: Trading in financial markets involves substantial risk of loss
- 💰 **No Guarantees**: Past performance does not guarantee future results
- 🎓 **Educational Use**: This system is intended for academic research and learning
- ⚖️ **Not Financial Advice**: Nothing in this repository constitutes financial advice
- 🔍 **Due Diligence**: Always conduct your own research before making investment decisions
- 👨‍💼 **Consult Professionals**: Consult with qualified financial professionals for investment guidance

### Legal Disclaimer

The authors, contributors, and affiliated institutions:
- Assume **no responsibility** for financial losses incurred through use of this system
- Make **no warranties** regarding the accuracy, reliability, or profitability of the system
- Do **not endorse** any specific trading strategies or investment decisions
- Are **not liable** for any damages resulting from the use of this software

### Academic Use

This software is released in support of open science and reproducible research. Users are encouraged to:
- Validate results independently
- Understand the limitations of the methodology
- Use appropriate risk management in any practical applications
- Cite the original research when building upon this work

**By using this software, you acknowledge that you have read, understood, and agree to this disclaimer.**

---

## 📧 Contact

### Corresponding Author

**[Corresponding Author Name]**  
- 📧 Email: corresponding.author@university.edu  
- 🏛️ Institution: [University/Institution Name]  
- 🔗 Website: [Personal/Lab Website]

### Co-Authors

**[Author 2 Name]**  
- 📧 Email: author2@university.edu  
- 💻 GitHub: [@author2](https://github.com/author2)

**[Author 3 Name]**  
- 📧 Email: author3@university.edu  
- 🔗 LinkedIn: [Author 3 LinkedIn](https://linkedin.com/in/author3)

### For Specific Inquiries

- **Paper Content & Methodology**: [author1@university.edu]
- **Code Implementation & Technical Issues**: [author2@university.edu]
- **Data Access & Replication**: [author3@university.edu]
- **Collaboration Opportunities**: [corresponding.author@university.edu]

### Reporting Issues

Found a bug or have a suggestion? Please:
1. Check existing [GitHub Issues](https://github.com/yourusername/trading-tpl-msfe/issues)
2. Create a new issue with detailed description
3. Include steps to reproduce (for bugs)

---

## 🙏 Acknowledgments

This research was supported by:

- **Funding**: [Grant Number], [Funding Agency Name]
- **Computational Resources**: [Institution HPC / Cloud Provider]
- **Data Providers**: Yahoo Finance, New York Stock Exchange
- **Open Source Community**: 
  - scikit-learn, XGBoost, LightGBM developers
  - SHAP library for explainable AI
  - Python scientific computing ecosystem

Special thanks to:
- Reviewers for their valuable feedback
- Conference/Workshop participants for insightful discussions
- [Any specific individuals or groups to acknowledge]

---

## 📊 Supplementary Materials

- **Appendix A**: Extended ablation studies - [PDF Link](#)
- **Appendix B**: Additional stock results - [PDF Link](#)
- **Appendix C**: Hyperparameter sensitivity analysis - [PDF Link](#)
- **Interactive Dashboard**: Explore results online - [Web Link](#)
- **Video Presentation**: Conference talk - [YouTube Link](#)
- **Slides**: Presentation slides - [PDF Link](#)

---

## 🔄 Version History & Updates

- **v1.0.0** (2024-XX-XX): Initial release with paper acceptance
- **v1.0.1** (2024-XX-XX): Bug fixes and documentation improvements
- **v1.1.0** (2024-XX-XX): Added support for additional ML models
- **Latest**: See [CHANGELOG.md](CHANGELOG.md) for detailed version history

### Upcoming Features

- [ ] Real-time trading integration
- [ ] Additional technical indicators
- [ ] Multi-asset portfolio optimization
- [ ] Enhanced visualization dashboard
- [ ] GPU acceleration for neural networks

---

## 📚 Related Publications

If you're interested in this work, you may also find these related publications useful:

1. [Related Paper 1] - [Link](#)
2. [Related Paper 2] - [Link](#)
3. [Related Paper 3] - [Link](#)

---

## 🌟 Star History

If you find this repository useful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/trading-tpl-msfe&type=Date)](https://star-history.com/#yourusername/trading-tpl-msfe&Date)

---

*Last Updated: [Current Date]*

**Repository**: https://github.com/yourusername/trading-tpl-msfe  
**Paper**: https://doi.org/your-doi-link  
**Documentation**: https://trading-tpl-msfe.readthedocs.io

---

<div align="center">

**Made with ❤️ for the Research Community**

If this work helps your research, please cite our paper and give us a ⭐!

</div>
