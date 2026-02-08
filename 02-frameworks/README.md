# Deep Learning Frameworks

A collection of comprehensive tutorials and implementations for popular deep learning frameworks.

## 📚 Contents

### PyTorch Tutorial
**Location**: `notebooks/tutorial - pytorch.ipynb`

A complete, production-ready PyTorch tutorial covering:
- PyTorch fundamentals (tensors, autograd, nn.Module)
- Computer vision with CIFAR-10 and Fashion-MNIST
- LeNet-5 and AlexNet implementations
- Complete training, validation, and evaluation pipelines
- Extensive visualizations with matplotlib and seaborn
- Model saving/loading and best practices

**Features**:
- ✅ 800+ lines of documented code
- ✅ Every function parameter explained
- ✅ Mermaid workflow diagrams
- ✅ Comprehensive evaluation metrics
- ✅ Performance comparison dashboards
- ✅ Real-world examples

See `notebooks/README.md` for detailed information.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

### Launch Jupyter

```bash
jupyter notebook "notebooks/tutorial - pytorch.ipynb"
```

## 📁 Project Structure

```
02-frameworks/
├── README.md                  # This file
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Lock file
├── notebooks/                # Jupyter notebooks
│   ├── README.md            # Notebook documentation
│   └── tutorial - pytorch.ipynb
├── data/                     # Dataset storage (auto-created)
└── saved_models/             # Model checkpoints (created during training)
```

## 🎯 Learning Path

**For PyTorch beginners**:
1. Start with `tutorial - pytorch.ipynb`
2. Follow along with each section sequentially
3. Experiment with the code and parameters
4. Complete the training exercises

**Expected time**: 3-5 hours for complete tutorial

## 📊 What You'll Build

By the end of the PyTorch tutorial, you'll have:

1. **Trained Models**
   - LeNet-5 on CIFAR-10 (~65-70% accuracy)
   - AlexNet on CIFAR-10 (~75-80% accuracy)

2. **Comprehensive Analysis**
   - Training history plots
   - Confusion matrices
   - Per-class performance metrics
   - Model comparison dashboards

3. **Practical Skills**
   - Data loading and preprocessing
   - Model architecture design
   - Training loop implementation
   - Model evaluation and visualization
   - Checkpoint saving/loading

## 🛠️ Requirements

- **Python**: 3.12 or higher
- **GPU**: Recommended (CUDA-capable) but not required
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Disk Space**: ~2GB for datasets and models

## 📦 Dependencies

Core libraries:
- `torch >= 2.10.0` - Deep learning framework
- `torchvision >= 0.20.0` - Vision datasets and models
- `matplotlib >= 3.10.0` - Plotting
- `seaborn >= 0.13.2` - Statistical visualizations
- `jupyter >= 1.1.1` - Interactive notebooks
- `tqdm >= 4.67.1` - Progress bars
- `scikit-learn >= 1.6.1` - Metrics and evaluation

See `pyproject.toml` for complete list.

## 💻 Hardware Recommendations

### Minimum
- CPU: Any modern multi-core processor
- RAM: 8GB
- GPU: Not required (will use CPU)
- Training time: 1-2 hours

### Recommended
- CPU: Modern multi-core processor (4+ cores)
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM (CUDA support)
- Training time: 15-30 minutes

## 🎓 Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of:
  - Lists, dictionaries, functions, classes
  - NumPy arrays (helpful but not essential)
  - Basic mathematics (matrices, derivatives concept)

### No Prior Experience Needed With
- PyTorch or deep learning frameworks
- Neural networks
- Computer vision
- CUDA programming

## 📈 Next Steps

After completing the PyTorch tutorial:

1. **Advanced PyTorch**
   - Implement ResNet, VGG, or EfficientNet
   - Try transfer learning with pre-trained models
   - Experiment with custom datasets

2. **Other Frameworks** (coming soon)
   - TensorFlow/Keras tutorial
   - JAX tutorial
   - ONNX for model deployment

3. **Specialized Topics**
   - Natural Language Processing with transformers
   - Generative models (VAE, GAN)
   - Reinforcement Learning

## 🐛 Troubleshooting

### Common Issues

**Import errors**:
```bash
# Reinstall dependencies
uv sync --force
```

**CUDA not available**:
- Check GPU compatibility: https://pytorch.org/get-started/locally/
- Reinstall PyTorch with correct CUDA version

**Out of memory**:
- Reduce batch size in the notebook
- Use smaller model
- Close other applications

**Slow performance**:
- Ensure GPU is being used (check notebook output)
- Increase batch size if memory allows
- Use fewer workers in DataLoader on Windows

## 📚 Additional Resources

### Official Documentation
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Learning Resources
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [FastAI Course](https://course.fast.ai/)

### Community
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)
- [PyTorch Discord](https://discord.gg/pytorch)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest improvements or new tutorials
- Add documentation
- Share your training results

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- CIFAR-10 and Fashion-MNIST dataset creators
- Open source community for tools and libraries

---

**Happy Learning! 🚀**
