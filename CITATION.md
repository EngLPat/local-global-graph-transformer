# Acknowledgments and Related Work

## Architecture Inspiration

This work builds upon the **MeshGraphNets** framework introduced by Pfaff et al. (2021) for learning mesh-based simulations. While MeshGraphNets uses purely local message passing, our FCLGA GraphTransformer extends this with hybrid local-global attention mechanisms for improved long-range dependency capture.

**Key differences from MeshGraphNets:**
- Hybrid local-global attention instead of pure local message passing
- Frequency-controlled global attention application
- Optimized for structural mechanics with composite materials
- Integration with Abaqus FEA workflow

**Original MeshGraphNets paper:**
```bibtex
@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Neural Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter W},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## Citation

If you use this code, please cite our work:

```bibtex
@article{patrignani2025hybrid,
  title={Graph Neural Networks with Hybrid Local-Global Attention for Effective 
         Prediction of Mechanical Response in Structures},
  author={Patrignani, Luca and Pinho, Silvestre T.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2025}
}
```

And consider citing the foundational MeshGraphNets work:

```bibtex
@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Neural Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter W},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## License

MIT License - See LICENSE file
