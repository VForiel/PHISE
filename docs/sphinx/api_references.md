# 🧩 API References

PHISE include a core set of classes to model the main entities manipulated by the library and several modules providing additional functionalities.

The classes are nested as follows:
- Context
    - Interferometer
        - Telescope
        - SuperKN
        - Camera
    - Target
        - Companion

As building such hierarchy of objects can be complex, please refer to the « Getting Started » guide for practical examples where you will get context templates based on VLTI and LIFE-like instruments.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Classes

classes/context.md
classes/kernel_nuller.md
classes/target.md
classes/telescope.md
classes/interferometer.md
classes/camera.md
classes/companion.md
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Modules

modules/coordinates.md
modules/phase.md
modules/signals.md
modules/ml.md
modules/test_statistics.md
```