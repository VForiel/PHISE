# API References

PHISE include a core set of classes to model the main entities manipulated by the library and several modules providing additional functionalities.

The classes are nested as follows:
- Context
    - Interferometer
        - Telescope
        - KernelNuller
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
classes/mmi.md
classes/target.md
classes/telescope.md
classes/interferometer.md
classes/camera.md
classes/companion.md


:caption: Modules
modules/test_statistics.md
```